//! Live waveform + mel-spec dev testbed for compatible digital stethoscopes.
//!
//! Pure Rust egui app. BLE work runs on a background tokio runtime,
//! decoded PCM samples flow over a crossbeam channel into the UI thread,
//! which repaints at ~60 fps from a fixed-size ring buffer.
//!
//! Build & run: `cargo run -p stetho-ui --release`.

mod inference;

use crossbeam_channel::{Receiver, Sender, TryRecvError};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use inference::{MurmurEngine, S3Engine, N_FRAMES_MURMUR, N_FRAMES_S3};
use std::collections::VecDeque;
use std::fs;
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use stetho_core::dsp::mel::{LogMelSpectrogram, N_MELS};
use stetho_core::dsp::presets::{EqPreset, FilterChain};
use stetho_core::{AdpcmDecoder, AUDIO_SAMPLE_RATE_HZ};
use tracing::{error, info};

const WAVEFORM_SECONDS: usize = 5;
const WAVEFORM_LEN: usize = AUDIO_SAMPLE_RATE_HZ as usize * WAVEFORM_SECONDS;
const SPECTROGRAM_FRAMES: usize = 128;
/// Legacy default operating threshold for the 4 s model, used only when a
/// package ships no `murmur_threshold` metadata. The shipped 5 s ensemble
/// carries its own calibrated threshold in the sidecar (`murmur_decision_config`
/// reads it), so this constant is just the backward-compatible fallback.
const MURMUR_RECORDING_MEAN_THRESHOLD: f32 = 0.49331352;
const DEFAULT_MODEL_DOWNLOAD_URL: &str =
    "https://github.com/ShaneBreazeale/openstetho/releases/latest/download/MurmurCNN.mlpackage.zip";
const CURRENT_MODEL_RELEASE_LABEL: &str = "v0.4.0-murmur-ensemble";
const CURRENT_LOCAL_MURMUR_RUN: &str = "release-circor-5s-spec93-top4-v1";
const CURRENT_LOCAL_S3_RUN: &str = "s3_circor_v10";
const DEFAULT_EQ_PRESET: EqPreset = EqPreset::Cardiac;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MurmurAggregation {
    Mean,
    TopKMean(usize),
}

impl MurmurAggregation {
    fn label(self) -> String {
        match self {
            Self::Mean => "mean".to_string(),
            Self::TopKMean(k) => format!("top{k}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MurmurDecisionConfig {
    aggregation: MurmurAggregation,
    threshold: f32,
}

impl Default for MurmurDecisionConfig {
    fn default() -> Self {
        Self {
            aggregation: MurmurAggregation::Mean,
            threshold: MURMUR_RECORDING_MEAN_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConnState {
    Idle,
    Scanning,
    Connecting,
    Streaming,
    Disconnected,
    Error,
}

/// Messages from the BLE worker to the UI thread.
enum Msg {
    State(ConnState),
    Status(String),
    Battery(u8),
    Samples(Vec<i16>),
    MelFrame([f32; N_MELS]),
    MurmurProb(f32),
    S3Prob(f32),
    ModelDownloadFinished(Result<PathBuf, String>),
}

/// Messages from the UI thread to the BLE worker.
enum Cmd {
    Connect,
    Disconnect,
    SetPreset(EqPreset),
    SetGainDb(f32),
    ResetCodec,
    SetModel(PathBuf),
}

struct App {
    rx: Receiver<Msg>,
    msg_tx: Sender<Msg>,
    cmd_tx: Sender<Cmd>,

    state: ConnState,
    status: String,
    battery: Option<u8>,

    /// Ring buffer of recent PCM samples.
    samples: Vec<i16>,
    /// Logical write head, monotonic — we keep only the last WAVEFORM_LEN.
    total_samples: u64,

    /// Rolling mel-spec frames.
    spec: Vec<[f32; N_MELS]>,

    preset: EqPreset,
    gain_db: f32,
    show_spec: bool,
    murmur_prob: Option<f32>,
    murmur_history: Vec<f32>,
    murmur_config: MurmurDecisionConfig,
    s3_prob: Option<f32>,
    s3_history: Vec<f32>,
    model_path: Option<PathBuf>,
    available_models: Vec<PathBuf>,
    model_download_busy: bool,
}

impl App {
    fn new(
        rx: Receiver<Msg>,
        msg_tx: Sender<Msg>,
        cmd_tx: Sender<Cmd>,
        model_path: Option<PathBuf>,
        available_models: Vec<PathBuf>,
    ) -> Self {
        Self {
            rx,
            msg_tx,
            cmd_tx,
            state: ConnState::Idle,
            status: "idle".into(),
            battery: None,
            samples: Vec::with_capacity(WAVEFORM_LEN),
            total_samples: 0,
            spec: Vec::with_capacity(SPECTROGRAM_FRAMES),
            preset: DEFAULT_EQ_PRESET,
            gain_db: 0.0,
            show_spec: true,
            murmur_prob: None,
            murmur_history: Vec::new(),
            murmur_config: murmur_decision_config(model_path.as_ref()),
            s3_prob: None,
            s3_history: Vec::with_capacity(60),
            model_path,
            available_models,
            model_download_busy: false,
        }
    }

    fn reset_murmur_aggregation(&mut self) {
        self.murmur_prob = None;
        self.murmur_history.clear();
    }

    fn murmur_decision_prob(&self) -> Option<f32> {
        if self.murmur_history.is_empty() {
            return None;
        }
        match self.murmur_config.aggregation {
            MurmurAggregation::Mean => {
                let sum: f32 = self.murmur_history.iter().sum();
                Some(sum / self.murmur_history.len() as f32)
            }
            MurmurAggregation::TopKMean(k) => {
                let mut probs = self.murmur_history.clone();
                probs.sort_by(|a, b| b.total_cmp(a));
                let n = probs.len().min(k.max(1));
                Some(probs[..n].iter().sum::<f32>() / n as f32)
            }
        }
    }

    fn toolbar_status(&self) -> Option<&str> {
        let status = self.status.as_str();
        if matches!(status, "idle" | "streaming" | "disconnected")
            || status.starts_with("model ")
            || status.starts_with("S3 model loaded")
        {
            return None;
        }
        Some(status)
    }

    fn drain(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(Msg::State(s)) => {
                    if s == ConnState::Streaming && self.state != ConnState::Streaming {
                        self.reset_murmur_aggregation();
                    }
                    self.state = s;
                }
                Ok(Msg::Status(s)) => self.status = s,
                Ok(Msg::Battery(b)) => self.battery = Some(b),
                Ok(Msg::Samples(buf)) => {
                    self.total_samples += buf.len() as u64;
                    if buf.len() >= WAVEFORM_LEN {
                        self.samples.clear();
                        self.samples
                            .extend_from_slice(&buf[buf.len() - WAVEFORM_LEN..]);
                    } else {
                        let overflow =
                            (self.samples.len() + buf.len()).saturating_sub(WAVEFORM_LEN);
                        if overflow > 0 {
                            self.samples.drain(..overflow);
                        }
                        self.samples.extend_from_slice(&buf);
                    }
                }
                Ok(Msg::MelFrame(f)) => {
                    if self.spec.len() == SPECTROGRAM_FRAMES {
                        self.spec.remove(0);
                    }
                    self.spec.push(f);
                }
                Ok(Msg::MurmurProb(p)) => {
                    self.murmur_prob = Some(p);
                    self.murmur_history.push(p);
                }
                Ok(Msg::S3Prob(p)) => {
                    self.s3_prob = Some(p);
                    if self.s3_history.len() == 60 {
                        self.s3_history.remove(0);
                    }
                    self.s3_history.push(p);
                }
                Ok(Msg::ModelDownloadFinished(result)) => {
                    self.model_download_busy = false;
                    match result {
                        Ok(path) => {
                            self.status = match display_label(&path) {
                                Some(label) => format!("model downloaded: {label}"),
                                None => "model downloaded".to_string(),
                            };
                            if !self.available_models.iter().any(|p| p == &path) {
                                self.available_models.push(path.clone());
                                self.available_models.sort();
                                self.available_models.dedup();
                            }
                            self.model_path = Some(path.clone());
                            self.murmur_config = murmur_decision_config(self.model_path.as_ref());
                            self.reset_murmur_aggregation();
                            self.s3_prob = None;
                            self.s3_history.clear();
                            let _ = self.cmd_tx.send(Cmd::SetModel(path));
                        }
                        Err(err) => {
                            self.status = format!("model download failed: {err}");
                        }
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.state = ConnState::Error;
                    self.status = "BLE worker died".into();
                    break;
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain();
        ctx.request_repaint_after(Duration::from_millis(16));

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("stetho-ui");
                ui.separator();
                let (label, can_act) = match self.state {
                    ConnState::Idle | ConnState::Disconnected | ConnState::Error => {
                        ("Connect", true)
                    }
                    ConnState::Scanning => ("Scanning…", false),
                    ConnState::Connecting => ("Connecting…", false),
                    ConnState::Streaming => ("Disconnect", true),
                };
                if ui.add_enabled(can_act, egui::Button::new(label)).clicked() {
                    let cmd = if matches!(self.state, ConnState::Streaming) {
                        Cmd::Disconnect
                    } else {
                        Cmd::Connect
                    };
                    let _ = self.cmd_tx.send(cmd);
                }
                if let Some(status) = self.toolbar_status() {
                    ui.separator();
                    ui.label(status);
                }
                if let Some(b) = self.battery {
                    ui.separator();
                    ui.label(format!("battery: {b}%"));
                }
                if matches!(self.state, ConnState::Streaming) {
                    ui.separator();
                    ui.label(format!("rx: {} samples", self.total_samples));
                }
                if let Some(p) = self.murmur_decision_prob() {
                    ui.separator();
                    let color = if p >= self.murmur_config.threshold {
                        egui::Color32::from_rgb(220, 60, 60)
                    } else {
                        egui::Color32::from_rgb(80, 180, 80)
                    };
                    let latest = self.murmur_prob.unwrap_or(p);
                    ui.colored_label(
                        color,
                        format!(
                            "murmur {}: {:.0}% latest {:.0}%",
                            self.murmur_config.aggregation.label(),
                            p * 100.0,
                            latest * 100.0
                        ),
                    );
                }
                if let Some(p) = self.s3_prob {
                    ui.separator();
                    // S3 is calibrated against synthetic injection; on real
                    // audio the meaningful operating window is roughly
                    // 0.93 (high sensitivity) to 0.99 (high specificity)
                    // per docs/real_validation_results.md. Color tracks that.
                    let color = if p > 0.93 {
                        egui::Color32::from_rgb(220, 60, 60)
                    } else if p > 0.5 {
                        egui::Color32::from_rgb(220, 180, 60)
                    } else {
                        egui::Color32::from_rgb(80, 180, 80)
                    };
                    ui.colored_label(color, format!("S3: {:.0}%", p * 100.0));
                }
                if !self.available_models.is_empty() {
                    ui.separator();
                    ui.label("model");
                    let current_label = self
                        .model_path
                        .as_ref()
                        .and_then(|p| display_label(p))
                        .unwrap_or_else(|| "(none)".to_string());
                    let mut chosen: Option<PathBuf> = None;
                    egui::ComboBox::from_id_salt("model_picker")
                        .selected_text(current_label)
                        .show_ui(ui, |ui| {
                            for path in &self.available_models {
                                let label = display_label(path)
                                    .unwrap_or_else(|| path.display().to_string());
                                let selected = self.model_path.as_deref() == Some(path);
                                if ui.selectable_label(selected, label).clicked() {
                                    chosen = Some(path.clone());
                                }
                            }
                        });
                    if let Some(p) = chosen {
                        if Some(&p) != self.model_path.as_ref() {
                            self.model_path = Some(p.clone());
                            self.murmur_config = murmur_decision_config(self.model_path.as_ref());
                            self.reset_murmur_aggregation();
                            self.s3_prob = None;
                            self.s3_history.clear();
                            let _ = self.cmd_tx.send(Cmd::SetModel(p));
                        }
                    }
                } else if let Some(mp) = &self.model_path {
                    ui.separator();
                    let label = mp.file_name().and_then(|s| s.to_str()).unwrap_or("model");
                    ui.weak(format!("model: {label}"));
                }
                ui.separator();
                let dl_label = if self.model_download_busy {
                    "Downloading model..."
                } else if self.model_path.is_some() {
                    "Update model"
                } else {
                    "Download model"
                };
                if ui
                    .add_enabled(!self.model_download_busy, egui::Button::new(dl_label))
                    .clicked()
                {
                    self.model_download_busy = true;
                    self.status = "downloading model".into();
                    spawn_model_download(self.msg_tx.clone());
                }
            });
            ui.horizontal(|ui| {
                ui.label("preset");
                let mut chosen = self.preset;
                egui::ComboBox::from_id_salt("preset")
                    .selected_text(format!("{:?}", self.preset))
                    .show_ui(ui, |ui| {
                        for p in [
                            EqPreset::None,
                            EqPreset::Wide,
                            EqPreset::Cardiac,
                            EqPreset::Pulmonary,
                        ] {
                            ui.selectable_value(&mut chosen, p, format!("{p:?}"));
                        }
                    });
                if chosen != self.preset {
                    self.preset = chosen;
                    let _ = self.cmd_tx.send(Cmd::SetPreset(self.preset));
                }
                ui.separator();
                ui.label("gain");
                let prev_gain = self.gain_db;
                ui.add(egui::Slider::new(&mut self.gain_db, -12.0..=48.0).suffix(" dB"));
                if (self.gain_db - prev_gain).abs() > 0.01 {
                    let _ = self.cmd_tx.send(Cmd::SetGainDb(self.gain_db));
                }
                ui.separator();
                ui.checkbox(&mut self.show_spec, "spectrogram");
                ui.separator();
                if ui.button("Reset codec").clicked() {
                    let _ = self.cmd_tx.send(Cmd::ResetCodec);
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("waveform — last 5 s");
            let peak: f64 = self
                .samples
                .iter()
                .map(|s| (*s as f64).abs())
                .fold(0.0, f64::max);
            // Auto-scale: 1.5× peak with a 500-sample floor so the axis
            // doesn't visually collapse during near-silence. Cap at 32k
            // (i16 range) so loud passages clip predictably.
            let y_bound = (peak * 1.5).clamp(500.0, 32_000.0);
            let pts: PlotPoints = self
                .samples
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let t = (i as f64) / AUDIO_SAMPLE_RATE_HZ as f64;
                    [t, *s as f64]
                })
                .collect();
            Plot::new("waveform")
                .height(220.0)
                .allow_zoom(false)
                .allow_drag(false)
                .allow_scroll(false)
                .show_axes([true, true])
                .include_y(-y_bound)
                .include_y(y_bound)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(pts));
                });

            if self.show_spec {
                ui.separator();
                ui.heading("log-mel spectrogram — newest on the right");
                let frames = &self.spec;
                let frame_count = frames.len().max(1);
                let h = 160.0;
                let (rect, _) = ui
                    .allocate_exact_size(egui::vec2(ui.available_width(), h), egui::Sense::hover());
                let painter = ui.painter_at(rect);
                if !frames.is_empty() {
                    let cell_w = rect.width() / frame_count as f32;
                    let cell_h = rect.height() / N_MELS as f32;
                    for (fi, frame) in frames.iter().enumerate() {
                        for (mi, &v) in frame.iter().enumerate() {
                            let x0 = rect.left() + fi as f32 * cell_w;
                            let y0 = rect.top() + (N_MELS - 1 - mi) as f32 * cell_h;
                            let r = egui::Rect::from_min_size(
                                egui::pos2(x0, y0),
                                egui::vec2(cell_w, cell_h),
                            );
                            painter.rect_filled(r, 0.0, colormap(v));
                        }
                    }
                } else {
                    painter.rect_filled(rect, 0.0, egui::Color32::from_rgb(20, 20, 30));
                }
            }
        });
    }
}

/// Number of past raw-dB frames retained for the display-side rolling
/// normalization. At hop=256, sr=4 kHz that's 64 ms/frame → ~78 frames
/// per 5 s of audio. Pick a window long enough to span at least one
/// full cardiac cycle so S1 and S2 fall inside the same reference.
const DB_HISTORY_FRAMES: usize = 80;

/// Convert one raw dB mel frame into a display value in roughly [0, 1].
///
/// Design choices for you to make in the body:
///   1. Rolling reference: max over `history`? max-per-bin? 95th
///      percentile? Global EMA? Tradeoff: max is sharpest but spiky;
///      percentile rejects outliers but costs a sort.
///   2. Dynamic range: subtract reference, clip to `[-DR, 0]` dB, then
///      remap to `[0, 1]`. Conventional `DR` = 60–80 dB. Smaller `DR`
///      gives more contrast but loses quiet detail.
///   3. Update order: push to history before or after computing the
///      reference? Including the current frame stabilises the very
///      first frames; excluding avoids the current loud beat
///      flattening itself.
///
/// Return `[0.0, 1.0]`-ish floats — `colormap()` clamps internally.
fn display_normalize(
    db_frame: [f32; N_MELS],
    history: &mut VecDeque<[f32; N_MELS]>,
) -> [f32; N_MELS] {
    const DYN_RANGE_DB: f32 = 60.0;
    const FLOOR_DB: f32 = -90.0;

    let mut out = [0.0_f32; N_MELS];
    for i in 0..N_MELS {
        let mut ref_db = FLOOR_DB;
        for past in history.iter() {
            if past[i] > ref_db {
                ref_db = past[i];
            }
        }
        let rel = (db_frame[i] - ref_db).clamp(-DYN_RANGE_DB, 0.0);
        out[i] = (rel + DYN_RANGE_DB) / DYN_RANGE_DB;
    }

    if history.len() == DB_HISTORY_FRAMES {
        history.pop_front();
    }
    history.push_back(db_frame);
    out
}

/// Map a display value in [0, 1] to an inferno-ish colormap.
fn colormap(z: f32) -> egui::Color32 {
    let t = z.clamp(0.0, 1.0);
    // Smooth inferno-ish: dark → purple → red → yellow → white.
    let stops = [
        (0.00, [0x00, 0x00, 0x05]),
        (0.25, [0x40, 0x0a, 0x67]),
        (0.50, [0x9a, 0x23, 0x6a]),
        (0.75, [0xe5, 0x6b, 0x2d]),
        (1.00, [0xfc, 0xff, 0xa4]),
    ];
    for w in stops.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if t >= lo.0 && t <= hi.0 {
            let f = ((t - lo.0) / (hi.0 - lo.0)).clamp(0.0, 1.0);
            let r = (lo.1[0] as f32 + (hi.1[0] as f32 - lo.1[0] as f32) * f) as u8;
            let g = (lo.1[1] as f32 + (hi.1[1] as f32 - lo.1[1] as f32) * f) as u8;
            let b = (lo.1[2] as f32 + (hi.1[2] as f32 - lo.1[2] as f32) * f) as u8;
            return egui::Color32::from_rgb(r, g, b);
        }
    }
    egui::Color32::BLACK
}

// ─── BLE worker ──────────────────────────────────────────────────────────────

fn spawn_ble_worker(tx: Sender<Msg>, cmd_rx: Receiver<Cmd>, model_path: Option<PathBuf>) {
    std::thread::Builder::new()
        .name("ble-worker".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    error!("tokio init failed: {e}");
                    return;
                }
            };
            rt.block_on(run_worker(tx, cmd_rx, model_path));
        })
        .expect("spawn ble worker");
}

/// State shared between the UI command consumer and the active streaming task.
#[derive(Clone)]
struct SessionConfig {
    preset: Arc<std::sync::Mutex<EqPreset>>,
    gain: Arc<std::sync::Mutex<f32>>,
    reset_flag: Arc<std::sync::atomic::AtomicBool>,
    model_path: Arc<std::sync::Mutex<Option<PathBuf>>>,
    model_dirty: Arc<std::sync::atomic::AtomicBool>,
}

impl SessionConfig {
    fn new(model_path: Option<PathBuf>) -> Self {
        Self {
            preset: Arc::new(std::sync::Mutex::new(DEFAULT_EQ_PRESET)),
            gain: Arc::new(std::sync::Mutex::new(1.0)),
            reset_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            model_path: Arc::new(std::sync::Mutex::new(model_path)),
            model_dirty: Arc::new(std::sync::atomic::AtomicBool::new(true)),
        }
    }
}

async fn run_worker(tx: Sender<Msg>, cmd_rx: Receiver<Cmd>, model_path: Option<PathBuf>) {
    let tx = Arc::new(tx);
    let cfg = SessionConfig::new(model_path);
    let mut current_task: Option<tokio::task::JoinHandle<()>> = None;
    let mut stop_tx: Option<tokio::sync::watch::Sender<bool>> = None;
    loop {
        let cmd = match tokio::task::spawn_blocking({
            let cmd_rx = cmd_rx.clone();
            move || cmd_rx.recv()
        })
        .await
        {
            Ok(Ok(c)) => c,
            _ => break,
        };
        match cmd {
            Cmd::Connect => {
                if current_task.is_some() {
                    continue;
                }
                let (s_tx, s_rx) = tokio::sync::watch::channel(false);
                stop_tx = Some(s_tx);
                let tx_cl = tx.clone();
                let cfg_cl = cfg.clone();
                current_task = Some(tokio::spawn(async move {
                    if let Err(e) = stream_session(tx_cl.clone(), s_rx, cfg_cl).await {
                        let _ = tx_cl.send(Msg::Status(format!("error: {e:#}")));
                        let _ = tx_cl.send(Msg::State(ConnState::Error));
                    }
                }));
            }
            Cmd::Disconnect => {
                // Optimistically tell the UI we're disconnected so the
                // button toggles immediately. CoreBluetooth's disconnect
                // can take seconds or hang outright on macOS; we don't
                // want to block the user on it.
                let _ = tx.send(Msg::State(ConnState::Disconnected));
                let _ = tx.send(Msg::Status("disconnected".into()));
                if let Some(s) = stop_tx.take() {
                    let _ = s.send(true);
                }
                if let Some(t) = current_task.take() {
                    // Reap the background session with a hard timeout so
                    // a hung CoreBluetooth call can't wedge the worker.
                    if tokio::time::timeout(Duration::from_secs(3), t)
                        .await
                        .is_err()
                    {
                        let _ = tx.send(Msg::Status(
                            "disconnect timed out — old session abandoned".into(),
                        ));
                    }
                }
            }
            Cmd::SetPreset(p) => {
                *cfg.preset.lock().unwrap() = p;
            }
            Cmd::SetGainDb(db) => {
                *cfg.gain.lock().unwrap() = 10f32.powf(db / 20.0);
            }
            Cmd::ResetCodec => {
                cfg.reset_flag
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
            Cmd::SetModel(p) => {
                *cfg.model_path.lock().unwrap() = Some(p);
                cfg.model_dirty
                    .store(true, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

async fn stream_session(
    tx: Arc<Sender<Msg>>,
    mut stop: tokio::sync::watch::Receiver<bool>,
    cfg: SessionConfig,
) -> anyhow::Result<()> {
    use btleplug::api::Peripheral as _;
    use futures::StreamExt;
    use stetho_core::ble;

    let _ = tx.send(Msg::State(ConnState::Scanning));
    let _ = tx.send(Msg::Status("scanning".into()));

    let mgr = ble::manager().await?;
    let adapter = ble::default_adapter(&mgr).await?;
    let _ = ble::scan(&adapter, Duration::from_secs(8)).await?;
    let peripheral = ble::resolve_peripheral(&adapter, "eko core").await?;

    let _ = tx.send(Msg::State(ConnState::Connecting));
    let _ = tx.send(Msg::Status("connecting".into()));
    ble::connect_and_discover(&peripheral).await?;
    if let Ok(Some(b)) = ble::read_battery(&peripheral).await {
        let _ = tx.send(Msg::Battery(b));
    }
    ble::subscribe_core2_adpcm(&peripheral).await?;

    let _ = tx.send(Msg::State(ConnState::Streaming));
    let _ = tx.send(Msg::Status("streaming".into()));
    let mut notifications = peripheral.notifications().await?;
    let mut decoder = AdpcmDecoder::new();
    let mut mel = LogMelSpectrogram::new(AUDIO_SAMPLE_RATE_HZ as f32);
    let mut samples_buf: Vec<i16> = Vec::with_capacity(2048);
    let mut mel_in: Vec<f32> = Vec::with_capacity(2048);
    let mut mel_out: Vec<[f32; N_MELS]> = Vec::with_capacity(8);
    let mut mel_db_out: Vec<[f32; N_MELS]> = Vec::with_capacity(8);
    let mut db_history: VecDeque<[f32; N_MELS]> = VecDeque::with_capacity(DB_HISTORY_FRAMES);
    let mut last_flush = Instant::now();

    let mut active_preset = *cfg.preset.lock().unwrap();
    let mut chain = FilterChain::new(active_preset);

    // Helper closure: load the path currently in cfg.model_path. Used
    // both at session start and any time the UI changes model selection
    // mid-stream (signalled via cfg.model_dirty).
    let load_current_model = |path: Option<&PathBuf>| -> Option<MurmurEngine> {
        match path {
            Some(p) => match MurmurEngine::load(p, murmur_frame_count(p)) {
                Ok(e) => {
                    let status = match display_label(p) {
                        Some(label) => format!("model loaded: {label}"),
                        None => "model loaded".to_string(),
                    };
                    let _ = tx.send(Msg::Status(status));
                    Some(e)
                }
                Err(err) => {
                    let _ = tx.send(Msg::Status(format!("model load failed: {err}")));
                    None
                }
            },
            None => None,
        }
    };

    // S3 model is opportunistic — prefer an `S3CNN_v2.mlpackage` sibling
    // from the release bundle, then fall back to the current local S3 run.
    // Absent both, the UI runs murmur-only as before.
    let load_s3_model = |murmur_path: Option<&PathBuf>| -> Option<(S3Engine, PathBuf)> {
        let p = murmur_path?;
        let dir = p.parent()?;
        let candidates = [
            dir.join("S3CNN_v2.mlpackage"),
            dir.join("S3CNN.mlpackage"),
            fallback_s3_model_path(),
        ];
        for candidate in candidates {
            if !candidate.exists() {
                continue;
            }
            return match S3Engine::load(&candidate, N_FRAMES_S3) {
                Ok(e) => {
                    let _ = tx.send(Msg::Status("S3 model loaded".to_string()));
                    Some((e, candidate))
                }
                Err(err) => {
                    let _ = tx.send(Msg::Status(format!("S3 model load failed: {err}")));
                    None
                }
            };
        }
        None
    };

    let initial_murmur_path = {
        let p = cfg.model_path.lock().unwrap().clone();
        cfg.model_dirty
            .store(false, std::sync::atomic::Ordering::Relaxed);
        p
    };
    let mut engine: Option<MurmurEngine> = load_current_model(initial_murmur_path.as_ref());
    let mut s3_engine_pair: Option<(S3Engine, PathBuf)> =
        load_s3_model(initial_murmur_path.as_ref());

    loop {
        tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_ok() && *stop.borrow() { break; }
            }
            n = notifications.next() => {
                let Some(notif) = n else { break };

                if cfg.reset_flag.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    decoder.reset();
                }

                if cfg.model_dirty.swap(false, std::sync::atomic::Ordering::Relaxed) {
                    let path = cfg.model_path.lock().unwrap().clone();
                    engine = load_current_model(path.as_ref());
                    s3_engine_pair = load_s3_model(path.as_ref());
                }

                // Refresh EQ chain on preset change.
                let want_preset = *cfg.preset.lock().unwrap();
                if want_preset != active_preset {
                    active_preset = want_preset;
                    chain = FilterChain::new(active_preset);
                }
                let gain = *cfg.gain.lock().unwrap();

                samples_buf.clear();
                decoder.decode_packet(&notif.value, &mut samples_buf);

                // Raw f32 buffer — what the model sees. Training pipeline
                // feeds raw audio straight to mel-spec, so we mirror that.
                mel_in.clear();
                mel_in.extend(samples_buf.iter().map(|s| *s as f32));

                // Display buffer — a copy with the EQ preset and gain
                // applied. Cosmetic only; never reaches the model.
                let mut display_buf: Vec<f32> = mel_in.clone();
                if let Some(ch) = chain.as_mut() {
                    ch.process_block(&mut display_buf);
                }
                if (gain - 1.0).abs() > 1e-3 {
                    for v in display_buf.iter_mut() {
                        *v *= gain;
                    }
                }

                // Push the post-EQ samples to the UI waveform plot.
                let out: Vec<i16> = display_buf
                    .iter()
                    .map(|v| v.clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                    .collect();
                let _ = tx.send(Msg::Samples(out));

                // Mel-spec + Core ML inference run on RAW samples so the
                // model sees exactly what it was trained on. Inference
                // path stays z-scored to match training; the UI gets the
                // raw dB frame separately and normalizes for display so
                // temporal contrast across cardiac cycles survives.
                mel_out.clear();
                mel_db_out.clear();
                mel.process_with_display(&mel_in, &mut mel_out, &mut mel_db_out);
                for (norm_frame, db_frame) in mel_out.drain(..).zip(mel_db_out.drain(..)) {
                    let display = display_normalize(db_frame, &mut db_history);
                    let _ = tx.send(Msg::MelFrame(display));
                    if let Some(eng) = engine.as_mut() {
                        if let Some(prob) = eng.push_frame(&norm_frame) {
                            let _ = tx.send(Msg::MurmurProb(prob));
                        }
                    }
                    if let Some((eng, _)) = s3_engine_pair.as_mut() {
                        if let Some(prob) = eng.push_frame(&norm_frame) {
                            let _ = tx.send(Msg::S3Prob(prob));
                        }
                    }
                }

                if last_flush.elapsed() > Duration::from_secs(2) {
                    if let Ok(Some(b)) = ble::read_battery(&peripheral).await {
                        let _ = tx.send(Msg::Battery(b));
                    }
                    last_flush = Instant::now();
                }
            }
        }
    }

    // Bound the disconnect; CoreBluetooth can wedge here indefinitely.
    let _ = tokio::time::timeout(Duration::from_secs(2), peripheral.disconnect()).await;
    Ok(())
}

fn main() -> Result<(), eframe::Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    // Model path: first CLI arg, preferred env var, or legacy env var.
    let explicit_model_path: Option<PathBuf> = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("OPENSTETHO_MODEL").map(PathBuf::from))
        .or_else(|| std::env::var_os("EKO_MODEL").map(PathBuf::from))
        .filter(|p| p.exists());

    // Explicit model dirs keep the old broad discovery behavior for
    // experiments. The default UI stays on the current release/local bundle
    // so legacy runs do not appear in the picker.
    let env_model_dirs = std::env::var("OPENSTETHO_MODEL_DIRS")
        .or_else(|_| std::env::var("EKO_MODEL_DIRS"))
        .ok();
    let available_models = match env_model_dirs {
        Some(model_dirs) => discover_models(&model_dirs, explicit_model_path.as_ref()),
        None => discover_current_models(explicit_model_path.as_ref()),
    };
    let model_path = explicit_model_path.or_else(|| available_models.first().cloned());

    let (msg_tx, msg_rx) = crossbeam_channel::unbounded::<Msg>();
    let (cmd_tx, cmd_rx) = crossbeam_channel::unbounded::<Cmd>();
    spawn_ble_worker(msg_tx.clone(), cmd_rx, model_path.clone());

    let opts = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 640.0])
            .with_title("stetho-ui — live stethoscope dev testbed"),
        ..Default::default()
    };
    eframe::run_native(
        "stetho-ui",
        opts,
        Box::new(move |_cc| {
            info!(
                "stetho-ui started ({} models discovered)",
                available_models.len()
            );
            Ok(Box::new(App::new(
                msg_rx,
                msg_tx,
                cmd_tx,
                model_path,
                available_models,
            )))
        }),
    )
}

fn discover_models(dirs: &str, current: Option<&PathBuf>) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for root in dirs.split(':') {
        let root = std::path::Path::new(root);
        if !root.exists() {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if is_murmur_model_artifact(&path) {
                    found.push(path);
                    continue;
                }
                for name in ["MurmurCNN.mlpackage", "MurmurCNN.mlmodel"] {
                    let candidate = path.join(name);
                    if candidate.exists() {
                        found.push(candidate);
                    }
                }
            }
        }
    }
    if let Some(c) = current {
        if is_murmur_model_artifact(c) && !found.iter().any(|p| p == c) {
            found.insert(0, c.clone());
        }
    }
    found.sort();
    found.dedup();
    found
}

fn discover_current_models(current: Option<&PathBuf>) -> Vec<PathBuf> {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    discover_current_models_in(&repo_root, default_model_download_dir(), current)
}

fn discover_current_models_in(
    repo_root: &std::path::Path,
    download_dir: PathBuf,
    current: Option<&PathBuf>,
) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let downloaded = download_dir.join("MurmurCNN.mlpackage");
    let local_current = repo_root
        .join("model/runs")
        .join(CURRENT_LOCAL_MURMUR_RUN)
        .join("MurmurCNN.mlpackage");

    // Prefer the downloaded latest-release bundle because it carries the
    // matching S3 sibling. Fall back to the latest local murmur export.
    for candidate in [downloaded, local_current] {
        if candidate.exists() && !found.iter().any(|p| p == &candidate) {
            found.push(candidate);
            break;
        }
    }

    if let Some(c) = current {
        if is_murmur_model_artifact(c) && !found.iter().any(|p| p == c) {
            found.insert(0, c.clone());
        }
    }
    found
}

fn is_murmur_model_artifact(path: &std::path::Path) -> bool {
    is_model_artifact(path)
        && path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|name| {
                name.eq_ignore_ascii_case("MurmurCNN.mlpackage")
                    || name.eq_ignore_ascii_case("MurmurCNN.mlmodel")
            })
            .unwrap_or(false)
}

fn is_model_artifact(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("mlpackage") || ext.eq_ignore_ascii_case("mlmodel"))
        .unwrap_or(false)
}

fn display_label(path: &std::path::Path) -> Option<String> {
    let filename = path.file_name().and_then(|s| s.to_str()).unwrap_or("model");
    let parent = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())?;
    if parent == "downloaded" && filename == "MurmurCNN.mlpackage" {
        return Some(format!("{CURRENT_MODEL_RELEASE_LABEL} (Murmur + S3)"));
    }
    if parent == CURRENT_LOCAL_MURMUR_RUN && filename == "MurmurCNN.mlpackage" {
        return Some("CirCor 2022 murmur 5s top4".to_string());
    }
    Some(format!("{parent}/{filename}"))
}

fn fallback_s3_model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("model/runs")
        .join(CURRENT_LOCAL_S3_RUN)
        .join("S3CNN_v2.mlpackage")
}

fn murmur_frame_count(path: &std::path::Path) -> usize {
    model_metadata(path)
        .and_then(|v| v.get("n_frames").and_then(|n| n.as_u64()))
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .unwrap_or(N_FRAMES_MURMUR)
}

fn murmur_decision_config(path: Option<&PathBuf>) -> MurmurDecisionConfig {
    let Some(path) = path else {
        return MurmurDecisionConfig::default();
    };
    let Some(metadata) = model_metadata(path) else {
        return MurmurDecisionConfig::default();
    };
    let aggregation = metadata
        .get("murmur_aggregation")
        .and_then(|v| v.as_str())
        .and_then(|value| {
            let topk = metadata
                .get("murmur_topk")
                .and_then(|v| v.as_u64())
                .and_then(|n| usize::try_from(n).ok());
            parse_murmur_aggregation(value, topk)
        })
        .unwrap_or(MurmurAggregation::Mean);
    let threshold = metadata
        .get("murmur_threshold")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .filter(|v| (0.0..=1.0).contains(v))
        .unwrap_or_else(|| MurmurDecisionConfig::default().threshold);

    MurmurDecisionConfig {
        aggregation,
        threshold,
    }
}

fn parse_murmur_aggregation(value: &str, topk: Option<usize>) -> Option<MurmurAggregation> {
    match value {
        "mean" => Some(MurmurAggregation::Mean),
        "topk_mean" => Some(MurmurAggregation::TopKMean(topk.unwrap_or(3).max(1))),
        "top3_mean" => Some(MurmurAggregation::TopKMean(3)),
        "top4_mean" => Some(MurmurAggregation::TopKMean(4)),
        "top5_mean" => Some(MurmurAggregation::TopKMean(5)),
        _ => None,
    }
}

fn model_metadata(model_path: &std::path::Path) -> Option<serde_json::Value> {
    model_metadata_path(model_path)
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
}

fn model_metadata_path(model_path: &std::path::Path) -> Option<PathBuf> {
    let stem = model_path.file_stem()?.to_str()?;
    Some(model_path.with_file_name(format!("{stem}.openstetho.json")))
}

fn default_model_download_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("OPENSTETHO_MODEL_DOWNLOAD_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("model/runs/downloaded")
}

fn spawn_model_download(tx: Sender<Msg>) {
    std::thread::Builder::new()
        .name("model-download".into())
        .spawn(move || {
            let url = std::env::var("OPENSTETHO_MODEL_DOWNLOAD_URL")
                .unwrap_or_else(|_| DEFAULT_MODEL_DOWNLOAD_URL.to_string());
            let out_dir = default_model_download_dir();
            let result = download_model(&url, &out_dir).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::ModelDownloadFinished(result));
        })
        .expect("spawn model download");
}

fn download_model(url: &str, out_dir: &std::path::Path) -> anyhow::Result<PathBuf> {
    fs::create_dir_all(out_dir)?;
    let tmp = out_dir.join("model-download.tmp");
    let response = ureq::get(url).call().map_err(|err| match err {
        ureq::Error::Status(404, _) if url == DEFAULT_MODEL_DOWNLOAD_URL => anyhow::anyhow!(
            "no default model release asset is published yet; set OPENSTETHO_MODEL_DOWNLOAD_URL to a hosted .mlpackage.zip or create a GitHub release asset named MurmurCNN.mlpackage.zip"
        ),
        other => anyhow::anyhow!("{other}"),
    })?;
    let mut reader = response.into_reader();
    let mut file = fs::File::create(&tmp)?;
    io::copy(&mut reader, &mut file)?;
    drop(file);

    let result = if url.ends_with(".zip") {
        extract_model_zip(&tmp, out_dir)
    } else {
        let filename = url
            .rsplit('/')
            .next()
            .filter(|s| !s.is_empty())
            .unwrap_or("MurmurCNN.mlmodel");
        let dest = out_dir.join(filename);
        fs::rename(&tmp, &dest)?;
        Ok(dest)
    };

    let _ = fs::remove_file(&tmp);
    result
}

fn extract_model_zip(
    zip_path: &std::path::Path,
    out_dir: &std::path::Path,
) -> anyhow::Result<PathBuf> {
    let extract_dir = out_dir.join("model-download.extract");
    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir)?;
    }
    fs::create_dir_all(&extract_dir)?;

    let file = fs::File::open(zip_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(path) = entry.enclosed_name().map(|p| p.to_owned()) else {
            continue;
        };
        let dest = extract_dir.join(path);
        if entry.is_dir() {
            fs::create_dir_all(&dest)?;
        } else {
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut outfile = fs::File::create(&dest)?;
            io::copy(&mut entry, &mut outfile)?;
        }
    }

    let models = find_downloaded_models(&extract_dir);
    if models.is_empty() {
        return Err(anyhow::anyhow!(
            "download did not contain a .mlpackage or .mlmodel"
        ));
    }
    let mut installed = Vec::with_capacity(models.len());
    for model in models {
        let dest = out_dir.join(
            model
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("downloaded model has no file name"))?,
        );
        if dest.exists() {
            if dest.is_dir() {
                fs::remove_dir_all(&dest)?;
            } else {
                fs::remove_file(&dest)?;
            }
        }
        fs::rename(&model, &dest)?;
        installed.push(dest);
    }
    let model = prefer_murmur_model(&installed)
        .ok_or_else(|| anyhow::anyhow!("download did not contain a .mlpackage or .mlmodel"))?;
    for metadata in find_model_metadata_files(&extract_dir) {
        if let Some(name) = metadata.file_name() {
            fs::rename(&metadata, out_dir.join(name))?;
        }
    }
    let _ = fs::remove_dir_all(&extract_dir);
    Ok(model)
}

fn prefer_murmur_model(paths: &[PathBuf]) -> Option<PathBuf> {
    paths
        .iter()
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|name| name.eq_ignore_ascii_case("MurmurCNN.mlpackage"))
                .unwrap_or(false)
        })
        .or_else(|| paths.first())
        .cloned()
}

fn find_downloaded_models(root: &std::path::Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    collect_downloaded_models(root, &mut found);
    found.sort();
    found
}

fn find_model_metadata_files(root: &std::path::Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    collect_model_metadata_files(root, &mut found);
    found.sort();
    found
}

fn collect_model_metadata_files(root: &std::path::Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|s| s.to_str())
            .map(|name| name.ends_with(".openstetho.json"))
            .unwrap_or(false)
        {
            found.push(path);
            continue;
        }
        if path.is_dir() {
            collect_model_metadata_files(&path, found);
        }
    }
}

fn collect_downloaded_models(root: &std::path::Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_model_artifact(&path) {
            found.push(path);
            continue;
        }
        if path.is_dir() {
            collect_downloaded_models(&path, found);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("openstetho-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn synthetic_model_zip() -> Vec<u8> {
        let cursor = io::Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(cursor);
        let opts = zip::write::SimpleFileOptions::default();
        zip.add_directory("S3CNN_v2.mlpackage/", opts).unwrap();
        zip.start_file("S3CNN_v2.mlpackage/Manifest.json", opts)
            .unwrap();
        zip.write_all(br#"{"model":"synthetic-s3"}"#).unwrap();
        zip.add_directory("MurmurCNN.mlpackage/", opts).unwrap();
        zip.start_file("MurmurCNN.mlpackage/Manifest.json", opts)
            .unwrap();
        zip.write_all(br#"{"model":"synthetic-murmur"}"#).unwrap();
        zip.start_file("MurmurCNN.openstetho.json", opts).unwrap();
        zip.write_all(
            br#"{"n_frames":78,"murmur_aggregation":"topk_mean","murmur_topk":4,"murmur_threshold":0.34336904}"#,
        )
        .unwrap();
        zip.finish().unwrap().into_inner()
    }

    fn serve_once(bytes: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/zip\r\nConnection: close\r\n\r\n",
                bytes.len()
            )
            .unwrap();
            stream.write_all(&bytes).unwrap();
        });
        format!("http://{addr}/MurmurCNN.mlpackage.zip")
    }

    #[test]
    fn downloads_and_discovers_zipped_mlpackage() {
        let out_dir = temp_dir("model-download");
        let url = serve_once(synthetic_model_zip());

        let model = download_model(&url, &out_dir).unwrap();

        assert_eq!(
            model.file_name().and_then(|s| s.to_str()),
            Some("MurmurCNN.mlpackage")
        );
        assert!(model.join("Manifest.json").exists());
        assert!(out_dir.join("S3CNN_v2.mlpackage/Manifest.json").exists());
        assert!(out_dir.join("MurmurCNN.openstetho.json").exists());
        assert_eq!(murmur_frame_count(&model), 78);
        let config = murmur_decision_config(Some(&model));
        assert_eq!(config.aggregation, MurmurAggregation::TopKMean(4));
        assert!((config.threshold - 0.34336904).abs() < f32::EPSILON);

        let discovered = discover_models(out_dir.to_str().unwrap(), None);
        assert_eq!(discovered, vec![model]);

        let _ = fs::remove_dir_all(out_dir);
    }

    #[test]
    fn downloaded_bundle_label_is_explicit() {
        let path = std::path::Path::new("model/runs/downloaded/MurmurCNN.mlpackage");

        assert_eq!(
            display_label(path).as_deref(),
            Some("v0.4.0-murmur-ensemble (Murmur + S3)")
        );
    }

    #[test]
    fn missing_metadata_uses_legacy_murmur_frame_count() {
        let path = std::path::Path::new("model/runs/release-circor-v2/MurmurCNN.mlpackage");

        assert_eq!(murmur_frame_count(path), N_FRAMES_MURMUR);
    }

    #[test]
    fn missing_metadata_uses_legacy_murmur_decision_config() {
        let path = PathBuf::from("model/runs/release-circor-v2/MurmurCNN.mlpackage");
        let config = murmur_decision_config(Some(&path));

        assert_eq!(config.aggregation, MurmurAggregation::Mean);
        assert_eq!(config.threshold, MURMUR_RECORDING_MEAN_THRESHOLD);
    }

    #[test]
    fn ensemble_5s_sidecar_drives_frame_count_and_threshold() {
        // Mirrors the sidecar written by export_ensemble for the shipped 5 s
        // cnn_bigru ensemble: 78-frame window, session-mean aggregation, and
        // the calibrated Youden operating threshold.
        let dir = temp_dir("ensemble-5s-sidecar");
        let model = dir.join("MurmurCNN.mlpackage");
        fs::create_dir_all(&model).unwrap();
        fs::write(
            dir.join("MurmurCNN.openstetho.json"),
            br#"{"architecture":"cnn_bigru_ensemble","ensemble_size":3,"window_aggregation":"prob_mean","murmur_aggregation":"mean","window_seconds":5.0,"n_frames":78,"murmur_threshold":0.535999}"#,
        )
        .unwrap();

        assert_eq!(murmur_frame_count(&model), 78);
        let config = murmur_decision_config(Some(&model));
        assert_eq!(config.aggregation, MurmurAggregation::Mean);
        assert!((config.threshold - 0.535999).abs() < 1e-6);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn default_discovery_uses_latest_downloaded_bundle_only() {
        let repo = temp_dir("model-discovery-repo");
        let download_dir = temp_dir("model-discovery-download");
        let downloaded = download_dir.join("MurmurCNN.mlpackage");
        let current = repo
            .join("model/runs")
            .join(CURRENT_LOCAL_MURMUR_RUN)
            .join("MurmurCNN.mlpackage");
        let old = repo
            .join("model/runs/release-circor-v1")
            .join("MurmurCNN.mlpackage");

        fs::create_dir_all(&downloaded).unwrap();
        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&old).unwrap();

        let discovered = discover_current_models_in(&repo, download_dir.clone(), None);

        assert_eq!(discovered, vec![downloaded]);
        assert!(!discovered.iter().any(|p| p == &current));
        assert!(!discovered.iter().any(|p| p == &old));

        let _ = fs::remove_dir_all(repo);
        let _ = fs::remove_dir_all(download_dir);
    }

    #[test]
    fn default_discovery_falls_back_to_current_local_run() {
        let repo = temp_dir("model-discovery-local");
        let download_dir = temp_dir("model-discovery-empty-download");
        let current = repo
            .join("model/runs")
            .join(CURRENT_LOCAL_MURMUR_RUN)
            .join("MurmurCNN.mlpackage");
        let old = repo
            .join("model/runs/release-circor-v1")
            .join("MurmurCNN.mlpackage");

        fs::create_dir_all(&current).unwrap();
        fs::create_dir_all(&old).unwrap();

        let discovered = discover_current_models_in(&repo, download_dir.clone(), None);

        assert_eq!(discovered, vec![current]);
        assert!(!discovered.iter().any(|p| p == &old));

        let _ = fs::remove_dir_all(repo);
        let _ = fs::remove_dir_all(download_dir);
    }

    #[test]
    fn default_404_explains_missing_release_asset() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .unwrap();
        });

        let out_dir = temp_dir("model-download-404");
        let err = download_model(&format!("http://{addr}/missing.zip"), &out_dir)
            .unwrap_err()
            .to_string();
        assert!(err.contains("status code 404"));

        let _ = fs::remove_dir_all(out_dir);
    }
}
