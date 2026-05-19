//! Log-mel spectrogram for 4 kHz stethoscope audio.
//!
//! 256-point FFT, no overlap, magnitude (not power), 32 mel bands,
//! log₁₀ × 10, per-frame z-score clipped at -80 dB. Mel filterbank
//! coefficients are generated from the standard Slaney formula
//! (Auditory Toolbox, Apple Technical Report #45, 1998).
//!
//! Preprocessor for the Core ML / ANE murmur classifier trained on
//! public PCG data (PhysioNet CirCor 2022).

use rustfft::{num_complex::Complex32, FftPlanner};
use std::sync::Arc;

/// Pipeline constants.
pub const N_FFT: usize = 256;
pub const HOP: usize = 256; // no overlap
pub const N_MELS: usize = 32;
pub const N_FFT_BINS: usize = N_FFT / 2 + 1; // 129 unique bins for a real 256-FFT
/// FFT bins fed to the mel filterbank. We keep only the lower half of
/// the spectrum (≤ ~1 kHz) so the mel bands concentrate on heart-band
/// content where S1/S2 + murmurs live.
pub const MEL_FFT_BINS: usize = 65;
pub const LOG_FLOOR: f32 = 1e-10;
pub const NORM_CLIP_DB: f32 = -80.0;

/// Slaney mel scale: linear below 1 kHz, logarithmic above.
fn hz_to_mel_slaney(f: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0; // 66.66… Hz / mel below break
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f32.ln()) / 27.0_f32;
    if f >= MIN_LOG_HZ {
        min_log_mel + (f / MIN_LOG_HZ).ln() / logstep
    } else {
        (f - F_MIN) / F_SP
    }
}

fn mel_to_hz_slaney(m: f32) -> f32 {
    const F_MIN: f32 = 0.0;
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = (MIN_LOG_HZ - F_MIN) / F_SP;
    let logstep = (6.4_f32.ln()) / 27.0_f32;
    if m >= min_log_mel {
        MIN_LOG_HZ * (logstep * (m - min_log_mel)).exp()
    } else {
        F_MIN + m * F_SP
    }
}

/// Construct an `N_MELS × MEL_FFT_BINS` triangular mel filterbank, Slaney
/// normalized so each filter's coefficients sum to a constant (per-band
/// energy preserving, librosa default).
fn build_mel_filterbank(sample_rate: f32) -> Vec<Vec<f32>> {
    let f_min = 20.0_f32;
    // Cap f_max at the highest frequency representable by MEL_FFT_BINS.
    // For 4 kHz audio and 256-FFT, bin 64 sits at 1000 Hz, which keeps the
    // mel filters inside the window of FFT bins we forward to them.
    let nyquist_of_mel_window = (MEL_FFT_BINS as f32 - 1.0) * sample_rate / N_FFT as f32;
    let f_max = (sample_rate / 2.0).min(nyquist_of_mel_window);
    let m_min = hz_to_mel_slaney(f_min);
    let m_max = hz_to_mel_slaney(f_max);

    let mut mel_pts = Vec::with_capacity(N_MELS + 2);
    for i in 0..(N_MELS + 2) {
        let m = m_min + (m_max - m_min) * (i as f32) / (N_MELS as f32 + 1.0);
        mel_pts.push(mel_to_hz_slaney(m));
    }

    let bin_to_hz = |bin: usize| (bin as f32) * sample_rate / (N_FFT as f32);
    let mut bank = vec![vec![0.0_f32; MEL_FFT_BINS]; N_MELS];
    for m in 0..N_MELS {
        let lower = mel_pts[m];
        let center = mel_pts[m + 1];
        let upper = mel_pts[m + 2];
        let norm = 2.0 / (upper - lower).max(1e-9);
        for bin in 0..MEL_FFT_BINS {
            let f = bin_to_hz(bin);
            let w = if f < lower || f > upper {
                0.0
            } else if f <= center {
                (f - lower) / (center - lower).max(1e-9)
            } else {
                (upper - f) / (upper - center).max(1e-9)
            };
            bank[m][bin] = w * norm;
        }
    }
    bank
}

fn z_score_clip(mut mel: [f32; N_MELS]) -> [f32; N_MELS] {
    let mean: f32 = mel.iter().sum::<f32>() / N_MELS as f32;
    let var: f32 = mel.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / N_MELS as f32;
    let std = var.sqrt().max(1e-9);
    for v in mel.iter_mut() {
        let z = (*v - mean) / std;
        *v = if z < NORM_CLIP_DB / 80.0 {
            NORM_CLIP_DB / 80.0
        } else {
            z
        };
    }
    mel
}

fn hann_window(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - ((2.0 * std::f32::consts::PI * i as f32) / denom).cos()))
        .collect()
}

/// Streaming log-mel spectrogram. `process()` consumes any number of
/// time-domain samples and emits one `N_MELS`-dim frame per `HOP`
/// samples buffered.
pub struct LogMelSpectrogram {
    fft: Arc<dyn rustfft::Fft<f32>>,
    window: Vec<f32>,
    filterbank: Vec<Vec<f32>>,
    pending: Vec<f32>,
    scratch: Vec<Complex32>,
    sample_rate: f32,
}

impl LogMelSpectrogram {
    pub fn new(sample_rate: f32) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        Self {
            fft,
            window: hann_window(N_FFT),
            filterbank: build_mel_filterbank(sample_rate),
            pending: Vec::with_capacity(N_FFT * 4),
            scratch: vec![Complex32::default(); N_FFT],
            sample_rate,
        }
    }

    pub fn sample_rate(&self) -> f32 {
        self.sample_rate
    }

    /// Drain any pending frames worth of audio, appending one
    /// `[N_MELS]` mel vector per complete frame to `out`.
    pub fn process(&mut self, samples: &[f32], out: &mut Vec<[f32; N_MELS]>) {
        self.pending.extend_from_slice(samples);
        while self.pending.len() >= N_FFT {
            let db = self.compute_db_frame();
            out.push(z_score_clip(db));
            // Hop = N_FFT, so the entire frame is consumed.
            self.pending.drain(..HOP);
        }
    }

    /// Drain pending audio, emitting one **z-scored** inference frame and
    /// one matching **raw dB** display frame per complete window. The two
    /// vectors grow in lockstep. Display layers should colormap the dB
    /// frame using a global / rolling normalization (per-frame z-score
    /// kills temporal contrast); the model must consume the z-scored
    /// frame to match training-time preprocessing.
    pub fn process_with_display(
        &mut self,
        samples: &[f32],
        out_norm: &mut Vec<[f32; N_MELS]>,
        out_db: &mut Vec<[f32; N_MELS]>,
    ) {
        self.pending.extend_from_slice(samples);
        while self.pending.len() >= N_FFT {
            let db = self.compute_db_frame();
            out_db.push(db);
            out_norm.push(z_score_clip(db));
            self.pending.drain(..HOP);
        }
    }

    fn compute_db_frame(&mut self) -> [f32; N_MELS] {
        // Copy + mean-center + window into scratch.
        let mean: f32 = self.pending[..N_FFT].iter().sum::<f32>() / N_FFT as f32;
        for i in 0..N_FFT {
            let s = (self.pending[i] - mean) * self.window[i];
            self.scratch[i] = Complex32::new(s, 0.0);
        }
        self.fft.process(&mut self.scratch);

        // Magnitude, first MEL_FFT_BINS bins only.
        let mut mag = [0.0_f32; MEL_FFT_BINS];
        for bin in 0..MEL_FFT_BINS {
            let c = self.scratch[bin];
            mag[bin] = (c.re * c.re + c.im * c.im).sqrt();
        }

        // Mel matmul.
        let mut mel = [0.0_f32; N_MELS];
        for m in 0..N_MELS {
            let row = &self.filterbank[m];
            let mut acc = 0.0_f32;
            for bin in 0..MEL_FFT_BINS {
                acc += row[bin] * mag[bin];
            }
            mel[m] = acc;
        }

        // log10 × 10, floor at LOG_FLOOR.
        for m in 0..N_MELS {
            mel[m] = 10.0 * (mel[m].max(LOG_FLOOR)).log10();
        }
        mel
    }

    pub fn reset(&mut self) {
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(freq: f32, fs: f32, n: usize) -> Vec<f32> {
        let dt = 1.0 / fs;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * (i as f32) * dt).sin())
            .collect()
    }

    #[test]
    fn pipeline_shapes() {
        let mut m = LogMelSpectrogram::new(4000.0);
        let samples = sine(100.0, 4000.0, 4000);
        let mut frames: Vec<[f32; N_MELS]> = Vec::new();
        m.process(&samples, &mut frames);
        // 4000 samples / hop 256 = ~15 frames
        assert!(frames.len() >= 14 && frames.len() <= 16, "{}", frames.len());
        for f in &frames {
            assert_eq!(f.len(), N_MELS);
        }
    }

    #[test]
    fn low_freq_lands_in_low_bin() {
        let mut m = LogMelSpectrogram::new(4000.0);
        let samples = sine(60.0, 4000.0, 4000);
        let mut frames: Vec<[f32; N_MELS]> = Vec::new();
        m.process(&samples, &mut frames);
        // Pick a middle frame (skip first to avoid window-leakage transient).
        let frame = &frames[frames.len() / 2];
        let argmax = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // 60 Hz should land in the lowest few mel bins (heart-sound range).
        assert!(
            argmax < N_MELS / 4,
            "60 Hz tone peaked in bin {argmax}; expected < {}",
            N_MELS / 4
        );
    }

    #[test]
    fn upper_band_freq_lands_in_upper_mel_bin() {
        // MEL_FFT_BINS = 65 covers ~0..1015 Hz at 4 kHz sr; test stays
        // inside that range to avoid the deliberately discarded high
        // half of the spectrum.
        let mut m = LogMelSpectrogram::new(4000.0);
        let samples = sine(800.0, 4000.0, 4000);
        let mut frames: Vec<[f32; N_MELS]> = Vec::new();
        m.process(&samples, &mut frames);
        let frame = &frames[frames.len() / 2];
        let argmax = frame
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            argmax > N_MELS / 2,
            "800 Hz tone peaked in bin {argmax}; expected > {}",
            N_MELS / 2
        );
    }

    #[test]
    fn mel_filterbank_shape() {
        let bank = build_mel_filterbank(4000.0);
        assert_eq!(bank.len(), N_MELS);
        for row in &bank {
            assert_eq!(row.len(), MEL_FFT_BINS);
        }
        // First filter should have most energy in the lowest bins.
        let first_low: f32 = bank[0][..10].iter().sum();
        let first_high: f32 = bank[0][30..].iter().sum();
        assert!(first_low > first_high);
        // Last filter should have most energy in the highest bins.
        let last_low: f32 = bank[N_MELS - 1][..30].iter().sum();
        let last_high: f32 = bank[N_MELS - 1][30..].iter().sum();
        assert!(last_high > last_low);
    }
}
