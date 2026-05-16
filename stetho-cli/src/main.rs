use anyhow::Result;
use btleplug::api::Peripheral as _;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use std::time::Duration;
use stetho_core::ble;
use stetho_core::{AUDIO_SAMPLE_RATE_HZ, BLOCK_SIZE};
use tokio::time::sleep;
use tracing::{info, warn};

#[derive(Parser)]
#[command(name = "stetho", about = "Digital stethoscope interoperability client")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print build configuration and known protocol constants.
    Info,

    /// Scan for compatible devices.
    Scan {
        /// Scan duration in seconds.
        #[arg(short, long, default_value_t = 10)]
        seconds: u64,
    },

    /// Connect to a device and dump its GATT tree + battery level.
    /// Target may be a Bluetooth address, a macOS PeripheralId substring,
    /// or a name substring (e.g. "core").
    Connect {
        target: String,
        /// Seconds to scan first, so the adapter learns about the device.
        #[arg(short, long, default_value_t = 8)]
        rescan: u64,
    },

    /// Connect, subscribe to a NOTIFY characteristic, hex-dump every
    /// notification until Ctrl-C or `--count` is reached.
    Stream {
        target: String,
        #[arg(long, default_value = "c320d257-d7be-46ac-9a37-7a4edfa84bce")]
        char: String,
        #[arg(short, long, default_value_t = 0)]
        count: usize,
        #[arg(short, long, default_value_t = 8)]
        rescan: u64,
        #[arg(short, long)]
        out: Option<std::path::PathBuf>,
    },

    /// Capture for N seconds. Writes <out>.hex (raw payloads) and <out>.wav
    /// (decoded PCM at 4 kHz mono i16).
    Capture {
        target: String,
        /// Output base path (no extension). Two files are written:
        /// `<out>.hex` and `<out>.wav`.
        #[arg(short, long)]
        out: std::path::PathBuf,
        /// Duration in seconds.
        #[arg(short, long, default_value_t = 30)]
        seconds: u64,
        /// Characteristic UUID. Defaults to Core 2 ADPCM audio.
        #[arg(long, default_value = "c320d257-d7be-46ac-9a37-7a4edfa84bce")]
        char: String,
        #[arg(short, long, default_value_t = 8)]
        rescan: u64,
    },

    /// Offline: decode a previously-captured `<base>.hex` to a WAV.
    DecodeHex {
        hex: std::path::PathBuf,
        #[arg(short, long)]
        out: std::path::PathBuf,
        /// EQ preset to apply after ADPCM decode.
        /// One of: none | wide | cardiac | pulmonary.
        #[arg(long, default_value = "none")]
        preset: String,
        /// Linear gain applied before writing the WAV. Heart sounds are
        /// faint; values 8..32 are reasonable. Ignored when --normalize.
        #[arg(short, long, default_value_t = 1.0)]
        gain: f32,
        /// Auto-scale output so peak hits ~-1 dBFS. Overrides --gain.
        #[arg(short, long)]
        normalize: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Info => cmd_info(),
        Cmd::Scan { seconds } => cmd_scan(seconds).await,
        Cmd::Connect { target, rescan } => cmd_connect(&target, rescan).await,
        Cmd::Stream {
            target,
            char,
            count,
            rescan,
            out,
        } => cmd_stream(&target, &char, count, rescan, out).await,
        Cmd::Capture {
            target,
            out,
            seconds,
            char,
            rescan,
        } => cmd_capture(&target, &out, seconds, &char, rescan).await,
        Cmd::DecodeHex {
            hex,
            out,
            preset,
            gain,
            normalize,
        } => cmd_decode_hex(&hex, &out, &preset, gain, normalize),
    }
}

fn parse_hex_line(line: &str) -> anyhow::Result<Vec<u8>> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(Vec::new());
    }
    anyhow::ensure!(line.len() % 2 == 0, "odd hex length");
    (0..line.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&line[i..i + 2], 16).map_err(Into::into))
        .collect()
}

fn write_wav(path: &std::path::Path, samples: &[i16]) -> anyhow::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: AUDIO_SAMPLE_RATE_HZ,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in samples {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    Ok(())
}

async fn cmd_capture(
    target: &str,
    out_base: &std::path::Path,
    seconds: u64,
    char: &str,
    rescan: u64,
) -> Result<()> {
    use std::io::Write as _;
    use stetho_core::AdpcmDecoder;

    let hex_path = out_base.with_extension("hex");
    let wav_path = out_base.with_extension("wav");
    if let Some(parent) = hex_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mgr = ble::manager().await?;
    let adapter = ble::default_adapter(&mgr).await?;
    let _ = ble::scan(&adapter, Duration::from_secs(rescan)).await?;
    let peripheral = ble::resolve_peripheral(&adapter, target).await?;
    ble::connect_and_discover(&peripheral).await?;
    let char_uuid = uuid::Uuid::parse_str(char).map_err(|e| anyhow::anyhow!("bad UUID: {e}"))?;
    info!("connected, subscribing to {char_uuid}");
    ble::subscribe(&peripheral, char_uuid).await?;
    let mut notifications = peripheral.notifications().await?;

    let mut hex_writer = std::io::BufWriter::new(std::fs::File::create(&hex_path)?);
    let mut decoder = AdpcmDecoder::new();
    let mut pcm: Vec<i16> = Vec::with_capacity(AUDIO_SAMPLE_RATE_HZ as usize * seconds as usize);

    info!(
        "recording {seconds}s → {} + {}",
        hex_path.display(),
        wav_path.display()
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(seconds);
    let mut n: usize = 0;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, notifications.next()).await {
            Ok(Some(notif)) => {
                n += 1;
                writeln!(hex_writer, "{}", hex(&notif.value))?;
                decoder.decode_packet(&notif.value, &mut pcm);
            }
            Ok(None) | Err(_) => break,
        }
    }
    hex_writer.flush().ok();
    write_wav(&wav_path, &pcm)?;

    let secs_of_audio = pcm.len() as f64 / AUDIO_SAMPLE_RATE_HZ as f64;
    info!(
        "captured {n} packets, {} samples ({:.2}s of audio)",
        pcm.len(),
        secs_of_audio
    );
    let observed_rate = n as f64 / seconds as f64;
    info!(
        "observed packet rate: {:.2} pkt/s  ⇒ {:.0} samples/s",
        observed_rate,
        observed_rate * 474.0
    );

    sleep(Duration::from_millis(100)).await;
    peripheral.disconnect().await.ok();
    Ok(())
}

fn cmd_decode_hex(
    hex_path: &std::path::Path,
    out_path: &std::path::Path,
    preset: &str,
    gain: f32,
    normalize: bool,
) -> Result<()> {
    use stetho_core::dsp::presets::{EqPreset, FilterChain};
    use stetho_core::AdpcmDecoder;

    let text = std::fs::read_to_string(hex_path)?;
    let mut decoder = AdpcmDecoder::new();
    let mut pcm: Vec<i16> = Vec::new();
    let mut packets: usize = 0;
    for line in text.lines() {
        let bytes = parse_hex_line(line)?;
        if bytes.is_empty() {
            continue;
        }
        decoder.decode_packet(&bytes, &mut pcm);
        packets += 1;
    }

    let mut floats: Vec<f32> = pcm.iter().map(|s| *s as f32).collect();
    let raw_peak = floats.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let raw_rms = (floats.iter().map(|v| v * v).sum::<f32>() / floats.len().max(1) as f32).sqrt();

    let parsed = match preset.to_lowercase().as_str() {
        "none" => EqPreset::None,
        "wide" => EqPreset::Wide,
        "cardiac" => EqPreset::Cardiac,
        "pulmonary" => EqPreset::Pulmonary,
        other => anyhow::bail!("unknown preset {other}; use none|wide|cardiac|pulmonary"),
    };
    if let Some(mut chain) = FilterChain::new(parsed) {
        chain.process_block(&mut floats);
    }
    let applied_gain = if normalize {
        let peak_after_eq = floats.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
        if peak_after_eq > 0.0 {
            (i16::MAX as f32 * 0.89) / peak_after_eq
        } else {
            1.0
        }
    } else {
        gain
    };
    for v in floats.iter_mut() {
        *v *= applied_gain;
    }
    let post_peak = floats.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let post_rms = (floats.iter().map(|v| v * v).sum::<f32>() / floats.len().max(1) as f32).sqrt();
    let mut clipped = 0u32;
    let out_pcm: Vec<i16> = floats
        .iter()
        .map(|v| {
            if *v > i16::MAX as f32 || *v < i16::MIN as f32 {
                clipped += 1;
            }
            v.clamp(i16::MIN as f32, i16::MAX as f32) as i16
        })
        .collect();
    write_wav(out_path, &out_pcm)?;

    info!(
        "decoded {packets} pkts → {} samples → {}",
        out_pcm.len(),
        out_path.display()
    );
    info!(
        "raw  peak={:.0} rms={:.1}  ({:.1} dBFS)",
        raw_peak,
        raw_rms,
        20.0 * (raw_peak.max(1.0) / i16::MAX as f32).log10()
    );
    info!(
        "post peak={:.0} rms={:.1}  preset={preset} gain={applied_gain:.2}{}  clipped={clipped}",
        post_peak,
        post_rms,
        if normalize { " (normalized)" } else { "" }
    );
    Ok(())
}

fn cmd_info() -> Result<()> {
    info!("stetho-cli {}", env!("CARGO_PKG_VERSION"));
    info!("audio sample rate: {} Hz", AUDIO_SAMPLE_RATE_HZ);
    info!("pipeline block size: {} samples", BLOCK_SIZE);
    Ok(())
}

async fn cmd_scan(seconds: u64) -> Result<()> {
    let mgr = ble::manager().await?;
    let adapter = ble::default_adapter(&mgr).await?;
    info!("scanning for {seconds}s");
    let hits = ble::scan(&adapter, Duration::from_secs(seconds)).await?;
    if hits.is_empty() {
        warn!("no compatible devices found");
    } else {
        println!("name\taddress\tid\trssi");
        for d in &hits {
            println!("{}\t{}\t{:?}\t{:?}", d.name, d.address, d.id, d.rssi);
        }
    }
    Ok(())
}

async fn cmd_connect(target: &str, rescan: u64) -> Result<()> {
    let mgr = ble::manager().await?;
    let adapter = ble::default_adapter(&mgr).await?;
    let _ = ble::scan(&adapter, Duration::from_secs(rescan)).await?;
    let peripheral = ble::resolve_peripheral(&adapter, target).await?;
    info!("connecting to {target}");
    ble::connect_and_discover(&peripheral).await?;
    info!("connected");
    ble::dump_gatt(&peripheral).await?;
    match ble::read_battery(&peripheral).await? {
        Some(pct) => info!("battery: {pct}%"),
        None => warn!("battery characteristic not present"),
    }
    peripheral.disconnect().await.ok();
    Ok(())
}

async fn cmd_stream(
    target: &str,
    char: &str,
    count: usize,
    rescan: u64,
    out: Option<std::path::PathBuf>,
) -> Result<()> {
    use std::io::Write as _;
    let mgr = ble::manager().await?;
    let adapter = ble::default_adapter(&mgr).await?;
    let _ = ble::scan(&adapter, Duration::from_secs(rescan)).await?;
    let peripheral = ble::resolve_peripheral(&adapter, target).await?;
    ble::connect_and_discover(&peripheral).await?;
    let char_uuid = uuid::Uuid::parse_str(char).map_err(|e| anyhow::anyhow!("bad UUID: {e}"))?;
    info!("connected, subscribing to {char_uuid}");
    ble::subscribe(&peripheral, char_uuid).await?;
    let mut notifications = peripheral.notifications().await?;
    let mut writer = match out {
        Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
        None => None,
    };
    let mut n: usize = 0;
    while let Some(notif) = notifications.next().await {
        n += 1;
        let line = format!(
            "{:5}  uuid={}  len={}  bytes={}",
            n,
            notif.uuid,
            notif.value.len(),
            hex(&notif.value)
        );
        println!("{line}");
        if let Some(w) = writer.as_mut() {
            writeln!(w, "{}", hex(&notif.value)).ok();
        }
        if count != 0 && n >= count {
            break;
        }
    }
    if let Some(mut w) = writer {
        w.flush().ok();
    }
    sleep(Duration::from_millis(100)).await;
    peripheral.disconnect().await.ok();
    Ok(())
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        use std::fmt::Write;
        write!(&mut s, "{:02x}", byte).ok();
    }
    s
}
