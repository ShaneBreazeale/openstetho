//! Sliding-window signal level, used to gate murmur inference on silence.
//!
//! The murmur model has never heard digital silence: an all-zero window
//! normalises to an all-zero mel-spec, and its score there is meaningless.
//! A stalled BLE stream or an idle codec produces exactly that, so the UI
//! shows "no signal" instead of a score when the window's RMS falls below
//! [`SILENCE_DBFS`].

use std::collections::VecDeque;

/// RMS level (dB relative to i16 full scale) below which a window counts
/// as no signal. Heart recordings from BLE stethoscopes can be quiet: the
/// public HLS-CMDS set (Littmann CORE) 4-s windows bottom out near -68 dBFS,
/// against about -47 dBFS for CirCor. -85 dBFS (about 1.8 LSB RMS) keeps
/// ~17 dB of margin below that while still catching digital silence and
/// an idle ADPCM codec toggling by ±1 LSB (-90 dBFS).
pub const SILENCE_DBFS: f32 = -85.0;

/// RMS of the last `len` samples. The sum of squares is kept as an exact
/// integer, so it never drifts however long the session runs.
pub struct WindowRms {
    buf: VecDeque<i16>,
    len: usize,
    sum_sq: u64,
}

impl WindowRms {
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "window length must be positive");
        Self {
            buf: VecDeque::with_capacity(len),
            len,
            sum_sq: 0,
        }
    }

    pub fn push(&mut self, samples: &[i16]) {
        for &s in samples {
            if self.buf.len() == self.len {
                if let Some(old) = self.buf.pop_front() {
                    self.sum_sq -= square(old);
                }
            }
            self.buf.push_back(s);
            self.sum_sq += square(s);
        }
    }

    /// Samples currently in the window (less than `len` until it fills).
    pub fn filled(&self) -> usize {
        self.buf.len()
    }

    pub fn rms(&self) -> f32 {
        if self.buf.is_empty() {
            return 0.0;
        }
        (self.sum_sq as f64 / self.buf.len() as f64).sqrt() as f32
    }

    /// RMS in dB relative to i16 full scale; `-inf` for an exactly zero window.
    pub fn dbfs(&self) -> f32 {
        let rms = self.rms();
        if rms == 0.0 {
            return f32::NEG_INFINITY;
        }
        20.0 * (rms / 32_768.0).log10()
    }

    pub fn is_silent(&self, threshold_dbfs: f32) -> bool {
        self.dbfs() < threshold_dbfs
    }
}

fn square(s: i16) -> u64 {
    let v = s as i64;
    (v * v) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIN: usize = 15_872; // 62 mel frames x 256-sample hop

    fn sine(amp: f32, n: usize) -> Vec<i16> {
        (0..n)
            .map(|i| (amp * (2.0 * std::f32::consts::PI * 100.0 * i as f32 / 4000.0).sin()).round() as i16)
            .collect()
    }

    #[test]
    fn empty_meter_reads_silent() {
        let m = WindowRms::new(WIN);
        assert_eq!(m.filled(), 0);
        assert_eq!(m.rms(), 0.0);
        assert_eq!(m.dbfs(), f32::NEG_INFINITY);
        assert!(m.is_silent(SILENCE_DBFS));
    }

    #[test]
    fn digital_silence_is_silent() {
        let mut m = WindowRms::new(WIN);
        m.push(&vec![0; WIN]);
        assert!(m.is_silent(SILENCE_DBFS));
    }

    #[test]
    fn idle_codec_chatter_is_silent() {
        // ADPCM with near-zero input toggles by about one LSB.
        let mut m = WindowRms::new(WIN);
        let chatter: Vec<i16> = (0..WIN).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        m.push(&chatter);
        assert!((m.dbfs() - (-90.31)).abs() < 0.01);
        assert!(m.is_silent(SILENCE_DBFS));
    }

    #[test]
    fn quietest_hls_cmds_heart_level_is_not_silent() {
        // HLS-CMDS (Littmann CORE) 4-s windows bottom out
        // near -68 dBFS; the gate must leave them alone.
        let amp = 32_768.0 * 10f32.powf(-68.0 / 20.0) * std::f32::consts::SQRT_2;
        let mut m = WindowRms::new(WIN);
        m.push(&sine(amp, WIN));
        assert!((m.dbfs() - (-68.0)).abs() < 0.5, "{}", m.dbfs());
        assert!(!m.is_silent(SILENCE_DBFS));
    }

    #[test]
    fn full_scale_square_is_zero_dbfs() {
        let mut m = WindowRms::new(WIN);
        let sq: Vec<i16> = (0..WIN).map(|i| if i % 40 < 20 { i16::MAX } else { i16::MIN }).collect();
        m.push(&sq);
        assert!(m.dbfs().abs() < 0.01, "{}", m.dbfs());
    }

    #[test]
    fn window_slides_so_old_loud_audio_ages_out() {
        let mut m = WindowRms::new(WIN);
        m.push(&sine(10_000.0, WIN));
        m.push(&vec![0; WIN - 1]);
        assert!(!m.is_silent(SILENCE_DBFS), "one loud sample still in the window");
        m.push(&[0]);
        assert!(m.is_silent(SILENCE_DBFS));
        assert_eq!(m.filled(), WIN);
    }

    #[test]
    fn running_rms_matches_direct_computation_after_many_pushes() {
        let mut m = WindowRms::new(1000);
        let mut all: Vec<i16> = Vec::new();
        let mut s: u32 = 12_345;
        for chunk in 0..50 {
            let block: Vec<i16> = (0..(37 + chunk * 11))
                .map(|_| {
                    s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    (s >> 16) as i16
                })
                .collect();
            m.push(&block);
            all.extend_from_slice(&block);
        }
        let tail = &all[all.len() - 1000..];
        let direct = (tail.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 1000.0).sqrt();
        assert!(((m.rms() as f64) - direct).abs() / direct < 1e-6);
    }

    #[test]
    fn partial_window_uses_samples_seen_so_far() {
        let mut m = WindowRms::new(WIN);
        m.push(&[3, -4]);
        assert_eq!(m.filled(), 2);
        assert!((m.rms() - (12.5f32).sqrt()).abs() < 1e-6);
    }
}
