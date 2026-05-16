//! Emit the log-mel spectrogram of a deterministic 60 Hz sine to stdout
//! in a Python-friendly text format. Used by `model/tests/test_parity.py`
//! to verify the Python preprocessor matches the Rust runtime numerically.
//!
//! Layout: `n_frames n_mels\n<f1_b0> <f1_b1> ... \n<f2_b0> ...`.

use std::f32::consts::PI;
use stetho_core::dsp::mel::{LogMelSpectrogram, N_MELS};

fn main() {
    let sr = 4000.0_f32;
    let n = 16_000_usize; // 4 s
    let freq = 60.0_f32;
    let amp = 10_000.0_f32; // same scale as i16 audio
    let samples: Vec<f32> = (0..n)
        .map(|i| amp * (2.0 * PI * freq * (i as f32) / sr).sin())
        .collect();

    let mut m = LogMelSpectrogram::new(sr);
    let mut frames: Vec<[f32; N_MELS]> = Vec::new();
    m.process(&samples, &mut frames);

    println!("{} {}", frames.len(), N_MELS);
    for frame in &frames {
        for (i, v) in frame.iter().enumerate() {
            if i > 0 {
                print!(" ");
            }
            print!("{:.6}", v);
        }
        println!();
    }
}
