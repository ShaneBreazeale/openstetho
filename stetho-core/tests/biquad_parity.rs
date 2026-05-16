//! Parity test: optimised f32 DF-II-T [`Biquad`] vs literal f64 DF-I
//! Butterworth reference formulas. We don't expect bit-equal floats,
//! only that the difference stays within ~1 LSB of an i16 output after
//! a long signal so the audible result is indistinguishable.

use stetho_core::dsp::biquad::Biquad;
use stetho_core::dsp::biquad_reference::{HighpassReference, LowpassReference};

const FS: u32 = 4000;
const N: usize = 16_000; // 4 seconds at 4 kHz

fn sine(freq: f64, fs: u32, n: usize, amp: f64) -> Vec<f64> {
    let dt = 1.0 / fs as f64;
    (0..n)
        .map(|i| amp * (2.0 * std::f64::consts::PI * freq * (i as f64) * dt).sin())
        .collect()
}

fn linear_sweep(fs: u32, n: usize, f0: f64, f1: f64, amp: f64) -> Vec<f64> {
    let dt = 1.0 / fs as f64;
    let mut phase: f64 = 0.0;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / n as f64;
        let f = f0 + (f1 - f0) * t;
        out.push(amp * phase.sin());
        phase += 2.0 * std::f64::consts::PI * f * dt;
    }
    out
}

fn impulse(n: usize, amp: f64) -> Vec<f64> {
    let mut v = vec![0.0; n];
    v[0] = amp;
    v
}

fn step(n: usize, amp: f64) -> Vec<f64> {
    vec![amp; n]
}

/// Run `f32` biquad through identical input as `f64` reference, return
/// peak absolute difference and RMS difference in i16-scale samples.
fn diff_against_reference_hp(fc: u32, input: &[f64]) -> (f64, f64) {
    let mut rust = Biquad::highpass(fc as f32, FS as f32);
    let mut reference = HighpassReference::new(FS, fc);
    let mut peak = 0.0_f64;
    let mut sum2 = 0.0_f64;
    for &x in input {
        let y_rust = rust.process_sample(x as f32) as f64;
        let y_reference = reference.process_sample(x);
        let d = (y_rust - y_reference).abs();
        peak = peak.max(d);
        sum2 += d * d;
    }
    (peak, (sum2 / input.len() as f64).sqrt())
}

fn diff_against_reference_lp(fc: u32, input: &[f64]) -> (f64, f64) {
    let mut rust = Biquad::lowpass(fc as f32, FS as f32);
    let mut reference = LowpassReference::new(FS, fc);
    let mut peak = 0.0_f64;
    let mut sum2 = 0.0_f64;
    for &x in input {
        let y_rust = rust.process_sample(x as f32) as f64;
        let y_reference = reference.process_sample(x);
        let d = (y_rust - y_reference).abs();
        peak = peak.max(d);
        sum2 += d * d;
    }
    (peak, (sum2 / input.len() as f64).sqrt())
}

const TOL_PEAK: f64 = 5.0; // i16 LSBs
const TOL_RMS: f64 = 1.0;

#[test]
fn highpass_35hz_matches_reference_on_sine() {
    let input = sine(60.0, FS, N, 10_000.0);
    let (peak, rms) = diff_against_reference_hp(35, &input);
    println!("HP35  sine60Hz   peak={peak:.3}  rms={rms:.3}");
    assert!(peak < TOL_PEAK, "HP35 peak diff {peak} > {TOL_PEAK} LSB");
    assert!(rms < TOL_RMS, "HP35 rms diff {rms} > {TOL_RMS} LSB");
}

#[test]
fn highpass_55hz_matches_reference_on_step() {
    let input = step(N, 10_000.0);
    let (peak, rms) = diff_against_reference_hp(55, &input);
    println!("HP55  step        peak={peak:.3}  rms={rms:.3}");
    assert!(peak < TOL_PEAK);
    assert!(rms < TOL_RMS);
}

#[test]
fn lowpass_100hz_matches_reference_on_impulse() {
    let input = impulse(N, 32000.0);
    let (peak, rms) = diff_against_reference_lp(100, &input);
    println!("LP100 impulse     peak={peak:.3}  rms={rms:.3}");
    assert!(peak < TOL_PEAK);
    assert!(rms < TOL_RMS);
}

#[test]
fn lowpass_850hz_matches_reference_on_sweep() {
    let input = linear_sweep(FS, N, 20.0, 1500.0, 10_000.0);
    let (peak, rms) = diff_against_reference_lp(850, &input);
    println!("LP850 sweep       peak={peak:.3}  rms={rms:.3}");
    assert!(peak < TOL_PEAK);
    assert!(rms < TOL_RMS);
}

#[test]
fn cardiac_chain_matches_reference_within_3lsb() {
    // Chain: HP35 → HP55 → LP100.
    let mut rust = vec![
        Biquad::highpass(35.0, FS as f32),
        Biquad::highpass(55.0, FS as f32),
        Biquad::lowpass(100.0, FS as f32),
    ];
    let mut reference_hp35 = HighpassReference::new(FS, 35);
    let mut reference_hp55 = HighpassReference::new(FS, 55);
    let mut reference_lp100 = LowpassReference::new(FS, 100);
    let input = sine(70.0, FS, N, 10_000.0);
    let mut peak = 0.0_f64;
    let mut sum2 = 0.0_f64;
    for &x in &input {
        let mut yr = x as f32;
        for b in rust.iter_mut() {
            yr = b.process_sample(yr);
        }
        let y_reference = reference_lp100
            .process_sample(reference_hp55.process_sample(reference_hp35.process_sample(x)));
        let d = (yr as f64 - y_reference).abs();
        peak = peak.max(d);
        sum2 += d * d;
    }
    let rms = (sum2 / input.len() as f64).sqrt();
    println!("CARDIAC chain     peak={peak:.3}  rms={rms:.3}");
    assert!(peak < 15.0, "Cardiac chain peak diff {peak} too high");
    assert!(rms < 3.0, "Cardiac chain rms diff {rms} too high");
}
