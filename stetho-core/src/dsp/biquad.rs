//! Tan-normalized Butterworth biquads (HP / LP).
//!
//! Direct-form II transposed for numerical stability on f32.

use std::f32::consts::PI;

const SQRT2_OVER_2: f32 = std::f32::consts::FRAC_1_SQRT_2;

#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// 2nd-order Butterworth lowpass at `fc` Hz, sampled at `fs` Hz.
    pub fn lowpass(fc: f32, fs: f32) -> Self {
        let k = (PI * fc / fs).tan();
        let q = SQRT2_OVER_2;
        let norm = 1.0 / (1.0 + k / q + k * k);
        let b0 = k * k * norm;
        let b1 = 2.0 * b0;
        let b2 = b0;
        let a1 = 2.0 * (k * k - 1.0) * norm;
        let a2 = (1.0 - k / q + k * k) * norm;
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// 2nd-order Butterworth highpass at `fc` Hz, sampled at `fs` Hz.
    pub fn highpass(fc: f32, fs: f32) -> Self {
        let k = (PI * fc / fs).tan();
        let q = SQRT2_OVER_2;
        let norm = 1.0 / (1.0 + k / q + k * k);
        let b0 = 1.0 * norm;
        let b1 = -2.0 * norm;
        let b2 = 1.0 * norm;
        let a1 = 2.0 * (k * k - 1.0) * norm;
        let a2 = (1.0 - k / q + k * k) * norm;
        Self {
            b0,
            b1,
            b2,
            a1,
            a2,
            z1: 0.0,
            z2: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let y = self.b0 * x + self.z1;
        self.z1 = self.b1 * x - self.a1 * y + self.z2;
        self.z2 = self.b2 * x - self.a2 * y;
        y
    }

    pub fn process_block(&mut self, buf: &mut [f32]) {
        for s in buf.iter_mut() {
            *s = self.process_sample(*s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dc_passes_through_lowpass() {
        let mut f = Biquad::lowpass(100.0, 4000.0);
        let mut last = 0.0;
        for _ in 0..2000 {
            last = f.process_sample(1.0);
        }
        assert_relative_eq!(last, 1.0, max_relative = 1e-3);
    }

    #[test]
    fn dc_blocked_by_highpass() {
        let mut f = Biquad::highpass(35.0, 4000.0);
        let mut last = 0.0;
        for _ in 0..4000 {
            last = f.process_sample(1.0);
        }
        assert!(last.abs() < 1e-2, "expected ~0, got {last}");
    }

    #[test]
    fn nyquist_blocked_by_lowpass() {
        let mut f = Biquad::lowpass(100.0, 4000.0);
        let mut sum = 0.0;
        for n in 0..1000 {
            let x = if n % 2 == 0 { 1.0 } else { -1.0 };
            let y = f.process_sample(x);
            if n > 500 {
                sum += y.abs();
            }
        }
        assert!(sum / 500.0 < 0.05, "got mean |y| = {}", sum / 500.0);
    }
}
