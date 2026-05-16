//! Direct-Form-I, f64 reference biquads for the formula parity test
//! of [`crate::dsp::biquad::Biquad`]. Not intended for hot-path use —
//! the optimised f32 `Biquad` is the production path. Coefficients are
//! the standard tan-prewarped Butterworth lowpass / highpass forms
//! (Bristow-Johnson, "Cookbook formulae for audio EQ biquads").

const SIN_PI_OVER_4: f64 = 0.7071067811865475; // sin(π/4) = √2/2

/// Direct-Form-I tan-prewarped Butterworth highpass, f64 throughout.
pub struct HighpassReference {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl HighpassReference {
    pub fn new(fs: u32, fc: u32) -> Self {
        let k = (std::f64::consts::PI * (fc as f64) / (fs as f64)).tan();
        let k2 = k * k;
        let denom = k2 + 2.0 * k * SIN_PI_OVER_4 + 1.0;
        let inv = 1.0 / denom;
        Self {
            b0: inv,
            b1: -2.0 * inv,
            b2: inv,
            a1: -2.0 * (1.0 - k2) / denom,
            a2: (k2 - 2.0 * k * SIN_PI_OVER_4 + 1.0) / denom,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

/// Direct-Form-I tan-prewarped Butterworth lowpass, f64.
///
/// Only the b-coefficients differ from the highpass form:
///   b0 = k²/denom, b1 = 2·k²/denom, b2 = k²/denom.
pub struct LowpassReference {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl LowpassReference {
    pub fn new(fs: u32, fc: u32) -> Self {
        let k = (std::f64::consts::PI * (fc as f64) / (fs as f64)).tan();
        let k2 = k * k;
        let denom = k2 + 2.0 * k * SIN_PI_OVER_4 + 1.0;
        let inv = 1.0 / denom;
        Self {
            b0: k2 * inv,
            b1: 2.0 * k2 * inv,
            b2: k2 * inv,
            a1: -2.0 * (1.0 - k2) / denom,
            a2: (k2 - 2.0 * k * SIN_PI_OVER_4 + 1.0) / denom,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    pub fn process_sample(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highpass_dc_blocked() {
        let mut f = HighpassReference::new(4000, 35);
        let mut last = 0.0;
        for _ in 0..4000 {
            last = f.process_sample(1.0);
        }
        assert!(last.abs() < 1e-3, "HP DC residue = {last}");
    }

    #[test]
    fn lowpass_dc_passes() {
        let mut f = LowpassReference::new(4000, 100);
        let mut last = 0.0;
        for _ in 0..4000 {
            last = f.process_sample(1.0);
        }
        assert!((last - 1.0).abs() < 1e-3, "LP DC gain = {last}");
    }
}
