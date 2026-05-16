//! Public filter presets for compatible digital stethoscope streams.
//!
//! These chains use standard Butterworth stages with cutoff frequencies
//! documented from owned-device interoperability testing. Presets whose
//! behavior has not been independently characterized are left unavailable.

use crate::dsp::biquad::Biquad;
use crate::AUDIO_SAMPLE_RATE_HZ;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum EqPreset {
    None,
    Wide,
    Cardiac,
    Pulmonary,
    // Unavailable until independently characterized.
    Diaphragm,
    Bell,
    Midrange,
    Extended,
}

pub struct FilterChain {
    stages: Vec<Biquad>,
}

impl FilterChain {
    pub fn new(preset: EqPreset) -> Option<Self> {
        let fs = AUDIO_SAMPLE_RATE_HZ as f32;
        let stages = match preset {
            EqPreset::None => Vec::new(),
            EqPreset::Wide => vec![
                Biquad::highpass(35.0, fs),
                Biquad::highpass(55.0, fs),
                Biquad::lowpass(850.0, fs),
            ],
            EqPreset::Cardiac => vec![
                Biquad::highpass(35.0, fs),
                Biquad::highpass(55.0, fs),
                Biquad::lowpass(100.0, fs),
            ],
            EqPreset::Pulmonary => vec![
                Biquad::highpass(35.0, fs),
                Biquad::highpass(100.0, fs),
                Biquad::lowpass(1000.0, fs),
            ],
            EqPreset::Diaphragm | EqPreset::Bell | EqPreset::Midrange | EqPreset::Extended => {
                return None
            }
        };
        Some(Self { stages })
    }

    pub fn process_block(&mut self, buf: &mut [f32]) {
        for stage in self.stages.iter_mut() {
            stage.process_block(buf);
        }
    }

    pub fn reset(&mut self) {
        for stage in self.stages.iter_mut() {
            stage.reset();
        }
    }
}
