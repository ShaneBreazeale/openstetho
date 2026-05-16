//! Mains-hum notch (50 Hz / 60 Hz).
//!
//! This module currently stubs out the API so the pipeline compiles. Add an
//! independently designed filter here once the notch requirements are fixed.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MainsFrequency {
    Off,
    Hz50,
    Hz60,
}

pub struct MainsNotch {
    _freq: MainsFrequency,
}

impl MainsNotch {
    pub fn new(freq: MainsFrequency) -> Self {
        Self { _freq: freq }
    }

    /// Currently a passthrough.
    pub fn process_block(&mut self, _buf: &mut [f32]) {}
}
