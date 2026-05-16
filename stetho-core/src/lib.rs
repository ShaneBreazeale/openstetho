//! Protocol + DSP core for compatible digital stethoscopes.
//!
//! Unofficial interoperability implementation. No vendor code, binaries, or
//! model bytes are vendored here.

pub mod ble;
pub mod codec;
pub mod dsp;

pub use codec::adpcm::AdpcmDecoder;
pub use dsp::biquad::Biquad;
pub use dsp::presets::EqPreset;

/// Audio sample rate emitted by compatible E4-class devices, in Hz.
pub const AUDIO_SAMPLE_RATE_HZ: u32 = 4000;

/// Pipeline block size in samples.
pub const BLOCK_SIZE: usize = 256;
