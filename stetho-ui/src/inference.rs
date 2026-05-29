//! Inference dispatch.
//!
//! Loads a Core ML `.mlpackage` mel-spec classifier and accumulates a
//! configurable number of mel frames before each prediction. Inputs:
//! `(1, 1, n_frames, N_MELS)` f32. Output: a single sigmoid logit.
//!
//! The runtime hosts two engines in parallel:
//!
//! * `MurmurEngine` — window length is taken from the model's sidecar
//!   `n_frames` metadata at load time (`murmur_frame_count`), so a single
//!   build serves any trained window. The current release model is the 5 s /
//!   78-frame cnn_bigru ensemble; `N_FRAMES_MURMUR = 62` (4 s) remains only the
//!   legacy fallback for older packages that ship no metadata.
//! * `S3Engine` — `N_FRAMES_S3 = 23` frames per window (1.5 s S2-anchored
//!   crop). Matches `S3CNN_v2`'s training shape.
//!
//! Both engines consume the *same* z-scored mel frame stream produced by
//! `LogMelSpectrogram::process_with_display`. Each one buffers
//! independently and emits a probability whenever its window is full.

use std::path::Path;
use stetho_core::dsp::mel::N_MELS;
use tracing::{info, warn};

/// Legacy fallback window length (4 s @ hop 256 / 4 kHz) used only when a
/// model ships no `n_frames` metadata. The shipped model's window length comes
/// from its sidecar instead — see `murmur_frame_count` in `main.rs`.
pub const N_FRAMES_MURMUR: usize = 62;
pub const N_FRAMES_S3: usize = 23;

/// Default for code that hasn't been updated to choose a window length —
/// historically the murmur engine's frame count.
#[allow(dead_code)]
pub const N_FRAMES: usize = N_FRAMES_MURMUR;

#[cfg(target_os = "macos")]
mod ml {
    use super::*;
    use coreml_native::{compile_model, BorrowedTensor, ComputeUnits, Model};

    pub struct MelEngine {
        model: Model,
        buf: Vec<f32>,
        input_name: String,
        output_name: String,
        n_frames: usize,
    }

    impl MelEngine {
        pub fn load(path: &Path, n_frames: usize) -> anyhow::Result<Self> {
            let needs_compile = path
                .extension()
                .map(|e| e.eq_ignore_ascii_case("mlpackage") || e.eq_ignore_ascii_case("mlmodel"))
                .unwrap_or(false);
            let load_path = if needs_compile {
                let compiled = compile_model(path)
                    .map_err(|e| anyhow::anyhow!("compile_model({}): {e}", path.display()))?;
                info!("compiled {} → {}", path.display(), compiled.display());
                compiled
            } else {
                path.to_path_buf()
            };
            let model = Model::load(&load_path, ComputeUnits::All)
                .map_err(|e| anyhow::anyhow!("Model::load({}): {e}", load_path.display()))?;
            let input_name = model
                .inputs()
                .first()
                .map(|f| f.name().to_string())
                .ok_or_else(|| anyhow::anyhow!("model has no input"))?;
            let output_name = model
                .outputs()
                .first()
                .map(|f| f.name().to_string())
                .ok_or_else(|| anyhow::anyhow!("model has no output"))?;
            info!(
                "loaded Core ML model {} ({} frames) — input '{}' output '{}'",
                path.display(),
                n_frames,
                input_name,
                output_name,
            );
            Ok(Self {
                model,
                buf: Vec::with_capacity(n_frames * N_MELS),
                input_name,
                output_name,
                n_frames,
            })
        }

        #[allow(dead_code)]
        pub fn n_frames(&self) -> usize {
            self.n_frames
        }

        pub fn push_frame(&mut self, frame: &[f32; N_MELS]) -> Option<f32> {
            self.buf.extend_from_slice(frame);
            if self.buf.len() < self.n_frames * N_MELS {
                return None;
            }
            let prob = self.predict_window();
            // 50 % overlap between successive windows so we get a fresh
            // prediction every n_frames/2 hops without waiting for a full
            // disjoint window.
            self.buf.drain(..(self.n_frames * N_MELS / 2));
            prob
        }

        fn predict_window(&self) -> Option<f32> {
            let shape = [1usize, 1, self.n_frames, N_MELS];
            let slice = &self.buf[..self.n_frames * N_MELS];
            let tensor = BorrowedTensor::from_f32(slice, &shape)
                .map_err(|e| warn!("BorrowedTensor::from_f32: {e}"))
                .ok()?;
            let result = self
                .model
                .predict(&[(self.input_name.as_str(), &tensor)])
                .map_err(|e| warn!("Core ML predict failed: {e}"))
                .ok()?;
            let (data, _shape) = result
                .get_f32(&self.output_name)
                .map_err(|e| warn!("Core ML get_f32: {e}"))
                .ok()?;
            let logit = *data.first()?;
            Some(sigmoid(logit))
        }
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    pub type Engine = MelEngine;
}

#[cfg(target_os = "macos")]
pub use ml::Engine as MurmurEngine;
#[cfg(target_os = "macos")]
pub type S3Engine = ml::Engine;

#[cfg(not(target_os = "macos"))]
pub struct MurmurEngine;
#[cfg(not(target_os = "macos"))]
impl MurmurEngine {
    pub fn load(_path: &Path, _n_frames: usize) -> anyhow::Result<Self> {
        anyhow::bail!("inference only available on macOS / iOS")
    }
    pub fn push_frame(&mut self, _frame: &[f32; N_MELS]) -> Option<f32> {
        None
    }
    pub fn n_frames(&self) -> usize {
        0
    }
}
#[cfg(not(target_os = "macos"))]
pub type S3Engine = MurmurEngine;
