//! Inference dispatch.
//!
//! Loads a Core ML `.mlpackage` mel-spec classifier and accumulates
//! `N_FRAMES` mel frames before each prediction. Inputs:
//! `(1, 1, N_FRAMES, N_MELS)` f32. Output: a single sigmoid logit.

use std::path::Path;
use stetho_core::dsp::mel::N_MELS;
use tracing::{info, warn};

pub const N_FRAMES: usize = 62;

#[cfg(target_os = "macos")]
mod ml {
    use super::*;
    use coreml_native::{compile_model, BorrowedTensor, ComputeUnits, Model};

    pub struct MelEngine {
        model: Model,
        buf: Vec<f32>,
        input_name: String,
        output_name: String,
    }

    impl MelEngine {
        pub fn load(path: &Path) -> anyhow::Result<Self> {
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
                "loaded Core ML model {} — input '{}' output '{}'",
                path.display(),
                input_name,
                output_name,
            );
            Ok(Self {
                model,
                buf: Vec::with_capacity(N_FRAMES * N_MELS),
                input_name,
                output_name,
            })
        }

        pub fn push_frame(&mut self, frame: &[f32; N_MELS]) -> Option<f32> {
            self.buf.extend_from_slice(frame);
            if self.buf.len() < N_FRAMES * N_MELS {
                return None;
            }
            let prob = self.predict_window();
            self.buf.drain(..(N_FRAMES * N_MELS / 2));
            prob
        }

        fn predict_window(&self) -> Option<f32> {
            let shape = [1usize, 1, N_FRAMES, N_MELS];
            let slice = &self.buf[..N_FRAMES * N_MELS];
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

#[cfg(not(target_os = "macos"))]
pub struct MurmurEngine;
#[cfg(not(target_os = "macos"))]
impl MurmurEngine {
    pub fn load(_path: &Path) -> anyhow::Result<Self> {
        anyhow::bail!("inference only available on macOS / iOS")
    }
    pub fn push_frame(&mut self, _frame: &[f32; N_MELS]) -> Option<f32> {
        None
    }
}
