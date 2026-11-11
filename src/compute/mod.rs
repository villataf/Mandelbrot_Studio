//! Choosing between the processor and the graphics card for the calculation.

pub mod cpu;
pub mod gpu;
pub mod perturb;
pub mod shaders;

use std::sync::atomic::AtomicU32;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

use crate::view::View;
use gpu::{GpuContext, PrecisionChoice};

/// Value of a point proven to be inside the set.
pub const INTERIOR: f32 = -1.0;
/// Value of a point that did not escape within the iteration limit.
pub const UNKNOWN: f32 = -2.0;

/// Returns true when the work should stop.
pub type Cancel<'a> = dyn Fn() -> bool + Sync + 'a;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FractalParams {
    pub view: View,
    pub max_iter: u32,
}

/// Where the calculation runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Engine {
    Cpu,
    Gpu,
}

impl Engine {
    pub fn label(self) -> &'static str {
        match self {
            Engine::Cpu => "Processore (CPU)",
            Engine::Gpu => "Scheda grafica (GPU)",
        }
    }
}

/// A finished calculation.
pub struct Computed {
    pub data: Vec<f32>,
    /// Plain words for what did the work, e.g. "GPU · RTX · doppia precisione emulata".
    pub used: String,
}

/// State of the graphics card set-up.
pub enum GpuState {
    NotStarted,
    Starting,
    Ready(Arc<GpuContext>),
    Failed(String),
}

/// Shared by the viewer and the exporters.
pub struct ComputeHub {
    pub gpu: RwLock<GpuState>,
}

impl ComputeHub {
    pub fn new() -> Arc<ComputeHub> {
        Arc::new(ComputeHub {
            gpu: RwLock::new(GpuState::NotStarted),
        })
    }

    /// Open the graphics card in the background.
    pub fn start_gpu(
        self: &Arc<Self>,
        preferred: Option<String>,
        on_done: impl FnOnce() + Send + 'static,
    ) {
        *self.gpu.write().unwrap() = GpuState::Starting;
        let hub = self.clone();
        std::thread::Builder::new()
            .name("gpu-setup".into())
            .spawn(move || {
                let state = match GpuContext::new(preferred.as_deref()) {
                    Ok(ctx) => GpuState::Ready(Arc::new(ctx)),
                    Err(e) => GpuState::Failed(e),
                };
                *hub.gpu.write().unwrap() = state;
                on_done();
            })
            .expect("thread");
    }

    pub fn gpu_context(&self) -> Option<Arc<GpuContext>> {
        match &*self.gpu.read().unwrap() {
            GpuState::Ready(ctx) if !ctx.is_lost() => Some(ctx.clone()),
            _ => None,
        }
    }

    /// Calculate an image with the chosen engine, falling back to the processor when the
    /// graphics card can't do it (missing, busy setting up, or not precise enough).
    pub fn render(
        &self,
        params: &FractalParams,
        w: u32,
        h: u32,
        engine: Engine,
        precision: PrecisionChoice,
        cancel: &Cancel<'_>,
        progress: Option<&AtomicU32>,
    ) -> Result<Option<Computed>, String> {
        let mut note = "";
        if engine == Engine::Gpu {
            match self.gpu_context() {
                Some(ctx) => match ctx.pick_precision(precision, &params.view, h) {
                    Some(p) => match ctx.render_with(p, params, w, h, cancel, progress) {
                        Ok(Some(data)) => {
                            return Ok(Some(Computed {
                                data,
                                used: format!("GPU · {} · {}", ctx.entry.name, p.label()),
                            }))
                        }
                        Ok(None) => return Ok(None),
                        Err(e) => {
                            eprintln!("{e}");
                            *self.gpu.write().unwrap() = GpuState::Failed(e);
                            note = " (errore della GPU)";
                        }
                    },
                    None => note = " (zoom oltre la precisione della GPU)",
                },
                None => note = " (GPU non disponibile)",
            }
        }
        Ok(
            cpu::render(params, w, h, cancel, progress).map(|data| Computed {
                data,
                used: format!(
                    "CPU · {} core · doppia precisione{note}",
                    rayon::current_num_threads()
                ),
            }),
        )
    }
}
