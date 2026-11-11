//! Escape-time calculation on the graphics card with wgpu compute shaders
//! (Vulkan, Metal or DirectX 12).

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::{cpu, shaders, Cancel, FractalParams};
use crate::view::View;

/// A graphics card that can be used for the calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct AdapterEntry {
    /// Stable identifier shown in the settings ("name (backend)").
    pub key: String,
    pub name: String,
    pub backend: String,
    pub kind: String,
    pub supports_f64: bool,
}

/// Floating-point precision used by the graphics card.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Precision {
    Single,
    /// One reference orbit in 64-bit precision on the processor; every pixel follows only its
    /// small difference from it, in fast single precision.
    Perturbation,
    DoubleEmulated,
    DoubleNative,
}

impl Precision {
    pub const ALL: [Precision; 4] = [
        Precision::Single,
        Precision::Perturbation,
        Precision::DoubleEmulated,
        Precision::DoubleNative,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Precision::Single => "singola precisione",
            Precision::Perturbation => "perturbazione (zoom profondo veloce)",
            Precision::DoubleEmulated => "doppia precisione emulata",
            Precision::DoubleNative => "doppia precisione nativa",
        }
    }

    /// Smallest pixel size (relative to the coordinate magnitude) this precision draws cleanly.
    pub fn min_relative_pixel(self) -> f64 {
        match self {
            Precision::Single => 2.0e-6,
            Precision::Perturbation => 0.0,
            Precision::DoubleEmulated => 1.0e-13,
            Precision::DoubleNative => 0.0,
        }
    }

    /// Rough cost of one iteration compared with single precision.
    fn cost(self) -> f64 {
        match self {
            Precision::Single => 1.0,
            Precision::Perturbation => 2.0,
            Precision::DoubleEmulated => 12.0,
            Precision::DoubleNative => 24.0,
        }
    }
}

/// The precision the person asked for in the settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrecisionChoice {
    Auto,
    Fixed(Precision),
}

impl PrecisionChoice {
    pub fn label(self) -> &'static str {
        match self {
            PrecisionChoice::Auto => "Automatica (in base allo zoom)",
            PrecisionChoice::Fixed(Precision::Single) => "Singola (più veloce)",
            PrecisionChoice::Fixed(Precision::Perturbation) => "Perturbazione (zoom profondo)",
            PrecisionChoice::Fixed(Precision::DoubleEmulated) => "Doppia emulata",
            PrecisionChoice::Fixed(Precision::DoubleNative) => "Doppia nativa",
        }
    }
}

/// How small a pixel is relative to the size of the coordinates.
pub fn relative_pixel(view: &View, height: u32) -> f64 {
    let mag = view.cx.abs().max(view.cy.abs()).max(2.0);
    view.pixel_size(height) / mag
}

fn backend_name(b: wgpu::Backend) -> &'static str {
    match b {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "DirectX 12",
        wgpu::Backend::Gl => "OpenGL",
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        _ => "?",
    }
}

fn kind_name(t: wgpu::DeviceType) -> &'static str {
    match t {
        wgpu::DeviceType::DiscreteGpu => "dedicata",
        wgpu::DeviceType::IntegratedGpu => "integrata",
        wgpu::DeviceType::VirtualGpu => "virtuale",
        wgpu::DeviceType::Cpu => "software",
        wgpu::DeviceType::Other => "altro",
    }
}

fn rank(info: &wgpu::AdapterInfo) -> (u8, u8) {
    let t = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => 0,
        wgpu::DeviceType::IntegratedGpu => 1,
        wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Other => 2,
        wgpu::DeviceType::Cpu => 3,
    };
    let b = match info.backend {
        wgpu::Backend::Vulkan | wgpu::Backend::Metal => 0,
        wgpu::Backend::Dx12 => 1,
        _ => 2,
    };
    (t, b)
}

fn new_instance() -> wgpu::Instance {
    wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        ..Default::default()
    })
}

fn sorted_adapters(instance: &wgpu::Instance) -> Vec<wgpu::Adapter> {
    let mut adapters: Vec<wgpu::Adapter> = instance
        .enumerate_adapters(wgpu::Backends::PRIMARY)
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect();
    adapters.sort_by_key(|a| rank(&a.get_info()));
    adapters
}

fn entry_of(adapter: &wgpu::Adapter) -> AdapterEntry {
    let info = adapter.get_info();
    let backend = backend_name(info.backend).to_string();
    AdapterEntry {
        key: format!("{} ({})", info.name, backend),
        name: info.name.clone(),
        backend,
        kind: kind_name(info.device_type).to_string(),
        supports_f64: adapter.features().contains(wgpu::Features::SHADER_F64),
    }
}

/// Graphics cards available on this computer, best first.
pub fn list_adapters() -> Vec<AdapterEntry> {
    sorted_adapters(&new_instance())
        .iter()
        .map(entry_of)
        .collect()
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    width: u32,
    height: u32,
    row_offset: u32,
    max_iter: u32,
    band_rows: u32,
    out_row0: u32,
    use_bulbs: u32,
    /// Always 0; hides values from the driver's optimiser (see the shaders).
    zero: u32,
    /// hi, mid, lo parts; the fourth value is the reference point's column offset in pixels.
    cx: [f32; 4],
    /// hi, mid, lo parts; the fourth value is the reference point's row offset in pixels.
    cy: [f32; 4],
    pix: [f32; 4],
    /// First column, number of columns of this dispatch, length of the reference orbit.
    cols: [u32; 4],
    /// Resumable kernels: 1 on the first dispatch, iterations per dispatch.
    run: [u32; 4],
}

/// Size of one pixel's saved state in the resumable kernel (see `State` in the shaders).
const STATE_BYTES: u64 = 40;

/// Split a 64-bit value into three 32-bit parts whose sum is the value.
fn split3(v: f64) -> [f32; 4] {
    let hi = v as f32;
    let r = v - hi as f64;
    let mid = r as f32;
    let lo = (r - mid as f64) as f32;
    [hi, mid, lo, 0.0]
}

/// An open graphics card with its compiled calculation programs.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub entry: AdapterEntry,
    layout: wgpu::BindGroupLayout,
    single: wgpu::ComputePipeline,
    perturbation: Option<wgpu::ComputePipeline>,
    emulated: Option<wgpu::ComputePipeline>,
    native: Option<wgpu::ComputePipeline>,
    /// Stand in for the reference orbit, the saved pixel states and the counter of
    /// unfinished pixels in the programs that don't use them.
    no_orbit: wgpu::Buffer,
    no_state: wgpu::Buffer,
    no_pending: wgpu::Buffer,
    lost: Arc<AtomicBool>,
    error: Arc<Mutex<Option<String>>>,
    max_binding: u64,
    /// What was found while setting the card up (shown in the settings).
    pub notes: Vec<String>,
}

impl GpuContext {
    /// Open the graphics card whose key is `preferred`, or the best one.
    pub fn new(preferred: Option<&str>) -> Result<GpuContext, String> {
        let instance = new_instance();
        let adapters = sorted_adapters(&instance);
        if adapters.is_empty() {
            return Err(
                "Nessuna scheda grafica compatibile (Vulkan, Metal o DirectX 12) trovata".into(),
            );
        }
        let adapter = preferred
            .and_then(|key| adapters.iter().find(|a| entry_of(a).key == key))
            .unwrap_or(&adapters[0]);
        let entry = entry_of(adapter);

        let mut features = wgpu::Features::empty();
        if entry.supports_f64 {
            features |= wgpu::Features::SHADER_F64;
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("mandelbrot-compute"),
                required_features: features,
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|e| format!("Impossibile aprire la scheda grafica: {e}"))?;

        let error = Arc::new(Mutex::new(None::<String>));
        {
            let error = error.clone();
            device.on_uncaptured_error(Box::new(move |e| {
                eprintln!("Errore della scheda grafica: {e}");
                *error.lock().unwrap() = Some(e.to_string());
            }));
        }
        let lost = Arc::new(AtomicBool::new(false));
        {
            let lost = lost.clone();
            device.set_device_lost_callback(move |_reason, msg| {
                eprintln!("Scheda grafica persa: {msg}");
                lost.store(true, Ordering::SeqCst);
            });
        }

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mandelbrot-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let small = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let no_orbit = small("mandelbrot-no-orbit", 16);
        let no_state = small("mandelbrot-no-state", STATE_BYTES);
        let no_pending = small("mandelbrot-no-pending", 16);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mandelbrot-pipeline-layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let build = |label: &str, src: String| -> Result<wgpu::ComputePipeline, String> {
            device.push_error_scope(wgpu::ErrorFilter::Internal);
            device.push_error_scope(wgpu::ErrorFilter::Validation);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            let validation = pollster::block_on(device.pop_error_scope());
            let internal = pollster::block_on(device.pop_error_scope());
            match validation.or(internal) {
                Some(e) => Err(e.to_string()),
                None => Ok(pipeline),
            }
        };

        let single = build("mandelbrot-f32", shaders::f32_source())?;
        let max_binding = (limits.max_storage_buffer_binding_size as u64)
            .min(limits.max_buffer_size)
            .max(4 * 1024 * 1024);

        let mut ctx = GpuContext {
            device: device.clone(),
            queue,
            entry,
            layout,
            single,
            perturbation: None,
            emulated: None,
            native: None,
            no_orbit,
            no_state,
            no_pending,
            lost,
            error,
            max_binding,
            notes: Vec::new(),
        };

        // Share of pixels that must match the processor's result.
        const PASS: f64 = 0.85;

        match build("mandelbrot-perturbation", shaders::perturbation_source()) {
            Ok(p) => {
                ctx.perturbation = Some(p);
                match ctx.self_test(Precision::Perturbation) {
                    Ok(score) if score >= PASS => {
                        ctx.notes.push("Perturbazione attiva (zoom profondi veloci)".into())
                    }
                    Ok(score) => {
                        ctx.perturbation = None;
                        ctx.notes.push(format!(
                            "Perturbazione non abbastanza precisa su questa scheda ({:.0}%)",
                            score * 100.0
                        ));
                    }
                    Err(e) => {
                        ctx.perturbation = None;
                        ctx.notes.push(format!("Perturbazione non riuscita: {e}"));
                    }
                }
            }
            Err(e) => ctx.notes.push(format!("Perturbazione non compilata: {e}")),
        }

        // Emulated double precision: try the FMA form first, then the splitting form, and keep
        // the first one that really gives double precision on this card.
        for (use_fma, name) in [(true, "FMA"), (false, "split")] {
            match build("mandelbrot-df", shaders::df_source(use_fma)) {
                Ok(p) => {
                    ctx.emulated = Some(p);
                    match ctx.self_test(Precision::DoubleEmulated) {
                        Ok(score) if score >= PASS => {
                            ctx.notes
                                .push(format!("Doppia precisione emulata attiva ({name})"));
                            break;
                        }
                        Ok(score) => {
                            if std::env::var_os("MANDELBROT_KEEP_EMULATED").is_none() {
                                ctx.emulated = None;
                            }
                            ctx.notes.push(format!(
                                "Doppia emulata ({name}) non abbastanza precisa su questa scheda ({:.0}% di pixel corretti)",
                                score * 100.0
                            ));
                        }
                        Err(e) => {
                            ctx.emulated = None;
                            ctx.notes
                                .push(format!("Doppia emulata ({name}) non riuscita: {e}"));
                        }
                    }
                }
                Err(e) => ctx
                    .notes
                    .push(format!("Doppia emulata ({name}) non compilata: {e}")),
            }
        }

        if ctx.entry.supports_f64 {
            match build("mandelbrot-f64", shaders::f64_source()) {
                Ok(p) => {
                    ctx.native = Some(p);
                    match ctx.self_test(Precision::DoubleNative) {
                        Ok(score) if score >= PASS => {
                            ctx.notes.push("Doppia precisione nativa attiva".into())
                        }
                        Ok(score) => {
                            ctx.native = None;
                            ctx.notes.push(format!(
                                "Doppia nativa non abbastanza precisa ({:.0}%)",
                                score * 100.0
                            ));
                        }
                        Err(e) => {
                            ctx.native = None;
                            ctx.notes.push(format!("Doppia nativa non riuscita: {e}"));
                        }
                    }
                }
                Err(e) => ctx.notes.push(format!("Doppia nativa non compilata: {e}")),
            }
        } else {
            ctx.notes
                .push("La scheda non supporta la doppia precisione nativa".into());
        }
        *ctx.error.lock().unwrap() = None;
        Ok(ctx)
    }

    pub fn has(&self, p: Precision) -> bool {
        match p {
            Precision::Single => true,
            Precision::Perturbation => self.perturbation.is_some(),
            Precision::DoubleEmulated => self.emulated.is_some(),
            Precision::DoubleNative => self.native.is_some(),
        }
    }

    pub fn is_lost(&self) -> bool {
        self.lost.load(Ordering::SeqCst)
    }

    /// The precision to use for `view`, or `None` when no precision on this card is enough.
    pub fn pick_precision(
        &self,
        choice: PrecisionChoice,
        view: &View,
        height: u32,
    ) -> Option<Precision> {
        match choice {
            PrecisionChoice::Fixed(p) if self.has(p) => Some(p),
            _ => {
                let rel = relative_pixel(view, height);
                Precision::ALL
                    .into_iter()
                    .find(|p| self.has(*p) && rel >= p.min_relative_pixel())
            }
        }
    }

    /// Compare a deep-zoom image against the processor's result.
    fn self_test(&self, precision: Precision) -> Result<f64, String> {
        let params = FractalParams {
            view: View {
                cx: -0.743_643_887_037_158_7,
                cy: 0.131_825_904_205_311_97,
                scale: 4.0e-10,
            },
            max_iter: 4000,
        };
        let (w, h) = (48u32, 32u32);
        let never = || false;
        let gpu = self
            .render_with(precision, &params, w, h, &never, None)?
            .ok_or("annullato")?;
        let reference = cpu::render(&params, w, h, &never, None).unwrap_or_default();
        let escaped = reference.iter().filter(|v| **v >= 0.0).count();
        if escaped < reference.len() / 3 {
            // The reference has too little detail to judge; trust the card.
            return Ok(1.0);
        }
        let matching = gpu
            .iter()
            .zip(&reference)
            .filter(|(a, b)| {
                (**a < 0.0 && **b < 0.0) || (**a >= 0.0 && **b >= 0.0 && (**a - **b).abs() < 2.0)
            })
            .count();
        Ok(matching as f64 / reference.len().max(1) as f64)
    }

    /// Calculate a `w`×`h` image with the given precision. `Ok(None)` when cancelled.
    pub fn render_with(
        &self,
        precision: Precision,
        p: &FractalParams,
        w: u32,
        h: u32,
        cancel: &Cancel<'_>,
        progress: Option<&AtomicU32>,
    ) -> Result<Option<Vec<f32>>, String> {
        if self.is_lost() {
            return Err("La scheda grafica non risponde più".into());
        }
        let pipeline = match precision {
            Precision::Single => &self.single,
            Precision::Perturbation => self
                .perturbation
                .as_ref()
                .ok_or("perturbazione non disponibile")?,
            Precision::DoubleEmulated => self
                .emulated
                .as_ref()
                .ok_or("doppia emulata non disponibile")?,
            Precision::DoubleNative => self
                .native
                .as_ref()
                .ok_or("doppia nativa non disponibile")?,
        };
        let (wu, hu) = (w as usize, h as usize);
        let mut out = vec![0f32; wu * hu];

        // Rows that fit in one storage buffer (at most 64 MiB at a time).
        let max_bytes = self.max_binding.min(64 * 1024 * 1024);
        let mut max_pixels = max_bytes / 4;
        let resumable = precision == Precision::Perturbation;
        if resumable {
            // Each pixel also keeps its saved state on the card.
            max_pixels = max_pixels.min(self.max_binding / STATE_BYTES).min(4 << 20);
        }
        let chunk_rows = (max_pixels / w as u64).clamp(1, h as u64) as u32;
        let chunk_bytes = chunk_rows as u64 * w as u64 * 4;
        let state = resumable.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mandelbrot-state"),
                size: chunk_rows as u64 * w as u64 * STATE_BYTES,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let pending = resumable.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mandelbrot-pending"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let pending_read = resumable.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mandelbrot-pending-read"),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        let storage = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot-out"),
            size: chunk_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot-readback"),
            size: chunk_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mandelbrot-params"),
            size: std::mem::size_of::<GpuParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Perturbation: pick the slowest-escaping point as the reference and send its orbit.
        let pix = p.view.pixel_size(h);
        let mut reference = ([0.0f32, 0.0f32], 0u32);
        let orbit_buffer = if precision == Precision::Perturbation {
            let (rx, ry) = super::perturb::reference_point(&p.view, w, h, p.max_iter, cancel);
            if cancel() {
                return Ok(None);
            }
            let max_len = (self.max_binding / 8).max(2) as usize;
            let orbit = super::perturb::reference_orbit(
                p.view.cx + rx * pix,
                p.view.cy - ry * pix,
                p.max_iter,
                max_len,
            );
            reference = ([rx as f32, ry as f32], orbit.len() as u32);
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mandelbrot-orbit"),
                size: (orbit.len() * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.queue
                .write_buffer(&buffer, 0, bytemuck::cast_slice(&orbit));
            Some(buffer)
        } else {
            None
        };
        let orbit_binding = orbit_buffer.as_ref().unwrap_or(&self.no_orbit);

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mandelbrot-bind"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: orbit_binding.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: state.as_ref().unwrap_or(&self.no_state).as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pending.as_ref().unwrap_or(&self.no_pending).as_entire_binding(),
                },
            ],
        });

        let mut cx = split3(p.view.cx);
        let mut cy = split3(p.view.cy);
        cx[3] = reference.0[0];
        cy[3] = reference.0[1];
        let mut params = GpuParams {
            width: w,
            height: h,
            row_offset: 0,
            max_iter: p.max_iter,
            band_rows: 0,
            out_row0: 0,
            // The cardioid test runs in single precision: only trust it away from deep zooms.
            use_bulbs: (pix > 1e-5) as u32,
            zero: 0,
            cx,
            cy,
            pix: split3(pix),
            cols: [0, w, reference.1, 0],
            run: [0, 0, 0, 0],
        };

        // Work is sent in pieces short enough (about a tenth of a second) that the system
        // never thinks the card has hung, however many iterations are asked for: several
        // rows at a time, or part of a row when the iterations are very high.
        // A piece never goes below MIN_PIECE pixels: smaller pieces would leave most of the
        // card idle, while one pixel's own iterations already set a floor on the time.
        const MIN_PIECE: f64 = 8192.0;
        let target = Duration::from_millis(120);
        let est_rate = 6.0e9 / precision.cost(); // iterations per 120 ms, a cautious first guess
        let mut budget = (est_rate / p.max_iter.max(1) as f64).max(MIN_PIECE); // pixels per piece
        // Resumable kernel: iterations per dispatch, first guessed so that even a picture
        // where every pixel needs them all stays within the target time.
        let mut iter_slice = 0f64;

        let mut chunk_start = 0u32;
        while chunk_start < h {
            let rows = chunk_rows.min(h - chunk_start);
            if let (true, Some(pending), Some(pending_read)) = (resumable, &pending, &pending_read) {
                if iter_slice == 0.0 {
                    iter_slice = (est_rate / (rows as f64 * w as f64)).clamp(64.0, 1.0e6);
                }
                params.row_offset = chunk_start;
                params.out_row0 = 0;
                params.band_rows = rows;
                params.cols = [0, w, reference.1, 0];
                let mut first = true;
                loop {
                    if cancel() {
                        return Ok(None);
                    }
                    params.run = [first as u32, iter_slice as u32, 0, 0];
                    self.queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                    self.queue.write_buffer(pending, 0, &[0u8; 4]);
                    let mut encoder = self
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("mandelbrot-slice") });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("mandelbrot-pass"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups(w.div_ceil(8), rows.div_ceil(8), 1);
                    }
                    encoder.copy_buffer_to_buffer(pending, 0, pending_read, 0, 4);
                    let started = Instant::now();
                    self.queue.submit(Some(encoder.finish()));
                    let slice = pending_read.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |res| {
                        let _ = tx.send(res);
                    });
                    self.device.poll(wgpu::Maintain::Wait);
                    let took = started.elapsed();
                    rx.recv()
                        .map_err(|_| "lettura dalla scheda grafica interrotta".to_string())?
                        .map_err(|e| format!("lettura dalla scheda grafica non riuscita: {e}"))?;
                    let left = u32::from_le_bytes(slice.get_mapped_range()[..4].try_into().unwrap_or([0; 4]));
                    pending_read.unmap();
                    self.check()?;
                    first = false;
                    if let Some(pr) = progress {
                        let total = (rows as u64 * w as u64).max(1);
                        let done_rows = (rows as u64 * (total - (left as u64).min(total)) / total) as u32;
                        pr.store(chunk_start + done_rows, Ordering::Relaxed);
                    }
                    if left == 0 {
                        break;
                    }
                    // Fewer pixels left means each dispatch is quicker: give the next one
                    // more iterations, always aiming at the target time.
                    let ratio = target.as_secs_f64() / took.as_secs_f64().max(1e-4);
                    iter_slice = (iter_slice * ratio.clamp(0.25, 4.0)).clamp(64.0, 1.0e7);
                }
            }
            let mut r = if resumable { rows } else { 0u32 };
            let mut col = 0u32;
            while r < rows {
                if cancel() {
                    return Ok(None);
                }
                let (br, c0, cw) = if col == 0 && budget >= w as f64 {
                    (((budget / w as f64) as u32).clamp(1, rows - r), 0, w)
                } else {
                    (1, col, (budget as u32).min(w - col))
                };
                params.row_offset = chunk_start + r;
                params.out_row0 = r;
                params.band_rows = br;
                params.cols = [c0, cw, reference.1, 0];
                self.queue
                    .write_buffer(&uniform, 0, bytemuck::bytes_of(&params));
                let mut encoder =
                    self.device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("mandelbrot-band"),
                        });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("mandelbrot-pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.dispatch_workgroups(cw.div_ceil(8), br.div_ceil(8), 1);
                }
                let started = Instant::now();
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                let took = started.elapsed();
                self.check()?;
                if c0 + cw >= w {
                    r += br;
                    col = 0;
                    if let Some(pr) = progress {
                        pr.fetch_add(br, Ordering::Relaxed);
                    }
                } else {
                    col = c0 + cw;
                }
                // Adapt the size of the next piece to the measured speed.
                let ratio = target.as_secs_f64() / took.as_secs_f64().max(1e-4);
                budget = (br as f64 * cw as f64 * ratio.clamp(0.25, 4.0)).max(MIN_PIECE);
            }

            let bytes = rows as u64 * w as u64 * 4;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mandelbrot-copy"),
                });
            encoder.copy_buffer_to_buffer(&storage, 0, &staging, 0, bytes);
            self.queue.submit(Some(encoder.finish()));
            let slice = staging.slice(0..bytes);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.recv()
                .map_err(|_| "lettura dalla scheda grafica interrotta".to_string())?
                .map_err(|e| format!("lettura dalla scheda grafica non riuscita: {e}"))?;
            {
                let mapped = slice.get_mapped_range();
                let start = chunk_start as usize * wu;
                out[start..start + rows as usize * wu]
                    .copy_from_slice(bytemuck::cast_slice(&mapped));
            }
            staging.unmap();
            self.check()?;
            chunk_start += rows;
        }
        Ok(Some(out))
    }

    fn check(&self) -> Result<(), String> {
        if self.is_lost() {
            return Err("La scheda grafica non risponde più".into());
        }
        if let Some(e) = self.error.lock().unwrap().take() {
            return Err(format!("Errore della scheda grafica: {e}"));
        }
        Ok(())
    }
}
