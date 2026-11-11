//! Long tasks that run in the background: saving a picture and exporting an animation.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::animation::Animation;
use crate::compute::gpu::PrecisionChoice;
use crate::compute::{ComputeHub, Engine, FractalParams};
use crate::export::{self, ImageOptions};
use crate::palette::{colorize_supersampled, ColorMap, Lut};
use crate::video::{self, Sink, VideoSettings};

/// Largest number of samples calculated for one picture (limits memory use).
pub const MAX_SAMPLES: u64 = 260_000_000;

/// Progress shared between a background task and the window.
#[derive(Default)]
pub struct JobStatus {
    pub title: String,
    pub cancel: AtomicBool,
    pub done: AtomicBool,
    /// 0..=1000
    pub permille: AtomicU32,
    pub rows_done: AtomicU32,
    pub message: Mutex<String>,
    pub result: Mutex<Option<Result<String, String>>>,
    pub started: Mutex<Option<Instant>>,
}

impl JobStatus {
    pub fn new(title: &str) -> Arc<JobStatus> {
        let s = JobStatus {
            title: title.into(),
            ..Default::default()
        };
        *s.started.lock().unwrap() = Some(Instant::now());
        Arc::new(s)
    }

    pub fn fraction(&self) -> f32 {
        self.permille.load(Ordering::Relaxed) as f32 / 1000.0
    }

    fn set_fraction(&self, f: f64) {
        self.permille
            .store((f.clamp(0.0, 1.0) * 1000.0) as u32, Ordering::Relaxed);
    }

    fn set_message(&self, m: impl Into<String>) {
        *self.message.lock().unwrap() = m.into();
    }

    fn finish(&self, r: Result<String, String>) {
        *self.result.lock().unwrap() = Some(r);
        self.done.store(true, Ordering::SeqCst);
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// How and with what a frame is calculated.
#[derive(Clone, Copy)]
pub struct ComputeChoice {
    pub engine: Engine,
    pub precision: PrecisionChoice,
}

/// Calculate and colour one RGB picture at `w`×`h` with supersampling.
fn render_rgb(
    hub: &ComputeHub,
    choice: ComputeChoice,
    params: &FractalParams,
    w: u32,
    h: u32,
    ss: u32,
    lut: &Lut,
    map: ColorMap,
    status: &JobStatus,
) -> Result<Option<(Vec<u8>, String)>, String> {
    let ss = ss.max(1);
    let (bw, bh) = (w * ss, h * ss);
    status.rows_done.store(0, Ordering::Relaxed);
    let cancel = || status.cancelled();
    let computed = hub.render(
        params,
        bw,
        bh,
        choice.engine,
        choice.precision,
        &cancel,
        Some(&status.rows_done),
    )?;
    Ok(computed.map(|c| {
        (
            colorize_supersampled(&c.data, w as usize, h as usize, ss as usize, lut, map),
            c.used,
        )
    }))
}

/// Save a picture in the background.
#[allow(clippy::too_many_arguments)]
pub fn start_image_export(
    hub: Arc<ComputeHub>,
    choice: ComputeChoice,
    params: FractalParams,
    w: u32,
    h: u32,
    ss: u32,
    lut: Lut,
    map: ColorMap,
    options: ImageOptions,
    path: PathBuf,
) -> Arc<JobStatus> {
    let status = JobStatus::new("Salvataggio immagine");
    let st = status.clone();
    std::thread::spawn(move || {
        st.set_message(format!("Calcolo di {w}×{h} pixel…"));
        let total_rows = (h * ss.max(1)) as f64;
        let watcher = {
            let st = st.clone();
            std::thread::spawn(move || {
                while !st.done.load(Ordering::Relaxed) {
                    let rows = st.rows_done.load(Ordering::Relaxed) as f64;
                    st.set_fraction(0.95 * rows / total_rows);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            })
        };
        let r = match render_rgb(&hub, choice, &params, w, h, ss, &lut, map, &st) {
            Ok(Some((rgb, used))) => {
                st.set_message("Scrittura del file…");
                export::save_rgb(&path, w, h, &rgb, &options)
                    .map(|_| format!("Immagine salvata in {} ({used})", path.display()))
            }
            Ok(None) => Err("Salvataggio annullato".into()),
            Err(e) => Err(e),
        };
        st.set_fraction(1.0);
        st.finish(r);
        let _ = watcher.join();
    });
    status
}

/// Export an animation in the background.
pub fn start_video_export(
    hub: Arc<ComputeHub>,
    choice: ComputeChoice,
    animation: Animation,
    settings: VideoSettings,
    ffmpeg: Option<PathBuf>,
    working_encoders: Vec<&'static str>,
    path: PathBuf,
) -> Arc<JobStatus> {
    let status = JobStatus::new("Esportazione animazione");
    let st = status.clone();
    std::thread::spawn(move || {
        let r = export_video(
            &hub,
            choice,
            &animation,
            &settings,
            ffmpeg,
            &working_encoders,
            &path,
            &st,
        );
        st.set_fraction(1.0);
        st.finish(r);
    });
    status
}

#[allow(clippy::too_many_arguments)]
fn export_video(
    hub: &ComputeHub,
    choice: ComputeChoice,
    animation: &Animation,
    s: &VideoSettings,
    ffmpeg: Option<PathBuf>,
    working: &[&'static str],
    path: &std::path::Path,
    status: &JobStatus,
) -> Result<String, String> {
    if animation.keyframes.is_empty() {
        return Err("Aggiungi almeno un keyframe".into());
    }
    let (w, h) = (s.width.max(16), s.height.max(16));
    if s.format.codec().is_some() && (w % 2 != 0 || h % 2 != 0) {
        return Err("Per H.264 e H.265 larghezza e altezza devono essere numeri pari".into());
    }
    let encoder = s
        .format
        .codec()
        .and_then(|c| video::pick_encoder(c, working, &s.encoder));
    if s.format.codec().is_some() && encoder.is_none() {
        return Err("Nessun codificatore video funzionante per questo formato: controlla ffmpeg nelle impostazioni".into());
    }
    let mut sink = Sink::open(s, path, ffmpeg.as_deref(), encoder)?;
    let frames = animation.frame_count(s.fps.max(1));
    let luts = animation.luts();
    let enc_label = encoder
        .map(|e| format!(" · codifica {}", e.label))
        .unwrap_or_default();
    let started = Instant::now();
    for f in 0..frames {
        if status.cancelled() {
            sink.abort();
            return Err("Esportazione annullata".into());
        }
        let t = f as f64 / s.fps.max(1) as f64;
        let spec = animation.sample(t).ok_or("animazione vuota")?;
        let lut = spec.lut(&luts);
        let params = FractalParams {
            view: spec.view,
            max_iter: spec.iterations,
        };
        let rendered = match render_rgb(
            hub,
            choice,
            &params,
            w,
            h,
            s.supersampling,
            &lut,
            spec.map,
            status,
        ) {
            Ok(Some(x)) => x,
            Ok(None) => {
                sink.abort();
                return Err("Esportazione annullata".into());
            }
            Err(e) => {
                sink.abort();
                return Err(e);
            }
        };
        if let Err(e) = sink.write(&rendered.0, w, h) {
            sink.abort();
            return Err(e);
        }
        let done = (f + 1) as f64;
        status.set_fraction(done / frames as f64);
        let per = started.elapsed().as_secs_f64() / done;
        let left = per * (frames as f64 - done);
        status.set_message(format!(
            "Fotogramma {} di {} · mancano circa {} · {}{}",
            f + 1,
            frames,
            format_duration(left),
            rendered.1,
            enc_label
        ));
    }
    status.set_message("Chiusura del file…");
    sink.finish()
}

pub fn format_duration(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s >= 3600 {
        format!("{}h {:02}m", s / 3600, (s % 3600) / 60)
    } else if s >= 60 {
        format!("{}m {:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

/// Check that a picture of `w`×`h` with supersampling `ss` fits in memory.
pub fn fits(w: u32, h: u32, ss: u32) -> bool {
    (w as u64 * ss as u64) * (h as u64 * ss as u64) <= MAX_SAMPLES
}
