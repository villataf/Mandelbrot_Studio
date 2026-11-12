//! Background helpers for the window: drawing the current view, suggesting iterations and
//! finding which video encoders work.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::compute::{ComputeHub, FractalParams};
use crate::jobs::ComputeChoice;
use crate::recommend::{self, Recommendation};
use crate::video::{self, EncoderProbe, VideoSettings};
use crate::view::View;

/// A request to draw the view.
#[derive(Clone, Copy, PartialEq)]
pub struct ViewRequest {
    pub params: FractalParams,
    pub w: u32,
    pub h: u32,
    pub engine: crate::compute::Engine,
    pub precision: crate::compute::gpu::PrecisionChoice,
}

pub struct ViewResult {
    pub gen: u64,
    pub request: ViewRequest,
    /// Size actually calculated (smaller for the quick first pass).
    pub w: u32,
    pub h: u32,
    pub data: Vec<f32>,
    pub preview: bool,
    pub used: String,
    pub elapsed: Duration,
    pub error: Option<String>,
}

/// Draws the view on a background thread.
///
/// A still view is drawn in two passes (a quick rough one, then the full one) and a newer
/// request cancels it at once. While the view is moving (a transition, the animation
/// preview, dragging or zooming) every request is a single quick pass that is allowed to
/// finish; the worker then jumps to the newest position. Otherwise each frame of the
/// movement would cancel the previous one and nothing new would appear until it stopped.
pub struct ViewWorker {
    tx: Sender<(u64, ViewRequest, bool)>,
    rx: Receiver<ViewResult>,
    /// Running work with a smaller number than this must stop.
    abort_below: Arc<AtomicU64>,
    pub rows_done: Arc<AtomicU32>,
    pub busy: Arc<AtomicBool>,
    gen: u64,
    /// Number of the last result handed to the window.
    shown_gen: u64,
}

impl ViewWorker {
    pub fn new(hub: Arc<ComputeHub>, ctx: eframe::egui::Context) -> ViewWorker {
        let (tx, job_rx) = channel::<(u64, ViewRequest, bool)>();
        let (res_tx, rx) = channel::<ViewResult>();
        let abort_below = Arc::new(AtomicU64::new(0));
        let rows_done = Arc::new(AtomicU32::new(0));
        let busy = Arc::new(AtomicBool::new(false));
        {
            let abort_below = abort_below.clone();
            let rows_done = rows_done.clone();
            let busy = busy.clone();
            std::thread::Builder::new()
                .name("view-render".into())
                .spawn(move || {
                    let mut was_moving = false;
                    while let Ok(mut job) = job_rx.recv() {
                        while let Ok(newer) = job_rx.try_recv() {
                            job = newer;
                        }
                        let (gen, req, moving) = job;
                        if abort_below.load(Ordering::SeqCst) > gen {
                            continue;
                        }
                        busy.store(true, Ordering::SeqCst);
                        let cancel = || abort_below.load(Ordering::SeqCst) > gen;
                        let cost = req.w as f64 * req.h as f64 * req.params.max_iter as f64;
                        let mut passes = vec![];
                        // Right after a movement the half-resolution picture is already on
                        // screen: a coarser rough pass would only blur it for a moment.
                        if !moving && !was_moving && cost > 4.0e7 && req.w >= 64 && req.h >= 64 {
                            passes.push(4u32);
                        }
                        was_moving = moving;
                        passes.push(1u32);
                        for div in passes {
                            let (w, h) = ((req.w / div).max(1), (req.h / div).max(1));
                            rows_done.store(0, Ordering::Relaxed);
                            let started = Instant::now();
                            let r = hub.render(
                                &req.params,
                                w,
                                h,
                                req.engine,
                                req.precision,
                                &cancel,
                                Some(&rows_done),
                            );
                            let result = match r {
                                Ok(Some(c)) => ViewResult {
                                    gen,
                                    request: req,
                                    w,
                                    h,
                                    data: c.data,
                                    preview: div > 1,
                                    used: c.used,
                                    elapsed: started.elapsed(),
                                    error: None,
                                },
                                Ok(None) => break,
                                Err(e) => ViewResult {
                                    gen,
                                    request: req,
                                    w,
                                    h,
                                    data: vec![],
                                    preview: div > 1,
                                    used: String::new(),
                                    elapsed: started.elapsed(),
                                    error: Some(e),
                                },
                            };
                            let failed = result.error.is_some();
                            if res_tx.send(result).is_err() {
                                return;
                            }
                            ctx.request_repaint();
                            if failed {
                                break;
                            }
                        }
                        busy.store(false, Ordering::SeqCst);
                        ctx.request_repaint();
                    }
                })
                .expect("thread");
        }
        ViewWorker {
            tx,
            rx,
            abort_below,
            rows_done,
            busy,
            gen: 0,
            shown_gen: 0,
        }
    }

    /// Ask for a picture. `moving`: the view is changing frame by frame, so let the work
    /// in progress finish instead of cancelling it.
    pub fn request(&mut self, req: ViewRequest, moving: bool) {
        self.gen += 1;
        if !moving {
            self.abort_below.store(self.gen, Ordering::SeqCst);
        }
        let _ = self.tx.send((self.gen, req, moving));
    }

    /// The newest finished result not shown yet, if any.
    pub fn poll(&mut self) -> Option<ViewResult> {
        let mut last: Option<ViewResult> = None;
        while let Ok(r) = self.rx.try_recv() {
            let newest = last.as_ref().map_or(self.shown_gen, |l| l.gen);
            // A rough pass never replaces a finished picture of the same request.
            if r.gen > newest || (r.gen == newest && !r.preview) {
                last = Some(r);
            }
        }
        if let Some(r) = &last {
            self.shown_gen = r.gen;
        }
        last
    }
}

/// Suggests iterations for the view in the background.
pub struct Recommender {
    tx: Sender<(u64, View, f64)>,
    rx: Receiver<(u64, View, Recommendation)>,
    latest: Arc<AtomicU64>,
    gen: u64,
    pub asked_for: Option<View>,
    pub busy: Arc<AtomicBool>,
}

impl Recommender {
    pub fn new(ctx: eframe::egui::Context) -> Recommender {
        let (tx, job_rx) = channel::<(u64, View, f64)>();
        let (res_tx, rx) = channel();
        let latest = Arc::new(AtomicU64::new(0));
        let busy = Arc::new(AtomicBool::new(false));
        {
            let latest = latest.clone();
            let busy = busy.clone();
            std::thread::Builder::new()
                .name("recommend".into())
                .spawn(move || {
                    while let Ok(mut job) = job_rx.recv() {
                        while let Ok(newer) = job_rx.try_recv() {
                            job = newer;
                        }
                        let (gen, view, aspect) = job;
                        busy.store(true, Ordering::SeqCst);
                        let cancel = || latest.load(Ordering::SeqCst) != gen;
                        if let Some(r) = recommend::analyse(&view, aspect, &cancel) {
                            let _ = res_tx.send((gen, view, r));
                            ctx.request_repaint();
                        }
                        busy.store(false, Ordering::SeqCst);
                    }
                })
                .expect("thread");
        }
        Recommender {
            tx,
            rx,
            latest,
            gen: 0,
            asked_for: None,
            busy,
        }
    }

    pub fn request(&mut self, view: View, aspect: f64) {
        self.gen += 1;
        self.latest.store(self.gen, Ordering::SeqCst);
        self.asked_for = Some(view);
        let _ = self.tx.send((self.gen, view, aspect));
    }

    pub fn poll(&self) -> Option<(View, Recommendation)> {
        let mut last = None;
        while let Ok((gen, v, r)) = self.rx.try_recv() {
            if gen == self.gen {
                last = Some((v, r));
            }
        }
        last
    }
}

/// What is known about ffmpeg and its encoders.
#[derive(Clone, Default)]
pub struct EncoderState {
    pub checking: bool,
    pub ffmpeg: Option<PathBuf>,
    pub probes: Vec<EncoderProbe>,
}

impl EncoderState {
    pub fn working(&self) -> Vec<&'static str> {
        self.probes
            .iter()
            .filter(|p| p.works)
            .map(|p| p.name)
            .collect()
    }
}

/// Look for ffmpeg and test its encoders in the background.
pub fn check_encoders(
    state: Arc<Mutex<EncoderState>>,
    custom_path: String,
    settings: VideoSettings,
    ctx: eframe::egui::Context,
) {
    state.lock().unwrap().checking = true;
    std::thread::spawn(move || {
        let ffmpeg = video::find_ffmpeg(&custom_path);
        let probes = ffmpeg
            .as_deref()
            .map(|f| video::probe_encoders(f, &settings))
            .unwrap_or_default();
        *state.lock().unwrap() = EncoderState {
            checking: false,
            ffmpeg,
            probes,
        };
        ctx.request_repaint();
    });
}

/// Compute choice helper.
pub fn choice(
    engine: crate::compute::Engine,
    precision: crate::compute::gpu::PrecisionChoice,
) -> ComputeChoice {
    ComputeChoice { engine, precision }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::gpu::PrecisionChoice;
    use crate::compute::Engine;

    fn req(scale: f64) -> ViewRequest {
        ViewRequest {
            params: FractalParams { view: View { cx: -0.743_643_887, cy: 0.131_825_904, scale }, max_iter: 3000 },
            w: 480,
            h: 270,
            engine: Engine::Cpu,
            precision: PrecisionChoice::Auto,
        }
    }

    /// A transition sends a new position every frame: pictures must keep arriving while it
    /// runs, and the still view at the end must arrive in full quality.
    #[test]
    fn pictures_arrive_during_movement() {
        let mut worker = ViewWorker::new(ComputeHub::new(), eframe::egui::Context::default());
        let mut during = 0;
        let mut scale = 1.0;
        for _ in 0..60 {
            scale *= 0.85;
            worker.request(req(scale), true);
            std::thread::sleep(Duration::from_millis(16));
            if worker.poll().is_some() {
                during += 1;
            }
        }
        assert!(during >= 3, "only {during} pictures during the movement");

        worker.request(req(scale), false);
        let started = Instant::now();
        let mut finished = false;
        while started.elapsed() < Duration::from_secs(30) && !finished {
            if let Some(r) = worker.poll() {
                finished = !r.preview && r.request == req(scale) && r.w == 480;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(finished, "the still view was never drawn in full");
    }
}
