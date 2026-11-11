//! The window: viewer, side panels, dialogs and keyboard/mouse controls.

mod panels;
mod viewer;
mod windows;

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, Key, Modifiers, Pos2, RichText, Vec2};

use crate::animation::{Animation, AnimationProject, Easing, Keyframe};
use crate::compute::gpu::AdapterEntry;
use crate::compute::{ComputeHub, Engine, FractalParams};
use crate::export::ImageFormat;
use crate::flight::Flight;
use crate::jobs::{self, JobStatus};
use crate::palette::{colorize, ColorMap, ColorSettings, Lut, Palette};
use crate::recommend::Recommendation;
use crate::settings::{self, AppSettings, Location, StartAt, Theme};
use crate::video::{VideoFormat, RESOLUTIONS};
use crate::view::View;
use crate::workers::{self, EncoderState, Recommender, ViewRequest, ViewWorker};

pub const MAX_ITERATIONS: u32 = 50_000_000;

struct Shown {
    request: ViewRequest,
    w: u32,
    h: u32,
    data: Vec<f32>,
    preview: bool,
    used: String,
    elapsed: Duration,
}

/// One calculated picture kept on screen, with the view it shows.
struct Layer {
    tex: egui::TextureHandle,
    view: View,
    aspect: f64,
    born: Instant,
}

impl Layer {
    /// Left, right, bottom, top in the complex plane.
    fn extent(&self) -> (f64, f64, f64, f64) {
        let v = self.view;
        let hw = v.scale * self.aspect;
        (v.cx - hw, v.cx + hw, v.cy - v.scale, v.cy + v.scale)
    }

    fn covers(&self, other: &Layer) -> bool {
        let (l, r, b, t) = self.extent();
        let (ol, or, ob, ot) = other.extent();
        l <= ol && r >= or && b <= ob && t >= ot
    }
}

/// Pictures kept under the newest one (like the map tiles of mandelbrot.site, which keep the
/// previous zoom levels visible while the new ones load).
const MAX_LAYERS: usize = 8;
/// How long a new picture takes to fade in.
pub const FADE: Duration = Duration::from_millis(200);

struct Toast {
    text: String,
    error: bool,
    at: Instant,
}

/// While the animation is being previewed: what to go back to.
struct PreviewState {
    view: View,
    iterations: u32,
}

pub struct MandelApp {
    hub: Arc<ComputeHub>,
    s: AppSettings,
    custom_palettes: Vec<Palette>,
    worker: ViewWorker,
    last_request: Option<ViewRequest>,
    shown: Option<Shown>,
    /// Pictures on screen, oldest first: the newest fades in over the older ones.
    layers: Vec<Layer>,
    recolor: bool,
    /// The worker delivered a new picture (not just new colours).
    new_picture: bool,
    lut: Lut,
    last_color: ColorSettings,
    view_px: (u32, u32),
    selection_start: Option<Pos2>,
    hover: Option<(f64, f64)>,
    edit_re: String,
    edit_im: String,
    edit_zoom: String,
    edits_for: Option<View>,
    editing_coords: bool,
    paste_text: String,
    fav_name: String,
    recommender: Recommender,
    recommendation: Option<(View, Recommendation)>,
    last_view: View,
    last_change: Instant,
    palette_name: String,
    animation: Animation,
    anim_time: f64,
    playing: bool,
    preview: Option<PreviewState>,
    preview_color: Option<(Lut, ColorMap)>,
    show_image_dialog: bool,
    image_size: usize,
    image_w: u32,
    image_h: u32,
    video_custom_res: bool,
    jobs: Vec<Arc<JobStatus>>,
    encoders: Arc<Mutex<EncoderState>>,
    adapters: Arc<Mutex<Vec<AdapterEntry>>>,
    toast: Option<Toast>,
    show_help: bool,
    show_about: bool,
    gpu_ready_seen: bool,
    /// The zoom-out/zoom-in transition in progress, and when it started.
    flight: Option<(Flight, Instant)>,
    /// The person dragged or zoomed with the mouse in this frame.
    moved: bool,
}

impl MandelApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = cc.egui_ctx.clone();
        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(6.0, 5.0);
        ctx.set_style(style);

        let mut s = AppSettings::load();
        if s.start_at == StartAt::DefaultPosition {
            s.view = View::default();
            s.iterations = AppSettings::default().iterations;
        }
        apply_theme(&ctx, s.theme);
        let hub = ComputeHub::new();
        {
            let c = ctx.clone();
            hub.start_gpu(s.gpu_adapter.clone(), move || c.request_repaint());
        }
        let adapters = Arc::new(Mutex::new(Vec::new()));
        {
            let adapters = adapters.clone();
            let c = ctx.clone();
            std::thread::spawn(move || {
                *adapters.lock().unwrap() = crate::compute::gpu::list_adapters();
                c.request_repaint();
            });
        }
        let encoders = Arc::new(Mutex::new(EncoderState::default()));
        workers::check_encoders(
            encoders.clone(),
            s.ffmpeg_path.clone(),
            s.video.clone(),
            ctx.clone(),
        );

        let lut = Lut::new(&s.color.palette);
        let last_color = s.color.clone();
        let show_help = s.show_help_on_start;
        let view = s.view;
        MandelApp {
            worker: ViewWorker::new(hub.clone(), ctx.clone()),
            recommender: Recommender::new(ctx.clone()),
            hub,
            custom_palettes: settings::load_custom_palettes(),
            last_request: None,
            shown: None,
            layers: Vec::new(),
            recolor: false,
            new_picture: false,
            lut,
            last_color,
            view_px: (800, 600),
            selection_start: None,
            hover: None,
            edit_re: String::new(),
            edit_im: String::new(),
            edit_zoom: String::new(),
            edits_for: None,
            editing_coords: false,
            paste_text: String::new(),
            fav_name: String::new(),
            recommendation: None,
            last_view: view,
            last_change: Instant::now(),
            palette_name: String::new(),
            animation: Animation::default(),
            anim_time: 0.0,
            playing: false,
            preview: None,
            preview_color: None,
            show_image_dialog: false,
            image_size: 0,
            image_w: 3840,
            image_h: 2160,
            video_custom_res: false,
            jobs: Vec::new(),
            encoders,
            adapters,
            toast: None,
            show_help,
            show_about: false,
            gpu_ready_seen: false,
            flight: None,
            moved: false,
            s,
        }
    }

    // ----------------------------------------------------------------------------------------
    // State helpers

    fn notify(&mut self, text: impl Into<String>, error: bool) {
        self.toast = Some(Toast {
            text: text.into(),
            error,
            at: Instant::now(),
        });
    }

    /// The person moved the view: leave the animation preview where it is and stop any
    /// transition where it has got to.
    fn navigated(&mut self) {
        if self.flight.take().is_some() {
            debug_log("transizione interrotta");
        }
        if self.preview.is_some() {
            self.preview = None;
            self.preview_color = None;
            self.playing = false;
            self.recolor = true;
        }
    }

    /// Jump to another place: with a zoom-out/zoom-in flight when that is switched on.
    fn go_to(&mut self, view: View, iterations: u32) {
        self.navigated();
        let iterations = iterations.clamp(1, MAX_ITERATIONS);
        if self.s.animate_moves && view != self.s.view {
            let aspect = self.view_px.0 as f64 / self.view_px.1.max(1) as f64;
            let f = Flight::new(self.s.view, view, self.s.iterations, iterations, aspect, self.s.flight_speed);
            debug_log(&format!("transizione iniziata: {:.1} s", f.duration));
            self.flight = Some((f, Instant::now()));
        } else {
            self.s.view = view;
            self.s.iterations = iterations;
        }
    }

    /// Move the flight on; called every frame.
    fn advance_flight(&mut self, ctx: &egui::Context) {
        let Some((f, started)) = &self.flight else { return };
        let t = started.elapsed().as_secs_f64() / f.duration;
        let (view, iterations) = f.at(t);
        self.s.view = view;
        self.s.iterations = iterations;
        if t >= 1.0 {
            debug_log("transizione finita");
            self.flight = None;
        } else {
            ctx.request_repaint();
        }
    }

    fn reset_view(&mut self) {
        let it = self.s.iterations.min(500);
        self.go_to(View::default(), it);
    }

    fn zoom_center(&mut self, factor: f64) {
        self.navigated();
        let (w, h) = (self.view_px.0 as f64, self.view_px.1 as f64);
        self.s.view.zoom_at(w * 0.5, h * 0.5, w, h, factor);
    }

    fn pan_fraction(&mut self, fx: f64, fy: f64) {
        self.navigated();
        let h = self.view_px.1 as f64;
        self.s.view.pan_pixels(-fx * h, -fy * h, h);
    }

    fn set_iterations(&mut self, n: u32) {
        self.s.iterations = n.clamp(1, MAX_ITERATIONS);
    }

    fn apply_location(&mut self, loc: &Location, with_colors: bool) {
        if let Some(v) = loc.view() {
            self.go_to(v, loc.iterations);
            if with_colors {
                if let Some(c) = &loc.color {
                    self.s.color = c.clone();
                }
            }
        }
    }

    fn active_colors(&self) -> (&Lut, ColorMap) {
        match &self.preview_color {
            Some((l, m)) => (l, *m),
            None => (&self.lut, ColorMap::from(&self.s.color)),
        }
    }

    fn compute_choice(&self) -> jobs::ComputeChoice {
        workers::choice(self.s.engine, self.s.precision)
    }

    fn engine_words(&self) -> &'static str {
        if self.s.engine == Engine::Gpu {
            "scheda grafica"
        } else {
            "processore"
        }
    }

    fn pick_folder(&self) -> Option<PathBuf> {
        self.s.last_folder.clone().filter(|p| p.is_dir())
    }

    fn remember_folder(&mut self, p: &Option<PathBuf>) {
        if let Some(parent) = p.as_ref().and_then(|p| p.parent()) {
            self.s.last_folder = Some(parent.to_path_buf());
        }
    }

    fn save_dialog(&mut self, title: &str, name: &str, filter: (&str, &[&str])) -> Option<PathBuf> {
        let mut d = rfd::FileDialog::new()
            .set_title(title)
            .set_file_name(name)
            .add_filter(filter.0, filter.1);
        if let Some(dir) = self.pick_folder() {
            d = d.set_directory(dir);
        }
        let p = d.save_file();
        self.remember_folder(&p);
        p
    }

    fn open_dialog(&mut self, title: &str, filter: (&str, &[&str])) -> Option<PathBuf> {
        let mut d = rfd::FileDialog::new()
            .set_title(title)
            .add_filter(filter.0, filter.1);
        if let Some(dir) = self.pick_folder() {
            d = d.set_directory(dir);
        }
        let p = d.pick_file();
        self.remember_folder(&p);
        p
    }

    fn save_coordinates(&mut self) {
        let name = format!("coordinate_zoom_{:.0e}.json", self.s.view.zoom());
        if let Some(path) = self.save_dialog("Salva coordinate", &name, ("Coordinate", &["json"])) {
            let path = path.with_extension("json");
            let loc = Location::new(
                "",
                &self.s.view,
                self.s.iterations,
                Some(self.s.color.clone()),
            );
            match loc.save(&path) {
                Ok(()) => self.notify(format!("Coordinate salvate in {}", path.display()), false),
                Err(e) => self.notify(e, true),
            }
        }
    }

    fn load_coordinates(&mut self) {
        if let Some(path) = self.open_dialog("Carica coordinate", ("Coordinate", &["json"])) {
            match Location::load(&path) {
                Ok(loc) => {
                    self.apply_location(&loc, true);
                    self.notify("Coordinate caricate", false);
                }
                Err(e) => self.notify(e, true),
            }
        }
    }

    fn copy_coordinates(&mut self, ctx: &egui::Context) {
        let loc = Location::new("", &self.s.view, self.s.iterations, None);
        ctx.copy_text(loc.to_clipboard_text());
        self.notify("Coordinate copiate negli appunti", false);
    }

    fn add_keyframe(&mut self) {
        self.navigated();
        let n = self.animation.keyframes.len();
        self.animation.keyframes.push(Keyframe {
            name: format!("Keyframe {}", n + 1),
            view: self.s.view,
            iterations: self.s.iterations,
            color: self.s.color.clone(),
            duration: 5.0,
            easing: Easing::Smooth,
        });
        self.notify(format!("Keyframe {} aggiunto", n + 1), false);
    }

    /// Show the animation at `t` seconds in the viewer.
    fn show_anim_time(&mut self, t: f64) {
        if self.animation.keyframes.is_empty() {
            return;
        }
        if self.preview.is_none() {
            self.preview = Some(PreviewState {
                view: self.s.view,
                iterations: self.s.iterations,
            });
        }
        self.anim_time = t.clamp(0.0, self.animation.total_duration());
        if let Some(spec) = self.animation.sample(self.anim_time) {
            self.s.view = spec.view;
            self.s.iterations = spec.iterations;
            let luts = self.animation.luts();
            self.preview_color = Some((spec.lut(&luts), spec.map));
            self.recolor = true;
        }
    }

    fn leave_preview(&mut self) {
        if let Some(p) = self.preview.take() {
            self.s.view = p.view;
            self.s.iterations = p.iterations;
        }
        self.preview_color = None;
        self.playing = false;
        self.recolor = true;
    }

    fn toggle_play(&mut self) {
        if self.animation.keyframes.len() < 2 {
            self.notify("Aggiungi almeno due keyframe per vedere l'animazione", true);
            return;
        }
        self.playing = !self.playing;
        if self.playing {
            if self.anim_time >= self.animation.total_duration() - 1e-6 {
                self.anim_time = 0.0;
            }
            let t = self.anim_time;
            self.show_anim_time(t);
        }
    }

    fn save_project(&mut self) {
        if let Some(path) = self.save_dialog(
            "Salva animazione",
            "animazione.mandelanim.json",
            ("Animazione", &["json"]),
        ) {
            let p = AnimationProject::new(self.animation.clone(), self.s.video.clone());
            let r = serde_json::to_string_pretty(&p)
                .map_err(|e| e.to_string())
                .and_then(|t| std::fs::write(&path, t).map_err(|e| e.to_string()));
            match r {
                Ok(()) => self.notify(format!("Animazione salvata in {}", path.display()), false),
                Err(e) => self.notify(format!("Impossibile salvare: {e}"), true),
            }
        }
    }

    fn load_project(&mut self) {
        if let Some(path) = self.open_dialog("Carica animazione", ("Animazione", &["json"])) {
            let r = std::fs::read_to_string(&path)
                .map_err(|e| e.to_string())
                .and_then(|t| {
                    serde_json::from_str::<AnimationProject>(&t).map_err(|e| e.to_string())
                });
            match r {
                Ok(p) => {
                    self.leave_preview();
                    self.animation = p.animation;
                    self.s.video = p.video;
                    self.anim_time = 0.0;
                    self.notify(
                        format!(
                            "Animazione caricata: {} keyframe",
                            self.animation.keyframes.len()
                        ),
                        false,
                    );
                }
                Err(e) => self.notify(format!("File di animazione non valido: {e}"), true),
            }
        }
    }

    fn image_dimensions(&self) -> (u32, u32) {
        match self.image_size {
            0 => self.view_px,
            i if i <= RESOLUTIONS.len() => (RESOLUTIONS[i - 1].1, RESOLUTIONS[i - 1].2),
            _ => (self.image_w.max(16), self.image_h.max(16)),
        }
    }

    fn start_image_save(&mut self) {
        let (w, h) = self.image_dimensions();
        let ss = self.s.image_supersampling.max(1);
        if !jobs::fits(w, h, ss) {
            self.notify("Immagine troppo grande con questo antialiasing: riduci la risoluzione o l'antialiasing", true);
            return;
        }
        let fmt = self.s.image.format;
        let name = format!("mandelbrot_{}x{}.{}", w, h, fmt.extension());
        let exts: &[&str] = match fmt {
            ImageFormat::Jpeg => &["jpg", "jpeg"],
            ImageFormat::Tiff => &["tif", "tiff"],
            ImageFormat::Png => &["png"],
        };
        if let Some(path) = self.save_dialog("Salva immagine", &name, (fmt.label(), exts)) {
            let path = crate::export::with_extension(&path, fmt);
            let (lut, map) = self.active_colors();
            let job = jobs::start_image_export(
                self.hub.clone(),
                self.compute_choice(),
                FractalParams {
                    view: self.s.view,
                    max_iter: self.s.iterations,
                },
                w,
                h,
                ss,
                lut.clone(),
                map,
                self.s.image,
                path,
            );
            self.jobs.push(job);
            self.show_image_dialog = false;
        }
    }

    fn start_video_export(&mut self) {
        if self.animation.keyframes.len() < 2 {
            self.notify("Servono almeno due keyframe per un'animazione", true);
            return;
        }
        let v = self.s.video.clone();
        if !jobs::fits(v.width, v.height, v.supersampling.max(1)) {
            self.notify("Risoluzione troppo grande con questo antialiasing", true);
            return;
        }
        let ext = v.format.extension();
        let name = match v.format {
            VideoFormat::TiffSequence => "fotogramma.tif".to_string(),
            _ => format!("mandelbrot_{}x{}.{}", v.width, v.height, ext),
        };
        if let Some(path) =
            self.save_dialog("Esporta animazione", &name, (v.format.label(), &[ext]))
        {
            let path = path.with_extension(ext);
            let st = self.encoders.lock().unwrap().clone();
            self.leave_preview();
            let job = jobs::start_video_export(
                self.hub.clone(),
                self.compute_choice(),
                self.animation.clone(),
                v,
                st.ffmpeg.clone(),
                st.working(),
                path,
            );
            self.jobs.push(job);
        }
    }

    fn restart_gpu(&mut self, ctx: &egui::Context) {
        let c = ctx.clone();
        self.hub
            .start_gpu(self.s.gpu_adapter.clone(), move || c.request_repaint());
        self.gpu_ready_seen = false;
        self.last_request = None;
    }

    // ----------------------------------------------------------------------------------------
    // Keyboard

    fn handle_keys(&mut self, ctx: &egui::Context) {
        let cmd = |key| ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, key));
        let cmd_shift =
            |key| ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND | Modifiers::SHIFT, key));
        if cmd_shift(Key::S) {
            self.save_coordinates();
        } else if cmd(Key::S) {
            self.show_image_dialog = true;
        }
        if cmd(Key::O) {
            self.load_coordinates();
        }
        if ctx.wants_keyboard_input() {
            return;
        }
        let (pressed, shift) = ctx.input(|i| {
            let keys = [
                Key::Plus,
                Key::Equals,
                Key::PageUp,
                Key::Minus,
                Key::PageDown,
                Key::ArrowLeft,
                Key::ArrowRight,
                Key::ArrowUp,
                Key::ArrowDown,
                Key::Home,
                Key::K,
                Key::I,
                Key::OpenBracket,
                Key::CloseBracket,
                Key::G,
                Key::Space,
                Key::F1,
                Key::F2,
                Key::F3,
                Key::Escape,
                Key::C,
            ];
            let pressed: Vec<Key> = keys
                .into_iter()
                .filter(|k| i.key_pressed(*k) && !i.modifiers.command)
                .collect();
            (pressed, i.modifiers.shift)
        });
        let zoom_step = if shift { 1.25 } else { 2.0 };
        let pan_step = if shift { 0.02 } else { 0.1 };
        for k in pressed {
            match k {
                Key::Plus | Key::Equals | Key::PageUp => self.zoom_center(zoom_step),
                Key::Minus | Key::PageDown => self.zoom_center(1.0 / zoom_step),
                Key::ArrowLeft => self.pan_fraction(-pan_step, 0.0),
                Key::ArrowRight => self.pan_fraction(pan_step, 0.0),
                Key::ArrowUp => self.pan_fraction(0.0, -pan_step),
                Key::ArrowDown => self.pan_fraction(0.0, pan_step),
                Key::Home => self.reset_view(),
                Key::K => self.add_keyframe(),
                Key::I => {
                    if let Some((_, r)) = self.recommendation.clone() {
                        self.set_iterations(r.iterations);
                    }
                }
                Key::OpenBracket => {
                    self.set_iterations((self.s.iterations as f64 / 1.25).round() as u32)
                }
                Key::CloseBracket => {
                    self.set_iterations((self.s.iterations as f64 * 1.25).round() as u32)
                }
                Key::G => {
                    self.s.engine = if self.s.engine == Engine::Gpu {
                        Engine::Cpu
                    } else {
                        Engine::Gpu
                    };
                    self.notify(format!("Calcolo con: {}", self.s.engine.label()), false);
                }
                Key::Space => self.toggle_play(),
                Key::F1 => self.show_help = !self.show_help,
                Key::F2 => self.s.show_left_panel = !self.s.show_left_panel,
                Key::F3 => self.s.show_right_panel = !self.s.show_right_panel,
                Key::C => self.copy_coordinates(ctx),
                Key::Escape => {
                    self.selection_start = None;
                    self.flight = None;
                    if self.preview.is_some() {
                        self.leave_preview();
                    }
                    self.show_image_dialog = false;
                    self.show_help = false;
                }
                _ => {}
            }
        }
    }

    fn menu_bar(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("File", |ui| {
                if ui.button("Salva immagine…   Ctrl+S").clicked() {
                    self.show_image_dialog = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Salva coordinate…   Ctrl+Maiusc+S").clicked() {
                    ui.close_menu();
                    self.save_coordinates();
                }
                if ui.button("Carica coordinate…   Ctrl+O").clicked() {
                    ui.close_menu();
                    self.load_coordinates();
                }
                ui.separator();
                if ui.button("Salva animazione…").clicked() {
                    ui.close_menu();
                    self.save_project();
                }
                if ui.button("Carica animazione…").clicked() {
                    ui.close_menu();
                    self.load_project();
                }
                ui.separator();
                if ui.button("Esci").clicked() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Vista", |ui| {
                if ui.button("Vista iniziale   Home").clicked() {
                    self.reset_view();
                    ui.close_menu();
                }
                if ui.button("Zoom avanti   +").clicked() {
                    self.zoom_center(2.0);
                    ui.close_menu();
                }
                if ui.button("Zoom indietro   −").clicked() {
                    self.zoom_center(0.5);
                    ui.close_menu();
                }
                ui.separator();
                ui.checkbox(&mut self.s.show_left_panel, "Pannello dei comandi   F2");
                ui.checkbox(&mut self.s.show_right_panel, "Pannello dell'animazione   F3");
                ui.checkbox(&mut self.s.animate_moves, "Transizione animata tra posizioni");
            });
            ui.menu_button("Impostazioni", |ui| {
                ui.label(RichText::new("Aspetto").strong());
                let before = self.s.theme;
                for t in [Theme::System, Theme::Light, Theme::Night] {
                    ui.radio_value(&mut self.s.theme, t, t.label());
                }
                if self.s.theme != before {
                    apply_theme(ctx, self.s.theme);
                }
                ui.separator();
                ui.label(RichText::new("All'apertura del programma").strong());
                for a in [StartAt::LastPosition, StartAt::DefaultPosition] {
                    ui.radio_value(&mut self.s.start_at, a, a.label());
                }
                ui.separator();
                ui.checkbox(&mut self.s.animate_moves, "Transizione animata tra posizioni");
                ui.add_enabled(
                    self.s.animate_moves,
                    egui::Slider::new(&mut self.s.flight_speed, 0.25..=4.0)
                        .logarithmic(true)
                        .fixed_decimals(2)
                        .text("velocità"),
                );
            });
            ui.menu_button("Aiuto", |ui| {
                if ui.button("Comandi mouse e tastiera   F1").clicked() {
                    self.show_help = true;
                    ui.close_menu();
                }
                if ui.button("Informazioni").clicked() {
                    self.show_about = true;
                    ui.close_menu();
                }
            });
            ui.separator();
            if ui.button("📷 Salva immagine").clicked() {
                self.show_image_dialog = true;
            }
            if ui.button("➕ Keyframe").clicked() {
                self.add_keyframe();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let night = ui.visuals().dark_mode;
                let (icon, tip) = if night {
                    ("☀", "Modalità giorno")
                } else {
                    ("🌙", "Modalità notte")
                };
                if ui.button(icon).on_hover_text(tip).clicked() {
                    self.s.theme = if night { Theme::Light } else { Theme::Night };
                    apply_theme(ctx, self.s.theme);
                }
            });
        });
    }

    /// Take finished background results.
    fn collect_results(&mut self) {
        if let Some(r) = self.worker.poll() {
            match r.error {
                Some(e) => self.notify(e, true),
                None => {
                    debug_log(&format!(
                        "immagine {}x{} scala {:.3e} iterazioni {} in {} ms{}",
                        r.w,
                        r.h,
                        r.request.params.view.scale,
                        r.request.params.max_iter,
                        r.elapsed.as_millis(),
                        if self.flight.is_some() { " (durante la transizione)" } else { "" }
                    ));
                    self.shown = Some(Shown {
                        request: r.request,
                        w: r.w,
                        h: r.h,
                        data: r.data,
                        preview: r.preview,
                        used: r.used,
                        elapsed: r.elapsed,
                    });
                    self.recolor = true;
                    self.new_picture = true;
                }
            }
        }
        if let Some((v, rec)) = self.recommender.poll() {
            if self.s.auto_iterations && v == self.s.view && self.preview.is_none() {
                self.set_iterations(rec.iterations);
            }
            self.recommendation = Some((v, rec));
        }
        let gpu_ready = self.hub.gpu_context().is_some();
        if gpu_ready != self.gpu_ready_seen {
            self.gpu_ready_seen = gpu_ready;
            self.last_request = None;
        }
        if self
            .toast
            .as_ref()
            .is_some_and(|t| t.at.elapsed() > Duration::from_secs(10))
        {
            self.toast = None;
        }
    }

    fn update_texture(&mut self, ctx: &egui::Context) {
        if self.s.color != self.last_color {
            self.lut = Lut::new(&self.s.color.palette);
            self.last_color = self.s.color.clone();
            self.recolor = true;
        }
        if !self.recolor {
            return;
        }
        self.recolor = false;
        let new_picture = std::mem::take(&mut self.new_picture);
        let Some(sh) = &self.shown else { return };
        let (lut, map) = self.active_colors();
        let mut rgba = vec![0u8; sh.data.len() * 4];
        colorize(&sh.data, lut, map, 4, &mut rgba);
        let img = egui::ColorImage::from_rgba_unmultiplied([sh.w as usize, sh.h as usize], &rgba);
        let view = sh.request.params.view;
        let aspect = sh.request.w as f64 / sh.request.h.max(1) as f64;
        match self.layers.last_mut() {
            // Only the colours changed: repaint the newest picture and forget older ones,
            // which still have the old colours.
            Some(top) if !new_picture => {
                top.tex.set(img, egui::TextureOptions::LINEAR);
                let keep = self.layers.len() - 1;
                self.layers.drain(..keep);
            }
            _ => {
                // A new picture goes on top of the older ones, which stay visible underneath
                // (scaled to where they belong) while it fades in.
                let tex = ctx.load_texture("frattale", img, egui::TextureOptions::LINEAR);
                let new = Layer { tex, view, aspect, born: Instant::now() };
                // Older pictures entirely inside the new one can never show again.
                self.layers.retain(|old| !new.covers(old));
                self.layers.push(new);
                if self.layers.len() > MAX_LAYERS {
                    self.layers.remove(0);
                }
            }
        }
    }

    /// Ask for new work when the view or the settings changed.
    fn schedule_work(&mut self, ctx: &egui::Context) {
        // While things move: half resolution, and each picture is allowed to finish, so the
        // view keeps updating during the movement.
        let moving = self.playing || self.flight.is_some() || self.moved;
        self.moved = false;
        let div = if moving { 2 } else { 1 };
        // While moving, a modest number of iterations keeps each picture to a few hundredths
        // of a second, so the movement is seen; the full count comes back when it stops.
        let max_iter = if moving {
            self.s
                .iterations
                .min((crate::recommend::heuristic(&self.s.view) * 4).max(2000))
        } else {
            self.s.iterations
        };
        let req = ViewRequest {
            params: FractalParams { view: self.s.view, max_iter },
            w: (self.view_px.0 / div).max(16),
            h: (self.view_px.1 / div).max(16),
            engine: self.s.engine,
            precision: self.s.precision,
        };
        if self.last_request != Some(req) {
            self.worker.request(req, moving);
            self.last_request = Some(req);
        }
        if moving {
            // One more frame after the movement ends sends the full-quality request.
            ctx.request_repaint();
        }

        // Suggest iterations once the view has stopped moving.
        if self.s.view != self.last_view {
            self.last_view = self.s.view;
            self.last_change = Instant::now();
        }
        let pointer_down = ctx.input(|i| i.pointer.any_down());
        if self.preview.is_none()
            && self.recommender.asked_for != Some(self.s.view)
            && !pointer_down
        {
            let wait = Duration::from_millis(250);
            if self.last_change.elapsed() >= wait {
                let aspect = self.view_px.0 as f64 / self.view_px.1.max(1) as f64;
                self.recommender.request(self.s.view, aspect);
            } else {
                ctx.request_repaint_after(wait);
            }
        }
        if self.worker.busy.load(Ordering::Relaxed) || self.recommender.busy.load(Ordering::Relaxed)
        {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }
}

impl eframe::App for MandelApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.collect_results();

        if self.playing {
            let dt = ctx.input(|i| i.stable_dt).min(0.1) as f64;
            let total = self.animation.total_duration();
            let mut t = self.anim_time + dt;
            if t >= total {
                t = total;
                self.playing = false;
            }
            self.show_anim_time(t);
            ctx.request_repaint();
        }

        self.handle_keys(ctx);
        self.advance_flight(ctx);

        egui::TopBottomPanel::top("menu").show(ctx, |ui| self.menu_bar(ctx, ui));
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| self.status_bar(ui));

        // Side panels fold down to a thin strip with a button to open them again.
        let strip = |id: &str| egui::SidePanel::left(id.to_owned()).resizable(false).exact_width(30.0);
        egui::SidePanel::show_animated_between(
            ctx,
            self.s.show_left_panel,
            strip("controls-folded"),
            egui::SidePanel::left("controls").resizable(true).default_width(330.0).min_width(260.0),
            |ui, how_open| {
                if how_open == 0.0 {
                    if ui.button("»").on_hover_text("Apri il pannello dei comandi (F2)").clicked() {
                        self.s.show_left_panel = true;
                    }
                } else {
                    self.left_panel(ui);
                }
            },
        );
        let strip = |id: &str| egui::SidePanel::right(id.to_owned()).resizable(false).exact_width(30.0);
        egui::SidePanel::show_animated_between(
            ctx,
            self.s.show_right_panel,
            strip("animation-folded"),
            egui::SidePanel::right("animation").resizable(true).default_width(320.0).min_width(260.0),
            |ui, how_open| {
                if how_open == 0.0 {
                    if ui.button("«").on_hover_text("Apri il pannello dell'animazione (F3)").clicked() {
                        self.s.show_right_panel = true;
                    }
                } else {
                    self.right_panel(ui);
                }
            },
        );
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ctx, |ui| self.viewer(ui));

        if self.show_image_dialog {
            self.image_dialog(ctx);
        }
        self.jobs_windows(ctx);
        self.help_windows(ctx);

        self.update_texture(ctx);
        self.schedule_work(ctx);
    }

    fn on_exit(&mut self) {
        if let Some(p) = self.preview.take() {
            self.s.view = p.view;
            self.s.iterations = p.iterations;
        }
        if let Err(e) = self.s.save() {
            eprintln!("Impossibile salvare le impostazioni: {e}");
        }
    }
}

/// Light, night (dark) or the system's choice for the whole window.
fn apply_theme(ctx: &egui::Context, theme: Theme) {
    ctx.set_theme(match theme {
        Theme::System => egui::ThemePreference::System,
        Theme::Light => egui::ThemePreference::Light,
        Theme::Night => egui::ThemePreference::Dark,
    });
}

/// When the environment variable MANDELBROT_LOG names a file, add a timed line to it (to
/// check what the window does, e.g. during a transition).
fn debug_log(line: &str) {
    use std::io::Write;
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    let Some(path) = std::env::var_os("MANDELBROT_LOG") else { return };
    let t = START.get_or_init(Instant::now).elapsed().as_secs_f64();
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{t:8.3} s  {line}");
    }
}

/// 1234567 → "1.234.567".
pub fn group_digits(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push('.');
        }
        out.push(c);
    }
    out
}
