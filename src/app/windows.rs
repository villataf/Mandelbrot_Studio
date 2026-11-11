//! The animation panel and the pop-up windows (save picture, progress, help).

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use eframe::egui::{self, Color32, RichText, Vec2};

use super::panels::gradient_bar;
use super::{group_digits, MandelApp};
use crate::animation::Easing;
use crate::export::{ImageFormat, TiffCompression};
use crate::jobs;
use crate::settings;
use crate::video::{self, RateMode, Speed, VideoFormat, ENCODERS, RESOLUTIONS};
use crate::view::format_zoom;
use crate::workers;

const WARN: Color32 = Color32::from_rgb(255, 200, 80);
const ERROR: Color32 = Color32::from_rgb(255, 140, 120);

enum KfAction {
    Go(usize),
    Update(usize),
    Up(usize),
    Down(usize),
    Delete(usize),
}

fn ss_label(k: u32) -> String {
    match k {
        0 | 1 => "Nessuno".into(),
        k => format!("{k}×{k} campioni per pixel"),
    }
}

fn human_bytes(b: f64) -> String {
    if b >= 1e9 {
        format!("{:.1} GB", b / 1e9)
    } else {
        format!("{:.0} MB", b / 1e6)
    }
}

impl MandelApp {
    pub(super) fn right_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("»").on_hover_text("Riduci il pannello (F3)").clicked() {
                self.s.show_right_panel = false;
            }
            ui.heading("🎞 Animazione");
        });
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.keyframes_ui(ui);
                ui.separator();
                egui::CollapsingHeader::new(RichText::new("🎬 Esporta video").strong())
                    .default_open(true)
                    .show(ui, |ui| self.video_settings(ui));
            });
    }

    fn keyframes_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui
                .button("➕ Aggiungi keyframe")
                .on_hover_text("Salva vista, iterazioni e colori attuali come keyframe (tasto K)")
                .clicked()
            {
                self.add_keyframe();
            }
            if ui
                .button("💾")
                .on_hover_text("Salva l'animazione su file")
                .clicked()
            {
                self.save_project();
            }
            if ui
                .button("📂")
                .on_hover_text("Carica un'animazione da file")
                .clicked()
            {
                self.load_project();
            }
        });
        if self.animation.keyframes.is_empty() {
            ui.label(
                RichText::new(
                    "Nessun keyframe. Scegli un punto, i colori e le iterazioni e premi \"Aggiungi keyframe\"; \
                     poi spostati altrove e aggiungine un altro. L'animazione passa dall'uno all'altro.",
                )
                .small()
                .weak(),
            );
        }

        let mut action = None;
        let n = self.animation.keyframes.len();
        let times = self.animation.keyframe_times();
        for (i, kf) in self.animation.keyframes.iter_mut().enumerate() {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("#{}", i + 1)).strong());
                    ui.add(egui::TextEdit::singleline(&mut kf.name).desired_width(120.0));
                    ui.label(RichText::new(format!("a {:.1} s", times[i])).small().weak());
                });
                ui.label(
                    RichText::new(format!(
                        "Zoom {} · {} iterazioni · {}",
                        format_zoom(kf.view.zoom()),
                        group_digits(kf.iterations),
                        kf.color.palette.name
                    ))
                    .small(),
                );
                gradient_bar(ui, &kf.color.palette, 8.0);
                ui.horizontal(|ui| {
                    if ui
                        .small_button("👁 Vai")
                        .on_hover_text("Mostra questo keyframe")
                        .clicked()
                    {
                        action = Some(KfAction::Go(i));
                    }
                    if ui
                        .small_button("⟳ Aggiorna")
                        .on_hover_text("Sostituisci con vista, iterazioni e colori attuali")
                        .clicked()
                    {
                        action = Some(KfAction::Update(i));
                    }
                    if ui
                        .add_enabled(i > 0, egui::Button::new("▲").small())
                        .clicked()
                    {
                        action = Some(KfAction::Up(i));
                    }
                    if ui
                        .add_enabled(i + 1 < n, egui::Button::new("▼").small())
                        .clicked()
                    {
                        action = Some(KfAction::Down(i));
                    }
                    if ui
                        .small_button("🗑")
                        .on_hover_text("Elimina keyframe")
                        .clicked()
                    {
                        action = Some(KfAction::Delete(i));
                    }
                });
                if i + 1 < n {
                    ui.horizontal(|ui| {
                        ui.label("→ al successivo in");
                        ui.add(
                            egui::DragValue::new(&mut kf.duration)
                                .range(0.1..=3600.0)
                                .speed(0.1)
                                .suffix(" s"),
                        );
                        egui::ComboBox::from_id_salt(("easing", i))
                            .selected_text(kf.easing.label())
                            .width(80.0)
                            .show_ui(ui, |ui| {
                                for e in Easing::ALL {
                                    ui.selectable_value(&mut kf.easing, e, e.label());
                                }
                            });
                    });
                }
            });
        }
        match action {
            Some(KfAction::Go(i)) => {
                self.leave_preview();
                let kf = self.animation.keyframes[i].clone();
                self.go_to(kf.view, kf.iterations);
                self.s.color = kf.color;
                self.anim_time = times[i];
            }
            Some(KfAction::Update(i)) => {
                self.leave_preview();
                let kf = &mut self.animation.keyframes[i];
                kf.view = self.s.view;
                kf.iterations = self.s.iterations;
                kf.color = self.s.color.clone();
                self.notify(format!("Keyframe {} aggiornato", i + 1), false);
            }
            Some(KfAction::Up(i)) => self.animation.keyframes.swap(i, i - 1),
            Some(KfAction::Down(i)) => self.animation.keyframes.swap(i, i + 1),
            Some(KfAction::Delete(i)) => {
                self.animation.keyframes.remove(i);
                if self.animation.keyframes.len() < 2 {
                    self.leave_preview();
                }
            }
            None => {}
        }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label("Ciclo dei colori:");
            ui.add(
                egui::DragValue::new(&mut self.animation.color_cycle_speed)
                    .range(-10.0..=10.0)
                    .speed(0.01)
                    .suffix(" giri/s"),
            )
            .on_hover_text("Fa scorrere la palette durante l'animazione (0 = ferma)");
        });
        ui.label(
            RichText::new("Ogni keyframe ha la sua palette: tra due keyframe i colori sfumano dall'una all'altra.")
                .small()
                .weak(),
        );

        ui.separator();
        let total = self.animation.total_duration();
        let fps = self.s.video.fps.max(1);
        ui.label(format!(
            "Durata {} · {} fotogrammi a {fps} fps",
            jobs::format_duration(total),
            if n >= 2 {
                self.animation.frame_count(fps)
            } else {
                0
            }
        ));
        ui.add_enabled_ui(n >= 2, |ui| {
            let mut t = self.anim_time;
            if ui
                .add(
                    egui::Slider::new(&mut t, 0.0..=total.max(0.001))
                        .text("s")
                        .fixed_decimals(2),
                )
                .changed()
            {
                self.playing = false;
                self.show_anim_time(t);
            }
            ui.horizontal(|ui| {
                if ui.button("⏮").on_hover_text("Torna all'inizio").clicked() {
                    self.playing = false;
                    self.show_anim_time(0.0);
                }
                let label = if self.playing {
                    "⏸ Pausa"
                } else {
                    "▶ Anteprima"
                };
                if ui
                    .button(label)
                    .on_hover_text("Barra spaziatrice")
                    .clicked()
                {
                    self.toggle_play();
                }
                if self.preview.is_some()
                    && ui
                        .button("✖ Chiudi anteprima")
                        .on_hover_text("Torna alla vista di prima (Esc)")
                        .clicked()
                {
                    self.leave_preview();
                }
            });
        });
    }

    fn video_settings(&mut self, ui: &mut egui::Ui) {
        let st = self.encoders.lock().unwrap().clone();
        let working = st.working();
        let frames = if self.animation.keyframes.len() >= 2 {
            self.animation.frame_count(self.s.video.fps.max(1))
        } else {
            0
        };
        let v = &mut self.s.video;

        egui::ComboBox::from_label("Formato")
            .selected_text(v.format.label())
            .show_ui(ui, |ui| {
                for f in VideoFormat::ALL {
                    ui.selectable_value(&mut v.format, f, f.label());
                }
            });

        let res_label = RESOLUTIONS
            .iter()
            .find(|r| r.1 == v.width && r.2 == v.height && !self.video_custom_res)
            .map(|r| r.0.to_string())
            .unwrap_or_else(|| format!("Personalizzata ({}×{})", v.width, v.height));
        egui::ComboBox::from_label("Risoluzione")
            .selected_text(res_label)
            .show_ui(ui, |ui| {
                for r in RESOLUTIONS {
                    if ui
                        .selectable_label(
                            !self.video_custom_res && v.width == r.1 && v.height == r.2,
                            r.0,
                        )
                        .clicked()
                    {
                        v.width = r.1;
                        v.height = r.2;
                        self.video_custom_res = false;
                    }
                }
                if ui
                    .selectable_label(self.video_custom_res, "Personalizzata…")
                    .clicked()
                {
                    self.video_custom_res = true;
                }
            });
        if self.video_custom_res {
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut v.width)
                        .range(16..=16384)
                        .speed(2.0)
                        .suffix(" px"),
                );
                ui.label("×");
                ui.add(
                    egui::DragValue::new(&mut v.height)
                        .range(16..=16384)
                        .speed(2.0)
                        .suffix(" px"),
                );
            });
        }
        if v.format.codec().is_some() {
            v.width &= !1;
            v.height &= !1;
        }
        ui.horizontal(|ui| {
            ui.label("Fotogrammi al secondo:");
            for f in [24u32, 25, 30, 50, 60] {
                ui.selectable_value(&mut v.fps, f, f.to_string());
            }
        });
        egui::ComboBox::from_label("Antialiasing")
            .selected_text(ss_label(v.supersampling))
            .show_ui(ui, |ui| {
                for k in 1..=3 {
                    ui.selectable_value(&mut v.supersampling, k, ss_label(k));
                }
            });

        match v.format {
            VideoFormat::H264 | VideoFormat::H265 => {
                let codec = v.format.codec().unwrap_or(video::Codec::H264);
                let auto_label = match video::pick_encoder(codec, &working, "") {
                    Some(e) => format!("Automatico: {}", e.label),
                    None => "Automatico".to_string(),
                };
                let selected = video::spec(&v.encoder)
                    .filter(|e| e.codec == codec)
                    .map(|e| e.label.to_string())
                    .unwrap_or(auto_label.clone());
                egui::ComboBox::from_label("Codificatore")
                    .selected_text(selected)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut v.encoder, String::new(), auto_label);
                        for e in ENCODERS.iter().filter(|e| e.codec == codec) {
                            let ok = working.contains(&e.name);
                            let reason = st
                                .probes
                                .iter()
                                .find(|p| p.name == e.name)
                                .map(|p| p.message.clone())
                                .unwrap_or_else(|| "non disponibile su questo sistema".into());
                            ui.add_enabled_ui(ok, |ui| {
                                ui.selectable_value(
                                    &mut v.encoder,
                                    e.name.to_string(),
                                    format!("{} ({})", e.label, e.name),
                                );
                            })
                            .response
                            .on_disabled_hover_text(format!("Non funziona qui: {reason}"));
                        }
                    });
                if st.checking {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Verifica dei codificatori video…");
                    });
                } else if st.ffmpeg.is_none() {
                    ui.label(
                        RichText::new("⚠ ffmpeg non trovato: serve per H.264/H.265. Indica dove si trova qui sotto.").color(WARN),
                    );
                } else {
                    match video::pick_encoder(codec, &working, &v.encoder) {
                        Some(e) if e.on_gpu() => {
                            ui.label(
                                RichText::new(format!(
                                    "✔ Codifica sulla scheda grafica: {}",
                                    e.label
                                ))
                                .small()
                                .color(Color32::LIGHT_GREEN),
                            );
                        }
                        Some(e) => {
                            ui.label(
                                RichText::new(format!(
                                    "⚠ Nessun codificatore GPU funzionante: verrà usato {}",
                                    e.label
                                ))
                                .small()
                                .color(WARN),
                            );
                        }
                        None => {
                            ui.label(
                                RichText::new(
                                    "⚠ Nessun codificatore funzionante per questo formato",
                                )
                                .color(ERROR),
                            );
                        }
                    }
                }
                if codec == video::Codec::H264 && (v.width > 4096 || v.height > 4096) {
                    ui.label(
                        RichText::new("Le schede grafiche codificano H.264 fino a 4096 pixel per lato: per l'8K usa H.265.")
                            .small()
                            .color(WARN),
                    );
                }
                ui.horizontal(|ui| {
                    ui.label("Compressione:");
                    ui.selectable_value(
                        &mut v.rate_mode,
                        RateMode::ConstantQuality,
                        "Qualità costante",
                    );
                    ui.selectable_value(&mut v.rate_mode, RateMode::Bitrate, "Bitrate");
                });
                match v.rate_mode {
                    RateMode::ConstantQuality => {
                        ui.add(egui::Slider::new(&mut v.quality, 0..=51).text("livello"));
                        ui.label(
                            RichText::new("0 = qualità massima (file grande) · 51 = file piccolo · consigliato 18–24")
                                .small()
                                .weak(),
                        );
                    }
                    RateMode::Bitrate => {
                        ui.add(
                            egui::Slider::new(&mut v.bitrate_mbps, 0.5..=400.0)
                                .logarithmic(true)
                                .text("Mbit/s"),
                        );
                    }
                }
                egui::ComboBox::from_label("Velocità di codifica")
                    .selected_text(v.speed.label())
                    .show_ui(ui, |ui| {
                        for s in Speed::ALL {
                            ui.selectable_value(&mut v.speed, s, s.label());
                        }
                    });
            }
            VideoFormat::Raw => {
                let bytes = frames as f64 * v.width as f64 * v.height as f64 * 3.0;
                ui.label(
                    RichText::new(format!(
                        "RGB 8 bit senza compressione né intestazione: circa {}. Accanto viene scritto un file .txt con risoluzione e fps.",
                        human_bytes(bytes)
                    ))
                    .small()
                    .weak(),
                );
            }
            VideoFormat::TiffSequence => {
                egui::ComboBox::from_label("Compressione TIFF")
                    .selected_text(v.tiff_compression.label())
                    .show_ui(ui, |ui| {
                        for c in TiffCompression::ALL {
                            ui.selectable_value(&mut v.tiff_compression, c, c.label());
                        }
                    });
                ui.label(
                    RichText::new("Un file TIFF per fotogramma, numerati in ordine.")
                        .small()
                        .weak(),
                );
            }
        }

        ui.add_space(4.0);
        let mut recheck = false;
        egui::CollapsingHeader::new("ffmpeg (per H.264/H.265)")
            .default_open(st.ffmpeg.is_none() && !st.checking)
            .show(ui, |ui| {
                match &st.ffmpeg {
                    Some(p) => ui.label(RichText::new(format!("Trovato: {}", p.display())).small()),
                    None => ui.label(RichText::new("Non trovato").small()),
                };
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut self.s.ffmpeg_path)
                            .hint_text("Percorso di ffmpeg (facoltativo)")
                            .desired_width(ui.available_width() - 80.0),
                    );
                    if ui.button("Sfoglia…").clicked() {
                        if let Some(p) = rfd::FileDialog::new()
                            .set_title("Dov'è ffmpeg?")
                            .pick_file()
                        {
                            self.s.ffmpeg_path = p.display().to_string();
                            recheck = true;
                        }
                    }
                });
                if ui.button("Ricontrolla i codificatori").clicked() {
                    recheck = true;
                }
                let ok: Vec<&str> = st
                    .probes
                    .iter()
                    .filter(|p| p.works)
                    .map(|p| p.name)
                    .collect();
                if !ok.is_empty() {
                    ui.label(
                        RichText::new(format!("Funzionanti: {}", ok.join(", ")))
                            .small()
                            .weak(),
                    );
                }
            });
        if recheck {
            workers::check_encoders(
                self.encoders.clone(),
                self.s.ffmpeg_path.clone(),
                self.s.video.clone(),
                ui.ctx().clone(),
            );
        }

        ui.add_space(6.0);
        let can = self.animation.keyframes.len() >= 2;
        if ui
            .add_enabled(
                can,
                egui::Button::new(RichText::new("🎬 Esporta animazione…").strong())
                    .min_size(Vec2::new(ui.available_width(), 30.0)),
            )
            .on_disabled_hover_text("Servono almeno due keyframe")
            .clicked()
        {
            self.start_video_export();
        }
        ui.label(
            RichText::new(format!(
                "I fotogrammi vengono calcolati con: {}",
                self.engine_words()
            ))
            .small()
            .weak(),
        );
    }

    pub(super) fn image_dialog(&mut self, ctx: &egui::Context) {
        let mut open = self.show_image_dialog;
        let mut save = false;
        let (vw, vh) = self.view_px;
        let (cw, ch) = (self.image_w, self.image_h);
        let label = move |i: usize| -> String {
            match i {
                0 => format!("Come la finestra ({vw}×{vh})"),
                i if i <= RESOLUTIONS.len() => RESOLUTIONS[i - 1].0.to_string(),
                _ => format!("Personalizzata ({cw}×{ch})"),
            }
        };
        egui::Window::new("Salva immagine")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                egui::ComboBox::from_label("Dimensione")
                    .selected_text(label(self.image_size))
                    .show_ui(ui, |ui| {
                        for i in 0..=RESOLUTIONS.len() + 1 {
                            ui.selectable_value(&mut self.image_size, i, label(i));
                        }
                    });
                if self.image_size > RESOLUTIONS.len() {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.image_w).range(16..=30000).speed(4.0).suffix(" px"));
                        ui.label("×");
                        ui.add(egui::DragValue::new(&mut self.image_h).range(16..=30000).speed(4.0).suffix(" px"));
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Formato:");
                    for f in ImageFormat::ALL {
                        ui.selectable_value(&mut self.s.image.format, f, f.label());
                    }
                });
                match self.s.image.format {
                    ImageFormat::Jpeg => {
                        ui.add(egui::Slider::new(&mut self.s.image.jpeg_quality, 1..=100).text("qualità JPEG"));
                        ui.label(
                            RichText::new("100 = qualità massima e file più grande · 1 = compressione massima")
                                .small()
                                .weak(),
                        );
                    }
                    ImageFormat::Tiff => {
                        egui::ComboBox::from_label("Compressione TIFF")
                            .selected_text(self.s.image.tiff_compression.label())
                            .show_ui(ui, |ui| {
                                for c in TiffCompression::ALL {
                                    ui.selectable_value(&mut self.s.image.tiff_compression, c, c.label());
                                }
                            });
                    }
                    ImageFormat::Png => {
                        ui.label(RichText::new("PNG: senza perdita di qualità").small().weak());
                    }
                }
                egui::ComboBox::from_label("Antialiasing")
                    .selected_text(ss_label(self.s.image_supersampling))
                    .show_ui(ui, |ui| {
                        for k in 1..=4 {
                            ui.selectable_value(&mut self.s.image_supersampling, k, ss_label(k));
                        }
                    });
                let (w, h) = self.image_dimensions();
                if !jobs::fits(w, h, self.s.image_supersampling.max(1)) {
                    ui.label(RichText::new("⚠ Troppo grande con questo antialiasing").color(ERROR));
                }
                ui.label(
                    RichText::new(format!(
                        "{w}×{h} pixel · {} iterazioni · calcolo su {}",
                        group_digits(self.s.iterations),
                        self.engine_words()
                    ))
                    .small()
                    .weak(),
                );
                ui.label(
                    RichText::new("L'altezza inquadrata resta quella della vista; con proporzioni diverse cambia la larghezza.")
                        .small()
                        .weak(),
                );
                ui.add_space(4.0);
                if ui.button(RichText::new("💾 Salva…").strong()).clicked() {
                    save = true;
                }
            });
        self.show_image_dialog = open;
        if save {
            self.start_image_save();
        }
    }

    pub(super) fn jobs_windows(&mut self, ctx: &egui::Context) {
        let mut finished = Vec::new();
        for (i, job) in self.jobs.iter().enumerate() {
            if job.done.load(Ordering::SeqCst) {
                finished.push(i);
                continue;
            }
            egui::Window::new(&job.title)
                .id(egui::Id::new(("job", Arc::as_ptr(job) as usize)))
                .collapsible(false)
                .resizable(false)
                .anchor(
                    egui::Align2::CENTER_BOTTOM,
                    Vec2::new(0.0, -40.0 - 120.0 * i as f32),
                )
                .show(ctx, |ui| {
                    ui.set_min_width(440.0);
                    ui.add(egui::ProgressBar::new(job.fraction()).show_percentage());
                    ui.label(RichText::new(job.message.lock().unwrap().clone()).small());
                    if let Some(t) = *job.started.lock().unwrap() {
                        ui.label(
                            RichText::new(format!(
                                "Tempo trascorso: {}",
                                jobs::format_duration(t.elapsed().as_secs_f64())
                            ))
                            .small()
                            .weak(),
                        );
                    }
                    if ui.button("Annulla").clicked() {
                        job.cancel.store(true, Ordering::SeqCst);
                    }
                });
            ctx.request_repaint_after(Duration::from_millis(200));
        }
        for i in finished.into_iter().rev() {
            let job = self.jobs.remove(i);
            let result = job.result.lock().unwrap().take();
            match result {
                Some(Ok(msg)) => self.notify(msg, false),
                Some(Err(msg)) => self.notify(msg, true),
                None => {}
            }
        }
    }

    pub(super) fn help_windows(&mut self, ctx: &egui::Context) {
        let mut open = self.show_help;
        egui::Window::new("Comandi")
            .open(&mut open)
            .collapsible(false)
            .default_width(480.0)
            .show(ctx, |ui| {
                ui.label(RichText::new("Mouse").strong());
                egui::Grid::new("help-mouse")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        for (a, b) in [
                            ("Rotellina", "zoom avanti/indietro nel punto indicato"),
                            ("Trascina (tasto sinistro)", "sposta la vista"),
                            (
                                "Trascina (tasto destro)",
                                "disegna un rettangolo e ingrandiscilo",
                            ),
                            ("Doppio clic sinistro", "centra e ingrandisci 2×"),
                            ("Doppio clic destro", "riduci 2×"),
                            ("Clic con la rotellina", "centra nel punto"),
                        ] {
                            ui.label(a);
                            ui.label(b);
                            ui.end_row();
                        }
                    });
                ui.add_space(6.0);
                ui.label(RichText::new("Tastiera").strong());
                egui::Grid::new("help-keys")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        for (a, b) in [
                            ("+  /  PagSu", "zoom avanti 2× (con Maiusc 1,25×)"),
                            ("−  /  PagGiù", "zoom indietro (con Maiusc 1,25×)"),
                            ("Frecce", "sposta la vista (con Maiusc a piccoli passi)"),
                            ("Home", "vista iniziale"),
                            ("[   ]", "meno / più iterazioni"),
                            ("I", "usa le iterazioni consigliate"),
                            ("G", "calcolo su scheda grafica / processore"),
                            ("K", "aggiungi keyframe"),
                            ("Spazio", "anteprima dell'animazione / pausa"),
                            ("C", "copia le coordinate"),
                            ("Ctrl+S", "salva immagine"),
                            ("Ctrl+Maiusc+S", "salva coordinate"),
                            ("Ctrl+O", "carica coordinate"),
                            ("Esc", "chiudi l'anteprima o le finestre"),
                            ("F1", "questa guida"),
                            ("F2  /  F3", "riduci o apri il pannello sinistro / destro"),
                        ] {
                            ui.label(RichText::new(a).monospace());
                            ui.label(b);
                            ui.end_row();
                        }
                    });
                ui.add_space(6.0);
                ui.checkbox(
                    &mut self.s.show_help_on_start,
                    "Mostra questa guida all'avvio",
                );
            });
        self.show_help = open;

        let mut about = self.show_about;
        egui::Window::new("Informazioni")
            .open(&mut about)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.heading("Mandelbrot Studio");
                ui.label(format!("Versione {}", env!("CARGO_PKG_VERSION")));
                ui.label("Esploratore dell'insieme di Mandelbrot con calcolo su CPU o GPU.");
                ui.label("Licenza GNU GPL v3 o successiva.");
                ui.label(
                    RichText::new(format!(
                        "Impostazioni e palette in: {}",
                        settings::config_dir().display()
                    ))
                    .small()
                    .weak(),
                );
            });
        self.show_about = about;
    }
}
