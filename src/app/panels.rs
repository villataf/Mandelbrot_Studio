//! The left panel (position, iterations, engine, colours) and the status bar.

use eframe::egui::{self, Color32, Key, Pos2, Rect, RichText, Sense, Stroke, Vec2};

use super::{group_digits, MandelApp, MAX_ITERATIONS};
use crate::compute::gpu::{Precision, PrecisionChoice};
use crate::compute::{Engine, GpuState};
use crate::palette::{self, ColorStop, Mapping, Palette};
use crate::settings::{self, Location};
use crate::view::{format_zoom, parse_number, View};

const WARN: Color32 = Color32::from_rgb(255, 200, 80);
const ERROR: Color32 = Color32::from_rgb(255, 140, 120);

impl MandelApp {
    pub(super) fn left_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Comandi");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("«").on_hover_text("Riduci il pannello (F2)").clicked() {
                    self.s.show_left_panel = false;
                }
            });
        });
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::CollapsingHeader::new(RichText::new("📍 Posizione").strong())
                    .default_open(true)
                    .show(ui, |ui| self.position_section(ui));
                egui::CollapsingHeader::new(RichText::new("🔁 Iterazioni").strong())
                    .default_open(true)
                    .show(ui, |ui| self.iterations_section(ui));
                egui::CollapsingHeader::new(RichText::new("⚙ Calcolo").strong())
                    .default_open(true)
                    .show(ui, |ui| self.engine_section(ui));
                egui::CollapsingHeader::new(RichText::new("🎨 Colori").strong())
                    .default_open(true)
                    .show(ui, |ui| self.colors_section(ui));
            });
    }

    fn position_section(&mut self, ui: &mut egui::Ui) {
        let decimals = self.s.view.coordinate_decimals(self.view_px.1);
        if self.edits_for != Some(self.s.view) && !self.editing_coords {
            self.edit_re = format!("{:.*}", decimals, self.s.view.cx);
            self.edit_im = format!("{:.*}", decimals, self.s.view.cy);
            self.edit_zoom = format!("{:.6e}", self.s.view.zoom());
            self.edits_for = Some(self.s.view);
        }
        let mut go = false;
        let mut editing = false;
        egui::Grid::new("coords")
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for (label, text) in [
                    ("Reale (Re)", &mut self.edit_re),
                    ("Immaginaria (Im)", &mut self.edit_im),
                    ("Zoom (×)", &mut self.edit_zoom),
                ] {
                    ui.label(label);
                    let r = ui.add(
                        egui::TextEdit::singleline(text)
                            .desired_width(f32::INFINITY)
                            .font(egui::TextStyle::Monospace),
                    );
                    editing |= r.has_focus();
                    if r.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)) {
                        go = true;
                    }
                    ui.end_row();
                }
            });
        self.editing_coords = editing;
        ui.horizontal(|ui| {
            if ui
                .button("➡ Vai")
                .on_hover_text("Vai alle coordinate scritte qui sopra (o premi Invio)")
                .clicked()
            {
                go = true;
            }
            if ui
                .button("⟲ Reimposta")
                .on_hover_text("Torna alla vista iniziale (tasto Home)")
                .clicked()
            {
                self.reset_view();
            }
            if ui
                .button("📋 Copia")
                .on_hover_text("Copia le coordinate negli appunti (tasto C)")
                .clicked()
            {
                self.copy_coordinates(ui.ctx());
            }
        });
        ui.checkbox(&mut self.s.animate_moves, "Transizione animata")
            .on_hover_text("Quando cambi posizione o zoomi: movimento fluido invece di un salto");
        if go {
            match (
                parse_number(&self.edit_re),
                parse_number(&self.edit_im),
                parse_number(&self.edit_zoom),
            ) {
                (Some(re), Some(im), Some(z)) if z > 0.0 => {
                    let mut target = View { cx: re, cy: im, scale: self.s.view.scale };
                    target.set_zoom(z);
                    let it = self.s.iterations;
                    self.go_to(target, it);
                    self.edits_for = None;
                    self.editing_coords = false;
                }
                _ => self.notify(
                    "Coordinate non valide: usa numeri come -0.75, 0.1 e 1e6",
                    true,
                ),
            }
        }
        ui.label(
            RichText::new(format!(
                "Altezza visibile {:.3e} · un pixel {:.2e}",
                2.0 * self.s.view.scale,
                self.s.view.pixel_size(self.view_px.1)
            ))
            .small()
            .weak(),
        );
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.paste_text)
                    .hint_text("Incolla qui coordinate copiate")
                    .desired_width(ui.available_width() - 50.0),
            );
            if ui.button("Usa").clicked() {
                match Location::from_clipboard_text(&self.paste_text) {
                    Some(loc) => {
                        self.apply_location(&loc, true);
                        self.paste_text.clear();
                    }
                    None => self.notify(
                        "Testo non riconosciuto: servono re=… im=… e zoom=… oppure scala=…",
                        true,
                    ),
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.button("💾 Salva coordinate…").clicked() {
                self.save_coordinates();
            }
            if ui.button("📂 Carica…").clicked() {
                self.load_coordinates();
            }
        });

        ui.add_space(4.0);
        ui.label(RichText::new("Preferiti").strong());
        let mut go_to = None;
        let mut remove = None;
        for (i, fav) in self.s.favourites.iter().enumerate() {
            ui.horizontal(|ui| {
                let zoom = fav
                    .view()
                    .map(|v| format_zoom(v.zoom()))
                    .unwrap_or_default();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button("🗑")
                        .on_hover_text("Togli dai preferiti")
                        .clicked()
                    {
                        remove = Some(i);
                    }
                    ui.label(RichText::new(&zoom).small().weak());
                    ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(format!("★ {}", fav.name))
                                    .wrap_mode(egui::TextWrapMode::Truncate),
                            )
                            .on_hover_text(format!(
                                "Re {}\nIm {}\nZoom {zoom}\nIterazioni {}",
                                fav.center_re, fav.center_im, fav.iterations
                            ))
                            .clicked()
                        {
                            go_to = Some(i);
                        }
                    });
                });
            });
        }
        if let Some(i) = go_to {
            let loc = self.s.favourites[i].clone();
            self.apply_location(&loc, true);
        }
        if let Some(i) = remove {
            self.s.favourites.remove(i);
            let _ = self.s.save();
        }
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.fav_name)
                    .hint_text("Nome")
                    .desired_width(ui.available_width() - 90.0),
            );
            if ui
                .button("★ Aggiungi")
                .on_hover_text("Salva questa posizione tra i preferiti")
                .clicked()
            {
                let name = if self.fav_name.trim().is_empty() {
                    format!("Posizione {}", self.s.favourites.len() + 1)
                } else {
                    self.fav_name.trim().to_string()
                };
                self.s.favourites.push(Location::new(
                    &name,
                    &self.s.view,
                    self.s.iterations,
                    Some(self.s.color.clone()),
                ));
                self.fav_name.clear();
                let _ = self.s.save();
            }
        });
    }

    fn iterations_section(&mut self, ui: &mut egui::Ui) {
        let mut it = self.s.iterations;
        ui.add(
            egui::Slider::new(&mut it, 16..=1_000_000)
                .logarithmic(true)
                .text("iterazioni")
                .clamping(egui::SliderClamping::Never),
        );
        ui.horizontal(|ui| {
            ui.label("Valore esatto:");
            ui.add(
                egui::DragValue::new(&mut it)
                    .range(1..=MAX_ITERATIONS)
                    .speed(10.0),
            );
            if ui.small_button("÷1,25").on_hover_text("Tasto [").clicked() {
                it = (it as f64 / 1.25).round() as u32;
            }
            if ui.small_button("×1,25").on_hover_text("Tasto ]").clicked() {
                it = (it as f64 * 1.25).round() as u32;
            }
        });
        if it != self.s.iterations {
            self.set_iterations(it);
        }

        ui.add_space(2.0);
        let current = self
            .recommendation
            .as_ref()
            .filter(|(v, _)| *v == self.s.view)
            .map(|(_, r)| r.clone());
        match current {
            Some(r) => {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(format!("Consigliate qui: {}", group_digits(r.iterations)))
                            .strong(),
                    );
                    if ui.button("Usa").on_hover_text("Tasto I").clicked() {
                        self.set_iterations(r.iterations);
                    }
                });
                ui.label(RichText::new(&r.reason).small().weak());
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!(
                        "Analisi della zona… (stima {})",
                        group_digits(crate::recommend::heuristic(&self.s.view))
                    ));
                });
            }
        }
        ui.checkbox(
            &mut self.s.auto_iterations,
            "Usa sempre le iterazioni consigliate",
        );
    }

    fn engine_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.s.engine, Engine::Gpu, "🖵 Scheda grafica (GPU)");
            ui.selectable_value(&mut self.s.engine, Engine::Cpu, "🖩 Processore (CPU)");
        });
        ui.label(
            RichText::new("Tasto G per passare dall'uno all'altro")
                .small()
                .weak(),
        );

        if self.s.engine == Engine::Gpu {
            let adapters = self.adapters.lock().unwrap().clone();
            let auto = "Automatica (la più potente)".to_string();
            let mut chosen = self.s.gpu_adapter.clone();
            egui::ComboBox::from_id_salt("gpu-adapter")
                .selected_text(chosen.clone().unwrap_or(auto.clone()))
                .width(ui.available_width() - 8.0)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut chosen, None, auto);
                    for a in &adapters {
                        let label = format!(
                            "{} · {}{}",
                            a.key,
                            a.kind,
                            if a.supports_f64 {
                                " · doppia nativa"
                            } else {
                                ""
                            }
                        );
                        ui.selectable_value(&mut chosen, Some(a.key.clone()), label);
                    }
                });
            if chosen != self.s.gpu_adapter {
                self.s.gpu_adapter = chosen;
                self.restart_gpu(ui.ctx());
            }

            egui::ComboBox::from_label("Precisione")
                .selected_text(self.s.precision.label())
                .show_ui(ui, |ui| {
                    for c in [
                        PrecisionChoice::Auto,
                        PrecisionChoice::Fixed(Precision::Single),
                        PrecisionChoice::Fixed(Precision::Perturbation),
                        PrecisionChoice::Fixed(Precision::DoubleEmulated),
                        PrecisionChoice::Fixed(Precision::DoubleNative),
                    ] {
                        ui.selectable_value(&mut self.s.precision, c, c.label());
                    }
                });

            let mut retry = false;
            match &*self.hub.gpu.read().unwrap() {
                GpuState::NotStarted | GpuState::Starting => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("Preparazione della scheda grafica…");
                    });
                }
                GpuState::Ready(ctx) => {
                    ui.label(
                        RichText::new(format!(
                            "✔ {} ({}, {})",
                            ctx.entry.name, ctx.entry.backend, ctx.entry.kind
                        ))
                        .color(Color32::LIGHT_GREEN),
                    );
                    for n in &ctx.notes {
                        ui.label(RichText::new(format!("• {n}")).small().weak());
                    }
                    if let PrecisionChoice::Fixed(p) = self.s.precision {
                        if !ctx.has(p) {
                            ui.label(
                                RichText::new(
                                    "Questa precisione non è disponibile qui: scelta automatica.",
                                )
                                .small()
                                .color(WARN),
                            );
                        }
                        if matches!(p, Precision::DoubleEmulated | Precision::DoubleNative)
                            && ctx.has(Precision::Perturbation)
                        {
                            ui.label(
                                RichText::new(
                                    "Suggerimento: «Automatica» o «Perturbazione» sono da 3 a 10 volte più veloci negli zoom profondi.",
                                )
                                .small()
                                .color(WARN),
                            );
                        }
                    }
                    if ctx
                        .pick_precision(self.s.precision, &self.s.view, self.view_px.1)
                        .is_none()
                    {
                        ui.label(
                            RichText::new("A questo zoom la scheda non ha abbastanza precisione: il calcolo passa al processore.")
                                .small()
                                .color(WARN),
                        );
                    }
                }
                GpuState::Failed(e) => {
                    ui.label(RichText::new(format!("⚠ GPU non disponibile: {e}")).color(ERROR));
                    ui.label(
                        RichText::new("Il calcolo usa il processore.")
                            .small()
                            .weak(),
                    );
                    retry = ui.button("Riprova").clicked();
                }
            }
            if retry {
                self.restart_gpu(ui.ctx());
            }
        } else {
            ui.label(
                RichText::new(format!(
                    "{} core, doppia precisione",
                    rayon::current_num_threads()
                ))
                .small()
                .weak(),
            );
        }
        if let Some(sh) = &self.shown {
            ui.label(
                RichText::new(format!(
                    "Ultimo calcolo: {} ms · {}×{} · {}",
                    sh.elapsed.as_millis(),
                    sh.w,
                    sh.h,
                    sh.used
                ))
                .small()
                .weak(),
            );
        }
    }

    fn colors_section(&mut self, ui: &mut egui::Ui) {
        if self.preview_color.is_some() {
            ui.label(
                RichText::new("Durante l'anteprima si vedono i colori dei keyframe; qui modifichi quelli della vista.")
                    .small()
                    .color(WARN),
            );
        }
        let presets = palette::presets();
        egui::ComboBox::from_label("Palette")
            .selected_text(&self.s.color.palette.name)
            .width(ui.available_width() - 70.0)
            .show_ui(ui, |ui| {
                ui.label(RichText::new("Predefinite").weak());
                for p in &presets {
                    if ui
                        .selectable_label(self.s.color.palette.name == p.name, &p.name)
                        .clicked()
                    {
                        self.s.color.palette = p.clone();
                    }
                }
                if !self.custom_palettes.is_empty() {
                    ui.separator();
                    ui.label(RichText::new("Personalizzate").weak());
                    for p in &self.custom_palettes {
                        if ui
                            .selectable_label(
                                self.s.color.palette.name == p.name,
                                format!("★ {}", p.name),
                            )
                            .clicked()
                        {
                            self.s.color.palette = p.clone();
                        }
                    }
                }
            });
        gradient_bar(ui, &self.s.color.palette, 20.0);

        egui::ComboBox::from_label("Distribuzione")
            .selected_text(self.s.color.mapping.label())
            .show_ui(ui, |ui| {
                for m in Mapping::ALL {
                    ui.selectable_value(&mut self.s.color.mapping, m, m.label());
                }
            });
        ui.add(
            egui::Slider::new(&mut self.s.color.density, 0.02..=50.0)
                .logarithmic(true)
                .text("ripetizioni"),
        );
        ui.add(egui::Slider::new(&mut self.s.color.offset, 0.0..=1.0).text("spostamento"));

        egui::CollapsingHeader::new("✏ Modifica i colori")
            .default_open(false)
            .show(ui, |ui| {
                let n = self.s.color.palette.stops.len();
                let mut remove = None;
                for (i, stop) in self.s.color.palette.stops.iter_mut().enumerate() {
                    ui.horizontal(|ui| {
                        egui::color_picker::color_edit_button_srgb(ui, &mut stop.color);
                        ui.add(egui::Slider::new(&mut stop.pos, 0.0..=1.0).fixed_decimals(3));
                        if n > 1
                            && ui
                                .small_button("✖")
                                .on_hover_text("Togli questo colore")
                                .clicked()
                        {
                            remove = Some(i);
                        }
                    });
                }
                if let Some(i) = remove {
                    self.s.color.palette.stops.remove(i);
                }
                ui.horizontal(|ui| {
                    if ui.button("➕ Colore").clicked() {
                        add_stop(&mut self.s.color.palette);
                    }
                    if ui
                        .button("Distribuisci")
                        .on_hover_text("Spazia i colori in modo uniforme")
                        .clicked()
                    {
                        self.s.color.palette.distribute_evenly();
                    }
                    if ui.button("Inverti").clicked() {
                        self.s.color.palette.reverse();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Colore dell'interno:");
                    egui::color_picker::color_edit_button_srgb(
                        ui,
                        &mut self.s.color.palette.interior,
                    );
                });
            });

        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.palette_name)
                    .hint_text("Nome della palette")
                    .desired_width(ui.available_width() - 120.0),
            );
            if ui.button("💾 Salva palette").clicked() {
                self.save_custom_palette();
            }
        });
        ui.horizontal(|ui| {
            let is_custom = self
                .custom_palettes
                .iter()
                .any(|p| p.name == self.s.color.palette.name);
            if ui
                .add_enabled(is_custom, egui::Button::new("🗑 Elimina"))
                .on_hover_text("Elimina la palette personalizzata selezionata")
                .clicked()
            {
                let name = self.s.color.palette.name.clone();
                self.custom_palettes.retain(|p| p.name != name);
                let _ = settings::save_custom_palettes(&self.custom_palettes);
                self.notify(format!("Palette \"{name}\" eliminata"), false);
            }
            if ui.button("Esporta…").clicked() {
                self.export_palette();
            }
            if ui.button("Importa…").clicked() {
                self.import_palettes();
            }
        });
    }

    fn save_custom_palette(&mut self) {
        let mut name = self.palette_name.trim().to_string();
        if name.is_empty() {
            name = self.s.color.palette.name.clone();
        }
        if palette::presets().iter().any(|p| p.name == name) {
            name = format!("{name} (personale)");
        }
        let mut p = self.s.color.palette.clone();
        p.name = name.clone();
        match self.custom_palettes.iter_mut().find(|c| c.name == name) {
            Some(existing) => *existing = p.clone(),
            None => self.custom_palettes.push(p.clone()),
        }
        self.s.color.palette = p;
        self.palette_name.clear();
        match settings::save_custom_palettes(&self.custom_palettes) {
            Ok(()) => self.notify(format!("Palette \"{name}\" salvata"), false),
            Err(e) => self.notify(format!("Impossibile salvare la palette: {e}"), true),
        }
    }

    fn export_palette(&mut self) {
        let name = format!("{}.palette.json", self.s.color.palette.name);
        if let Some(path) = self.save_dialog("Esporta palette", &name, ("Palette", &["json"])) {
            let r = serde_json::to_string_pretty(&self.s.color.palette)
                .map_err(|e| e.to_string())
                .and_then(|t| std::fs::write(&path, t).map_err(|e| e.to_string()));
            match r {
                Ok(()) => self.notify("Palette esportata", false),
                Err(e) => self.notify(e, true),
            }
        }
    }

    fn import_palettes(&mut self) {
        let Some(path) = self.open_dialog("Importa palette", ("Palette", &["json"])) else {
            return;
        };
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => return self.notify(e.to_string(), true),
        };
        let list: Vec<Palette> = serde_json::from_str::<Vec<Palette>>(&text)
            .or_else(|_| serde_json::from_str::<Palette>(&text).map(|p| vec![p]))
            .unwrap_or_default();
        if list.is_empty() {
            return self.notify("Il file non contiene palette valide", true);
        }
        let count = list.len();
        for p in list {
            match self.custom_palettes.iter_mut().find(|c| c.name == p.name) {
                Some(existing) => *existing = p.clone(),
                None => self.custom_palettes.push(p.clone()),
            }
            self.s.color.palette = p;
        }
        let _ = settings::save_custom_palettes(&self.custom_palettes);
        self.notify(format!("{count} palette importate"), false);
    }

    pub(super) fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let d = self.s.view.coordinate_decimals(self.view_px.1);
            let (x, y) = self.hover.unwrap_or((self.s.view.cx, self.s.view.cy));
            ui.label(RichText::new(format!("Re {x:+.d$}  Im {y:+.d$}")).monospace());
            ui.separator();
            ui.label(format!("Zoom {}", format_zoom(self.s.view.zoom())));
            ui.separator();
            ui.label(format!("{} iterazioni", group_digits(self.s.iterations)));
            if let Some(sh) = &self.shown {
                ui.separator();
                let tag = if sh.preview {
                    " (anteprima veloce)"
                } else {
                    ""
                };
                ui.label(
                    RichText::new(format!("{} ms{tag} · {}", sh.elapsed.as_millis(), sh.used))
                        .weak(),
                );
            }
            if let Some(t) = &self.toast {
                ui.separator();
                ui.label(RichText::new(&t.text).color(if t.error {
                    ERROR
                } else {
                    Color32::LIGHT_GREEN
                }));
            }
        });
    }
}

/// A strip showing a palette.
pub(super) fn gradient_bar(ui: &mut egui::Ui, p: &Palette, height: f32) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
    let painter = ui.painter_at(rect);
    let n = 96;
    for i in 0..n {
        let t0 = i as f32 / n as f32;
        let t1 = (i + 1) as f32 / n as f32;
        let c = p.sample((t0 + t1) * 0.5);
        let r = Rect::from_min_max(
            Pos2::new(rect.min.x + rect.width() * t0, rect.min.y),
            Pos2::new(rect.min.x + rect.width() * t1 + 0.5, rect.max.y),
        );
        painter.rect_filled(
            r,
            0.0,
            Color32::from_rgb(c[0] as u8, c[1] as u8, c[2] as u8),
        );
    }
    painter.rect_stroke(
        rect,
        2.0,
        Stroke::new(1.0f32, Color32::from_gray(90)),
        egui::StrokeKind::Inside,
    );
}

/// Add a colour in the middle of the widest gap.
fn add_stop(p: &mut Palette) {
    let mut s: Vec<f32> = p.stops.iter().map(|s| s.pos).collect();
    s.sort_by(|a, b| a.total_cmp(b));
    let mut best = (0.0f32, 0.5f32);
    for i in 0..s.len() {
        let a = s[i];
        let b = if i + 1 < s.len() {
            s[i + 1]
        } else {
            s[0] + 1.0
        };
        if b - a > best.0 {
            best = (b - a, (a + (b - a) * 0.5).rem_euclid(1.0));
        }
    }
    let c = p.sample(best.1);
    p.stops.push(ColorStop {
        pos: best.1,
        color: [c[0] as u8, c[1] as u8, c[2] as u8],
    });
}
