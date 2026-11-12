//! The fractal view in the middle of the window, with its mouse controls.

use std::sync::atomic::Ordering;

use eframe::egui::{self, Color32, PointerButton, Pos2, Rect, Sense, Stroke, Vec2};

use super::MandelApp;
use crate::view::View;

impl MandelApp {
    pub(super) fn viewer(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();
        self.view_px = (
            ((rect.width() * ppp).round() as u32).max(16),
            ((rect.height() * ppp).round() as u32).max(16),
        );
        let (w, h) = (rect.width() as f64, rect.height() as f64);
        let local = |p: Pos2| ((p.x - rect.min.x) as f64, (p.y - rect.min.y) as f64);

        self.hover = resp.hover_pos().map(|p| {
            let (x, y) = local(p);
            self.s.view.pixel_to_complex(x, y, w, h)
        });

        self.view_points = (w, h);

        // Zooms are animated (a short eased zoom, as on mandelbrot.site); dragging follows
        // the mouse at once.
        if resp.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.raw_scroll_delta.y, i.zoom_delta()));
            let factor = (scroll as f64 * 0.0045).exp() * pinch as f64;
            if (factor - 1.0).abs() > 1e-6 {
                if let Some(p) = resp.hover_pos() {
                    let (x, y) = local(p);
                    self.zoom_about(x, y, factor, super::WHEEL_STEP_SECS);
                }
            }
        }
        let mut changed = false;
        if resp.dragged_by(PointerButton::Primary) {
            let d = resp.drag_delta();
            if d != Vec2::ZERO {
                self.zoom_step = None;
                self.s.view.pan_pixels(d.x as f64, d.y as f64, h);
                changed = true;
            }
        }
        if resp.drag_started_by(PointerButton::Secondary) {
            self.selection_start = resp.interact_pointer_pos();
        }
        let pointer = ui.input(|i| i.pointer.interact_pos());
        let selection = match (self.selection_start, pointer) {
            (Some(a), Some(b)) => Some(Rect::from_two_pos(a, b)),
            _ => None,
        };
        if resp.drag_stopped_by(PointerButton::Secondary) {
            if let Some(sel) = selection {
                if sel.width() > 6.0 && sel.height() > 6.0 {
                    let (sx, sy) = local(sel.center());
                    let (x, y) = self.s.view.pixel_to_complex(sx, sy, w, h);
                    let f = (sel.width() as f64 / w).max(sel.height() as f64 / h);
                    let mut target = View { cx: x, cy: y, scale: self.s.view.scale * f };
                    target.clamp();
                    self.zoom_to(target, super::ZOOM_STEP_SECS * 1.6);
                }
            }
            self.selection_start = None;
        }
        if resp.double_clicked_by(PointerButton::Primary) {
            if let Some(p) = resp.interact_pointer_pos() {
                let (x, y) = local(p);
                let (cx, cy) = self.s.view.pixel_to_complex(x, y, w, h);
                let mut target = View { cx, cy, scale: self.target_view().scale / 2.0 };
                target.clamp();
                self.zoom_to(target, super::ZOOM_STEP_SECS);
            }
        }
        if resp.double_clicked_by(PointerButton::Secondary) {
            if let Some(p) = resp.interact_pointer_pos() {
                let (x, y) = local(p);
                self.zoom_about(x, y, 0.5, super::ZOOM_STEP_SECS);
            }
        }
        if resp.clicked_by(PointerButton::Middle) {
            if let Some(p) = resp.interact_pointer_pos() {
                let (x, y) = local(p);
                let (cx, cy) = self.s.view.pixel_to_complex(x, y, w, h);
                let mut target = View { cx, cy, scale: self.target_view().scale };
                target.clamp();
                self.zoom_to(target, super::ZOOM_STEP_SECS);
            }
        }
        if changed {
            self.navigated();
            self.moved = true;
        }

        // Paint the pictures where they belong in the current view, so zooming and moving
        // respond at once while the new picture is calculated. As on mandelbrot.site, older
        // pictures (other zoom levels) stay underneath, scaled, and each new one fades in on
        // top: a jump or a zoom never shows empty background.
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_gray(12));
        let corner = |x: f64, y: f64| {
            let (px, py) = self.s.view.complex_to_pixel(x, y, w, h);
            Pos2::new(rect.min.x + px as f32, rect.min.y + py as f32)
        };
        let placed: Vec<Option<Rect>> = self
            .layers
            .iter()
            .map(|l| {
                let v0 = l.view;
                let img = Rect::from_two_pos(
                    corner(v0.cx - v0.scale * l.aspect, v0.cy + v0.scale),
                    corner(v0.cx + v0.scale * l.aspect, v0.cy - v0.scale),
                );
                (img.is_finite() && img.width() < 1.0e6 && img.height() < 1.0e6 && img.width() > 0.5)
                    .then_some(img)
            })
            .collect();
        let n = self.layers.len();
        let mut fading = false;
        for (i, (layer, img)) in self.layers.iter().zip(&placed).enumerate() {
            let Some(img) = img else { continue };
            let alpha = if i + 1 == n {
                (layer.born.elapsed().as_secs_f32() / super::FADE.as_secs_f32()).min(1.0)
            } else {
                1.0
            };
            fading |= alpha < 1.0;
            painter.image(
                layer.tex.id(),
                *img,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE.gamma_multiply(alpha),
            );
        }
        if fading {
            ui.ctx().request_repaint();
        }
        if let Some(sel) = selection {
            painter.rect_filled(sel, 0.0, Color32::from_rgba_unmultiplied(255, 255, 255, 28));
            painter.rect_stroke(
                sel,
                0.0,
                Stroke::new(1.5f32, Color32::WHITE),
                egui::StrokeKind::Middle,
            );
        }

        // Small notes in the corner.
        let mut lines: Vec<String> = Vec::new();
        if self.worker.busy.load(Ordering::Relaxed) {
            let rows = self.worker.rows_done.load(Ordering::Relaxed);
            let total = self.last_request.map(|r| r.h).unwrap_or(1).max(1);
            let pct = (rows as f32 / total as f32 * 100.0).min(100.0);
            lines.push(format!("Calcolo in corso… {pct:.0}%"));
        }
        if self.preview.is_some() {
            lines.push(format!(
                "Anteprima animazione · {:.2} s di {:.2} s",
                self.anim_time,
                self.animation.total_duration()
            ));
        }
        if self.s.view.at_precision_limit() {
            lines.push("Limite di precisione raggiunto: questo è lo zoom massimo".into());
        }
        let mut y = rect.min.y + 8.0;
        for l in lines {
            let galley =
                painter.layout_no_wrap(l, egui::FontId::proportional(13.0), Color32::WHITE);
            let r = Rect::from_min_size(
                Pos2::new(rect.min.x + 8.0, y),
                galley.size() + Vec2::new(12.0, 6.0),
            );
            painter.rect_filled(r, 4.0, Color32::from_black_alpha(170));
            painter.galley(r.min + Vec2::new(6.0, 3.0), galley, Color32::WHITE);
            y += r.height() + 4.0;
        }
    }
}
