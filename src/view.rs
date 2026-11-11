//! Position in the complex plane and the mapping between pixels and coordinates.

use serde::{Deserialize, Serialize};

/// Half of the visible height (in complex units) at zoom 1x.
pub const BASE_SCALE: f64 = 1.25;
/// Smallest half-height allowed: below this, 64-bit floating point runs out of digits.
pub const MIN_SCALE: f64 = 2.0e-12;
/// Largest half-height allowed.
pub const MAX_SCALE: f64 = 8.0;

/// A view of the plane: its centre and half of its visible height.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct View {
    pub cx: f64,
    pub cy: f64,
    pub scale: f64,
}

impl Default for View {
    fn default() -> Self {
        View {
            cx: -0.6,
            cy: 0.0,
            scale: BASE_SCALE,
        }
    }
}

impl View {
    /// Size of one pixel in complex units for an image `height` pixels tall.
    pub fn pixel_size(&self, height: u32) -> f64 {
        2.0 * self.scale / height.max(1) as f64
    }

    /// Magnification relative to the starting view.
    pub fn zoom(&self) -> f64 {
        BASE_SCALE / self.scale
    }

    pub fn set_zoom(&mut self, zoom: f64) {
        self.scale = BASE_SCALE / zoom;
        self.clamp();
    }

    pub fn clamp(&mut self) {
        if !self.scale.is_finite() {
            self.scale = BASE_SCALE;
        }
        self.scale = self.scale.clamp(MIN_SCALE, MAX_SCALE);
        if !self.cx.is_finite() {
            self.cx = 0.0;
        }
        if !self.cy.is_finite() {
            self.cy = 0.0;
        }
        self.cx = self.cx.clamp(-4.0, 4.0);
        self.cy = self.cy.clamp(-4.0, 4.0);
    }

    /// Whether the zoom is at the precision limit.
    pub fn at_precision_limit(&self) -> bool {
        self.scale <= MIN_SCALE * 1.0001
    }

    /// Complex coordinate of a point given in (continuous) pixel units of a `w`×`h` image.
    pub fn pixel_to_complex(&self, px: f64, py: f64, w: f64, h: f64) -> (f64, f64) {
        let ps = 2.0 * self.scale / h;
        (self.cx + (px - w * 0.5) * ps, self.cy - (py - h * 0.5) * ps)
    }

    /// Pixel position of a complex coordinate in a `w`×`h` image.
    pub fn complex_to_pixel(&self, x: f64, y: f64, w: f64, h: f64) -> (f64, f64) {
        let ps = 2.0 * self.scale / h;
        ((x - self.cx) / ps + w * 0.5, -(y - self.cy) / ps + h * 0.5)
    }

    /// Zoom by `factor` (>1 zooms in) keeping the point under (px, py) fixed.
    pub fn zoom_at(&mut self, px: f64, py: f64, w: f64, h: f64, factor: f64) {
        let (x, y) = self.pixel_to_complex(px, py, w, h);
        self.scale /= factor;
        self.scale = self.scale.clamp(MIN_SCALE, MAX_SCALE);
        let ps = 2.0 * self.scale / h;
        self.cx = x - (px - w * 0.5) * ps;
        self.cy = y + (py - h * 0.5) * ps;
        self.clamp();
    }

    /// Move the view by a number of pixels.
    pub fn pan_pixels(&mut self, dx: f64, dy: f64, h: f64) {
        let ps = 2.0 * self.scale / h;
        self.cx -= dx * ps;
        self.cy += dy * ps;
        self.clamp();
    }

    /// Number of decimals worth showing for coordinates at this zoom.
    pub fn coordinate_decimals(&self, height: u32) -> usize {
        let ps = self.pixel_size(height);
        ((-ps.log10()).ceil() as i64 + 3).clamp(4, 17) as usize
    }
}

/// A plain-text representation that survives a round trip exactly.
pub fn f64_exact(v: f64) -> String {
    format!("{v:?}")
}

/// Parse a number typed by a person: accepts a decimal comma and spaces.
pub fn parse_number(s: &str) -> Option<f64> {
    let cleaned: String = s
        .trim()
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '\'' && *c != '_')
        .map(|c| if c == ',' { '.' } else { c })
        .collect();
    let cleaned = cleaned.trim_end_matches(['x', 'X']);
    let v: f64 = cleaned.parse().ok()?;
    v.is_finite().then_some(v)
}

/// Human-friendly zoom, e.g. "1.25e+08×".
pub fn format_zoom(zoom: f64) -> String {
    if zoom < 1000.0 {
        format!("{zoom:.2}×")
    } else {
        format!("{zoom:.3e}×")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoom_keeps_point_under_cursor() {
        let mut v = View::default();
        let before = v.pixel_to_complex(100.0, 50.0, 800.0, 600.0);
        v.zoom_at(100.0, 50.0, 800.0, 600.0, 3.0);
        let after = v.pixel_to_complex(100.0, 50.0, 800.0, 600.0);
        assert!((before.0 - after.0).abs() < 1e-12);
        assert!((before.1 - after.1).abs() < 1e-12);
        assert!((v.zoom() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn pixel_round_trip() {
        let v = View {
            cx: -0.75,
            cy: 0.1,
            scale: 0.01,
        };
        let (x, y) = v.pixel_to_complex(123.0, 456.0, 1000.0, 800.0);
        let (px, py) = v.complex_to_pixel(x, y, 1000.0, 800.0);
        assert!((px - 123.0).abs() < 1e-6 && (py - 456.0).abs() < 1e-6);
    }

    #[test]
    fn parses_human_numbers() {
        assert_eq!(parse_number(" -0,75 "), Some(-0.75));
        assert_eq!(parse_number("1e10x"), Some(1e10));
        assert_eq!(parse_number("abc"), None);
        let x = -0.743_643_887_037_158_7_f64;
        assert_eq!(parse_number(&f64_exact(x)), Some(x));
    }
}
