//! Smooth "fly to" between two views: zoom out, travel, zoom back in, following the
//! optimal zoom-and-pan path of van Wijk and Nuij (2003), so the movement on screen looks
//! steady whatever the distance and the zoom levels.
//!
//! The constants follow the "fly to" of mandelbrot.site (its map library, Leaflet): ρ = 1.42,
//! 0.8 seconds per unit of path, an ease-out curve 1 − (1 − t)^1.5 that sets off at once and
//! lands gently, and the view measured by its larger side.

use crate::view::View;

/// How strongly the path zooms out to travel.
const RHO: f64 = 1.42;
/// Seconds per unit of path length at normal speed.
const SECONDS_PER_UNIT: f64 = 0.8;

#[derive(Clone, Debug)]
pub struct Flight {
    pub from: View,
    pub to: View,
    pub iterations_from: u32,
    pub iterations_to: u32,
    /// Seconds the flight lasts.
    pub duration: f64,
    // Path constants.
    dist: f64,
    r0: f64,
    length: f64,
    pure_zoom: Option<f64>,
    /// Larger side of the view divided by its half-height (2 × the wider of width/height ratio).
    side: f64,
}

/// The ease-out curve of mandelbrot.site's flights.
pub fn ease(t: f64) -> f64 {
    1.0 - (1.0 - t.clamp(0.0, 1.0)).powf(1.5)
}

impl Flight {
    /// A flight from `from` to `to` in a window `aspect` times as wide as it is tall;
    /// `speed` scales the duration (1 = normal).
    pub fn new(
        from: View,
        to: View,
        iterations_from: u32,
        iterations_to: u32,
        aspect: f64,
        speed: f64,
    ) -> Flight {
        let side = 2.0 * aspect.clamp(0.1, 10.0).max(1.0);
        let (w0, w1) = (from.scale.max(1e-300) * side, to.scale.max(1e-300) * side);
        let dist = ((to.cx - from.cx).powi(2) + (to.cy - from.cy).powi(2)).sqrt();
        let rho2 = RHO * RHO;
        let (r0, length, pure_zoom) = if dist < 1e-12 * w0.min(w1) {
            // Same centre: only zoom.
            let k = (w1 / w0).ln();
            (0.0, k.abs() / RHO, Some(k))
        } else {
            // b and r as in the paper; ln(-b + √(b²+1)) is written as -asinh(b) so it stays
            // accurate when one view is zoomed in very far (b huge).
            let b0 = (w1 * w1 - w0 * w0 + rho2 * rho2 * dist * dist) / (2.0 * w0 * rho2 * dist);
            let b1 = (w1 * w1 - w0 * w0 - rho2 * rho2 * dist * dist) / (2.0 * w1 * rho2 * dist);
            let r0 = -b0.asinh();
            let r1 = -b1.asinh();
            (r0, (r1 - r0) / RHO, None)
        };
        // Long journeys take longer, within reason.
        let duration = (SECONDS_PER_UNIT * length / speed.max(0.05)).clamp(0.4, 10.0);
        Flight { from, to, iterations_from, iterations_to, duration, dist, r0, length, pure_zoom, side }
    }

    /// Where the flight is at `t` (0..1 of its duration).
    pub fn at(&self, t: f64) -> (View, u32) {
        let t = t.clamp(0.0, 1.0);
        if t >= 1.0 {
            return (self.to, self.iterations_to);
        }
        let e = ease(t);
        let s = e * self.length;
        let w0 = self.from.scale.max(1e-300) * self.side;
        let (scale, along) = match self.pure_zoom {
            Some(k) => (w0 * (k * e).exp(), 0.0),
            None => {
                let rho2 = RHO * RHO;
                let r0 = self.r0;
                let u = w0 / rho2 * (r0.cosh() * (RHO * s + r0).tanh() - r0.sinh());
                let w = w0 * r0.cosh() / (RHO * s + r0).cosh();
                (w, (u / self.dist).clamp(0.0, 1.0))
            }
        };
        let mut view = View {
            cx: self.from.cx + (self.to.cx - self.from.cx) * along,
            cy: self.from.cy + (self.to.cy - self.from.cy) * along,
            scale: if scale.is_finite() && scale > 0.0 { scale / self.side } else { self.to.scale },
        };
        view.clamp();
        let li = (self.iterations_from.max(1) as f64).ln();
        let lj = (self.iterations_to.max(1) as f64).ln();
        let iterations = (li + (lj - li) * e).exp().round().max(1.0) as u32;
        (view, iterations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * a.abs().max(b.abs()).max(1e-300)
    }

    #[test]
    fn starts_and_ends_in_place_and_zooms_out_between() {
        let a = View { cx: -0.75, cy: 0.1, scale: 1e-6 };
        let b = View { cx: 0.28, cy: 0.01, scale: 2e-3 };
        let f = Flight::new(a, b, 2000, 500, 1.5, 1.0);
        let (v0, i0) = f.at(0.0);
        assert!(close(v0.cx, a.cx, 1e-9) && close(v0.scale, a.scale, 1e-6), "{v0:?}");
        assert_eq!(i0, 2000);
        let (mid, _) = f.at(0.5);
        assert!(mid.scale > b.scale, "should zoom out on the way: {mid:?}");
        let (v1, i1) = f.at(1.0);
        assert_eq!(v1, b);
        assert_eq!(i1, 500);
        // Just before landing it is already almost there.
        let (near, _) = f.at(0.999);
        assert!((near.cx - b.cx).abs() < 1e-3 && close(near.scale, b.scale, 0.05), "{near:?}");
    }

    #[test]
    fn deep_start_is_accurate() {
        let a = View { cx: -0.743_643_887_037_158_7, cy: 0.131_825_904_205_311_97, scale: 1e-11 };
        let f = Flight::new(a, View::default(), 5000, 500, 1.5, 1.0);
        let (v, _) = f.at(0.001);
        assert!((v.cx - a.cx).abs() < 1e-10 && v.scale < 1e-9, "{v:?}");
        assert!(f.duration > 1.0 && f.duration <= 10.0);
    }

    #[test]
    fn pure_zoom() {
        let a = View { cx: -0.5, cy: 0.0, scale: 1.0 };
        let b = View { cx: -0.5, cy: 0.0, scale: 1e-4 };
        let f = Flight::new(a, b, 100, 100, 1.5, 1.0);
        let (m, _) = f.at(0.5);
        assert!(close(m.scale, (1e-4f64).powf(ease(0.5)), 1e-9));
        assert_eq!(f.at(1.0).0, b);
    }
}
