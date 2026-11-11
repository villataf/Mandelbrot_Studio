//! Perturbation: one reference orbit computed precisely on the processor, so the graphics
//! card only follows each pixel's small difference from it, in fast single precision.

use rayon::prelude::*;

use super::cpu::{self, BAILOUT2};
use super::{Cancel, INTERIOR, UNKNOWN};
use crate::view::View;

/// The point (as an offset in pixels from the centre of a `w`×`h` image) that keeps
/// iterating longest among a coarse grid: pixels that outlast the reference lose precision,
/// so the longest-lived point makes the best reference.
pub fn reference_point(view: &View, w: u32, h: u32, max_iter: u32, cancel: &Cancel<'_>) -> (f64, f64) {
    let ps = view.pixel_size(h);
    let (gx, gy) = (24u32, 14u32);
    let mut points: Vec<(f64, f64)> = vec![(0.5 * (w % 2) as f64, 0.5 * (h % 2) as f64)];
    for j in 0..gy {
        for i in 0..gx {
            let px = ((i as f64 + 0.5) / gx as f64 * w as f64).floor();
            let py = ((j as f64 + 0.5) / gy as f64 * h as f64).floor();
            points.push((px + 0.5 - w as f64 * 0.5, py + 0.5 - h as f64 * 0.5));
        }
    }
    let scored: Vec<(f64, f64, f64)> = points
        .par_iter()
        .map(|&(dx, dy)| {
            if cancel() {
                return (dx, dy, -1.0);
            }
            let v = cpu::iterate(view.cx + dx * ps, view.cy - dy * ps, max_iter);
            let score = if v == INTERIOR {
                f64::INFINITY
            } else if v == UNKNOWN {
                max_iter as f64 + 1.0
            } else {
                v as f64
            };
            (dx, dy, score)
        })
        .collect();
    scored
        .into_iter()
        .max_by(|a, b| {
            a.2.total_cmp(&b.2)
                // On a tie, the point nearest the centre.
                .then((b.0 * b.0 + b.1 * b.1).total_cmp(&(a.0 * a.0 + a.1 * a.1)))
        })
        .map(|(dx, dy, _)| (dx, dy))
        .unwrap_or((0.0, 0.0))
}

/// The orbit Z₀ = 0, Zₙ₊₁ = Zₙ² + C in 64-bit precision, rounded to 32 bits for the card. It
/// stops after the first value that escapes, after `max_iter` steps or at `max_len` values,
/// and always holds at least two values.
pub fn reference_orbit(cx: f64, cy: f64, max_iter: u32, max_len: usize) -> Vec<[f32; 2]> {
    let mut orbit = Vec::with_capacity((max_iter as usize + 1).min(max_len).min(1 << 22));
    orbit.push([0.0, 0.0]);
    let (mut x, mut y) = (0.0f64, 0.0f64);
    for _ in 0..max_iter.max(1) {
        let nx = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = nx;
        orbit.push([x as f32, y as f32]);
        if x * x + y * y > BAILOUT2 || orbit.len() >= max_len.max(2) {
            break;
        }
    }
    orbit
}

/// The same calculation the graphics card does, on the processor (used by the tests).
#[cfg(test)]
pub fn iterate_perturbed(orbit: &[[f32; 2]], dc: [f32; 2], max_iter: u32) -> f32 {
    let last = orbit.len() - 1;
    let (mut dx, mut dy) = (0.0f32, 0.0f32);
    let mut m = 0usize;
    for n in 1..=max_iter {
        let [zx, zy] = orbit[m];
        let ndx = 2.0 * (zx * dx - zy * dy) + (dx * dx - dy * dy) + dc[0];
        dy = 2.0 * (zx * dy + zy * dx) + 2.0 * dx * dy + dc[1];
        dx = ndx;
        m += 1;
        let (x, y) = (orbit[m][0] + dx, orbit[m][1] + dy);
        let mz = x * x + y * y;
        if mz > BAILOUT2 as f32 {
            return (n as f32 + 1.0 - (0.5 * mz.ln()).log2()).max(0.0);
        }
        if mz < dx * dx + dy * dy || m >= last {
            dx = x;
            dy = y;
            m = 0;
        }
    }
    UNKNOWN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation_matches_direct_calculation_when_zoomed_in() {
        let view = View { cx: -0.743_643_887_037_158_7, cy: 0.131_825_904_205_311_97, scale: 1.0e-9 };
        let (w, h) = (64u32, 36u32);
        let never = || false;
        let (rx, ry) = reference_point(&view, w, h, 5000, &never);
        let ps = view.pixel_size(h);
        let orbit = reference_orbit(view.cx + rx * ps, view.cy - ry * ps, 5000, usize::MAX);
        let mut same = 0;
        for row in 0..h {
            for col in 0..w {
                let dx = col as f64 + 0.5 - w as f64 * 0.5;
                let dy = row as f64 + 0.5 - h as f64 * 0.5;
                let direct = cpu::iterate(view.cx + dx * ps, view.cy - dy * ps, 5000);
                let dc = [((dx - rx) * ps) as f32, (-(dy - ry) * ps) as f32];
                let pert = iterate_perturbed(&orbit, dc, 5000);
                if (direct < 0.0 && pert < 0.0) || (direct - pert).abs() < 0.5 {
                    same += 1;
                }
            }
        }
        let share = same as f64 / (w * h) as f64;
        assert!(share > 0.85, "only {share} of pixels match");
    }
}
