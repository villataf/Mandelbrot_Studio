//! Escape-time calculation on the processor, in 64-bit floating point, on every core.

use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

use super::{Cancel, FractalParams, INTERIOR, UNKNOWN};

/// Squared escape radius: a large radius gives smooth colour bands.
pub const BAILOUT2: f64 = 65536.0;

/// Iterate one point. Returns the smooth iteration count, [`INTERIOR`] for points proven
/// inside the set, or [`UNKNOWN`] when the iteration limit is reached first.
#[inline]
pub fn iterate(cx: f64, cy: f64, max_iter: u32) -> f32 {
    // Main cardioid and period-2 bulb: inside for sure.
    let xq = cx - 0.25;
    let q = xq * xq + cy * cy;
    if q * (q + xq) <= 0.25 * cy * cy {
        return INTERIOR;
    }
    let xb = cx + 1.0;
    if xb * xb + cy * cy <= 0.0625 {
        return INTERIOR;
    }

    let (mut x, mut y, mut x2, mut y2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    // Brent-style cycle detection: an orbit that comes back to a saved point is periodic.
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    let mut next_save = 8u32;
    let mut i = 0u32;
    while i < max_iter {
        y = 2.0 * x * y + cy;
        x = x2 - y2 + cx;
        x2 = x * x;
        y2 = y * y;
        i += 1;
        let m = x2 + y2;
        if m > BAILOUT2 {
            let mu = i as f64 + 1.0 - (0.5 * m.ln()).log2();
            return mu.max(0.0) as f32;
        }
        if (x - sx).abs() < 1e-15 && (y - sy).abs() < 1e-15 {
            return INTERIOR;
        }
        if i == next_save {
            sx = x;
            sy = y;
            next_save = next_save.saturating_mul(2);
        }
    }
    UNKNOWN
}

/// Calculate a `w`×`h` image. Returns `None` when cancelled.
pub fn render(
    p: &FractalParams,
    w: u32,
    h: u32,
    cancel: &Cancel<'_>,
    progress: Option<&AtomicU32>,
) -> Option<Vec<f32>> {
    let (wu, hu) = (w as usize, h as usize);
    let mut data = vec![0f32; wu * hu];
    let ps = p.view.pixel_size(h);
    let half_w = w as f64 * 0.5;
    let half_h = h as f64 * 0.5;
    data.par_chunks_mut(wu).enumerate().for_each(|(row, out)| {
        if cancel() {
            return;
        }
        let cy = p.view.cy - (row as f64 + 0.5 - half_h) * ps;
        for (col, v) in out.iter_mut().enumerate() {
            let cx = p.view.cx + (col as f64 + 0.5 - half_w) * ps;
            *v = iterate(cx, cy, p.max_iter);
        }
        if let Some(pr) = progress {
            pr.fetch_add(1, Ordering::Relaxed);
        }
    });
    if cancel() {
        None
    } else {
        Some(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_points() {
        assert_eq!(iterate(0.0, 0.0, 1000), INTERIOR);
        assert_eq!(iterate(-1.0, 0.0, 1000), INTERIOR);
        assert!(iterate(1.0, 1.0, 1000) >= 0.0);
        assert!(iterate(0.5, 0.0, 1000) > 0.0);
        // Period-3 bulb point: found interior by cycle detection or by running out.
        assert!(iterate(-0.122, 0.745, 5000) < 0.0);
    }

    #[test]
    fn smooth_values_are_continuous() {
        let a = iterate(0.26, 0.0, 5000);
        let b = iterate(0.26 + 1e-7, 0.0, 5000);
        assert!(a > 0.0 && (a - b).abs() < 0.1, "{a} {b}");
    }
}
