//! Suggesting how many iterations a place needs.

use rayon::prelude::*;

use crate::compute::{cpu, Cancel, UNKNOWN};
use crate::view::{View, BASE_SCALE};

/// A suggested iteration count and how it was found.
#[derive(Clone, Debug, PartialEq)]
pub struct Recommendation {
    pub iterations: u32,
    pub reason: String,
}

/// A rule of thumb from the zoom depth alone.
pub fn heuristic(view: &View) -> u32 {
    let depth = (BASE_SCALE / view.scale).log10().max(0.0);
    round_nice(100.0 + 60.0 * depth.powf(1.6))
}

/// Round to two significant figures.
pub fn round_nice(v: f64) -> u32 {
    if v < 100.0 {
        return v.round().max(16.0) as u32;
    }
    let mag = 10f64.powf(v.log10().floor() - 1.0);
    ((v / mag).round() * mag) as u32
}

/// Look at a small sample of the place with a high limit and see how far the slowest
/// escaping points go. Returns `None` when cancelled.
pub fn analyse(view: &View, aspect: f64, cancel: &Cancel<'_>) -> Option<Recommendation> {
    let rule = heuristic(view);
    let cap = (rule as u64 * 20).clamp(5_000, 200_000) as u32;
    let w = 96u32;
    let h = ((w as f64 / aspect.max(0.2)).round() as u32).clamp(24, 96);
    let ps = 2.0 * view.scale / h as f64;
    let values: Vec<f32> = (0..h)
        .into_par_iter()
        .flat_map_iter(|row| {
            let cy = view.cy - (row as f64 + 0.5 - h as f64 * 0.5) * ps;
            (0..w).map(move |col| {
                if cancel() {
                    return UNKNOWN;
                }
                let cx = view.cx + (col as f64 + 0.5 - w as f64 * 0.5) * ps;
                cpu::iterate(cx, cy, cap)
            })
        })
        .collect();
    if cancel() {
        return None;
    }
    let mut escaped: Vec<f32> = values.iter().copied().filter(|v| *v >= 0.0).collect();
    let unknown = values.iter().filter(|v| **v == UNKNOWN).count();
    if escaped.is_empty() {
        return Some(Recommendation {
            iterations: rule,
            reason: "zona interna all'insieme: valore stimato dallo zoom".into(),
        });
    }
    escaped.sort_by(|a, b| a.total_cmp(b));
    let idx = ((escaped.len() as f64 * 0.995) as usize).min(escaped.len() - 1);
    let slow = escaped[idx] as f64;
    let mut suggested = (slow * 1.25 + 64.0).max(100.0);
    if unknown > values.len() / 50 && suggested < rule as f64 {
        // Many points still undecided at the analysis limit: lean on the rule of thumb.
        suggested = suggested.max(rule as f64);
    }
    let iterations = round_nice(suggested).min(cap);
    Some(Recommendation {
        iterations,
        reason: format!(
            "analisi di {} punti: il 99,5% di quelli esterni esce entro {} iterazioni",
            values.len(),
            slow.ceil() as u64
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deeper_needs_more() {
        let shallow = View::default();
        let deep = View {
            cx: -0.743_643_887_037_158_7,
            cy: 0.131_825_904_205_311_97,
            scale: 1e-8,
        };
        let never = || false;
        let a = analyse(&shallow, 1.5, &never).unwrap();
        let b = analyse(&deep, 1.5, &never).unwrap();
        assert!(b.iterations > a.iterations, "{a:?} {b:?}");
        assert!(heuristic(&deep) > heuristic(&shallow));
    }

    #[test]
    fn rounding() {
        assert_eq!(round_nice(1234.0), 1200);
        assert_eq!(round_nice(56.0), 56);
    }
}
