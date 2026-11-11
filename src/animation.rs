//! Keyframes and the smooth path between them.

use serde::{Deserialize, Serialize};

use crate::palette::{ColorMap, ColorSettings, Lut, Mapping};
use crate::video::VideoSettings;
use crate::view::View;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Easing {
    Linear,
    Smooth,
    EaseIn,
    EaseOut,
}

impl Easing {
    pub const ALL: [Easing; 4] = [
        Easing::Smooth,
        Easing::Linear,
        Easing::EaseIn,
        Easing::EaseOut,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Easing::Linear => "Costante",
            Easing::Smooth => "Morbida",
            Easing::EaseIn => "Accelera",
            Easing::EaseOut => "Rallenta",
        }
    }

    pub fn apply(self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Easing::Linear => t,
            Easing::Smooth => t * t * (3.0 - 2.0 * t),
            Easing::EaseIn => t * t,
            Easing::EaseOut => 1.0 - (1.0 - t) * (1.0 - t),
        }
    }
}

/// One stop of the animation: where to be, how many iterations and which colours.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Keyframe {
    pub name: String,
    pub view: View,
    pub iterations: u32,
    pub color: ColorSettings,
    /// Seconds to travel from this keyframe to the next one.
    pub duration: f32,
    /// How the movement to the next keyframe speeds up and slows down.
    pub easing: Easing,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Animation {
    pub keyframes: Vec<Keyframe>,
    /// Extra palette movement in cycles per second (colour cycling).
    pub color_cycle_speed: f32,
}

/// Everything needed to draw one moment of the animation.
#[derive(Clone, Debug)]
pub struct FrameSpec {
    pub view: View,
    pub iterations: u32,
    pub from: usize,
    pub to: usize,
    /// Blend between the two keyframes' palettes, 0..1.
    pub color_mix: f32,
    pub map: ColorMap,
}

impl FrameSpec {
    /// The colour table for this moment, given one table per keyframe.
    pub fn lut(&self, luts: &[Lut]) -> Lut {
        if self.from == self.to || self.color_mix <= 0.0 {
            luts[self.from].clone()
        } else {
            Lut::blend(&luts[self.from], &luts[self.to], self.color_mix)
        }
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

impl Animation {
    /// Length of the whole animation in seconds.
    pub fn total_duration(&self) -> f64 {
        let n = self.keyframes.len();
        if n < 2 {
            return 0.0;
        }
        self.keyframes[..n - 1]
            .iter()
            .map(|k| k.duration.max(0.0) as f64)
            .sum()
    }

    /// When each keyframe is reached, in seconds.
    pub fn keyframe_times(&self) -> Vec<f64> {
        let mut t = 0.0;
        self.keyframes
            .iter()
            .map(|k| {
                let at = t;
                t += k.duration.max(0.0) as f64;
                at
            })
            .collect()
    }

    pub fn frame_count(&self, fps: u32) -> u64 {
        (self.total_duration() * fps as f64).round() as u64 + 1
    }

    pub fn luts(&self) -> Vec<Lut> {
        self.keyframes
            .iter()
            .map(|k| Lut::new(&k.color.palette))
            .collect()
    }

    /// The animation at `time` seconds.
    pub fn sample(&self, time: f64) -> Option<FrameSpec> {
        let n = self.keyframes.len();
        if n == 0 {
            return None;
        }
        let cycle = self.color_cycle_speed as f64 * time;
        let single = |i: usize| {
            let k = &self.keyframes[i];
            let mut map = ColorMap::from(&k.color);
            map.offset = (map.offset as f64 + cycle).rem_euclid(1.0) as f32;
            FrameSpec {
                view: k.view,
                iterations: k.iterations,
                from: i,
                to: i,
                color_mix: 0.0,
                map,
            }
        };
        if n == 1 || time <= 0.0 {
            return Some(single(0));
        }
        let times = self.keyframe_times();
        let i = match times.iter().rposition(|t| *t <= time) {
            Some(i) if i < n - 1 => i,
            _ => return Some(single(n - 1)),
        };
        let a = &self.keyframes[i];
        let b = &self.keyframes[i + 1];
        let span = a.duration.max(0.0) as f64;
        let u = if span > 0.0 {
            (time - times[i]) / span
        } else {
            1.0
        };
        let e = a.easing.apply(u);

        // Zoom moves at a steady pace in log scale.
        let (la, lb) = (a.view.scale.ln(), b.view.scale.ln());
        let scale = lerp(la, lb, e).exp();
        // The centre moves so the destination approaches steadily on screen: when zooming
        // in, the target point stays in view instead of drifting.
        let w = if (a.view.scale - b.view.scale).abs() > 1e-9 * a.view.scale.max(b.view.scale) {
            ((a.view.scale - scale) / (a.view.scale - b.view.scale)).clamp(0.0, 1.0)
        } else {
            e
        };
        let view = View {
            cx: lerp(a.view.cx, b.view.cx, w),
            cy: lerp(a.view.cy, b.view.cy, w),
            scale,
        };
        let iterations = lerp(
            (a.iterations.max(1) as f64).ln(),
            (b.iterations.max(1) as f64).ln(),
            e,
        )
        .exp()
        .round()
        .max(1.0) as u32;
        let density = (lerp(
            (a.color.density.max(1e-4) as f64).ln(),
            (b.color.density.max(1e-4) as f64).ln(),
            e,
        ))
        .exp();
        // Offsets wrap around: take the short way.
        let mut d = b.color.offset as f64 - a.color.offset as f64;
        d -= d.round();
        let offset = (a.color.offset as f64 + d * e + cycle).rem_euclid(1.0);
        let mapping: Mapping = if e < 0.5 {
            a.color.mapping
        } else {
            b.color.mapping
        };
        Some(FrameSpec {
            view,
            iterations,
            from: i,
            to: i + 1,
            color_mix: e as f32,
            map: ColorMap {
                density: density as f32,
                offset: offset as f32,
                mapping,
            },
        })
    }
}

/// An animation saved to disk with its video settings.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationProject {
    pub kind: String,
    pub version: u32,
    pub animation: Animation,
    pub video: VideoSettings,
}

impl AnimationProject {
    pub const KIND: &'static str = "mandelbrot-studio-animazione";

    pub fn new(animation: Animation, video: VideoSettings) -> Self {
        AnimationProject {
            kind: Self::KIND.into(),
            version: 1,
            animation,
            video,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kf(cx: f64, scale: f64, iters: u32, duration: f32) -> Keyframe {
        Keyframe {
            name: String::new(),
            view: View { cx, cy: 0.0, scale },
            iterations: iters,
            color: ColorSettings::default(),
            duration,
            easing: Easing::Linear,
        }
    }

    #[test]
    fn path_hits_keyframes() {
        let anim = Animation {
            keyframes: vec![
                kf(-0.5, 1.0, 100, 2.0),
                kf(-0.75, 1e-4, 1000, 3.0),
                kf(-0.7, 1e-6, 2000, 1.0),
            ],
            color_cycle_speed: 0.0,
        };
        assert_eq!(anim.total_duration(), 5.0);
        assert_eq!(anim.frame_count(30), 151);
        let s0 = anim.sample(0.0).unwrap();
        assert_eq!(s0.view, anim.keyframes[0].view);
        let s1 = anim.sample(2.0).unwrap();
        assert!((s1.view.scale / 1e-4 - 1.0).abs() < 1e-9);
        assert_eq!(s1.iterations, 1000);
        let end = anim.sample(10.0).unwrap();
        assert_eq!(end.view, anim.keyframes[2].view);
        // Half-way in time is half-way in log scale.
        let mid = anim.sample(1.0).unwrap();
        assert!((mid.view.scale.ln() - (1e-4f64).ln() / 2.0).abs() < 1e-9);
        assert!(mid.color_mix > 0.49 && mid.color_mix < 0.51);
    }

    #[test]
    fn colour_cycling_moves_offset() {
        let anim = Animation {
            keyframes: vec![kf(0.0, 1.0, 100, 1.0)],
            color_cycle_speed: 0.25,
        };
        let s = anim.sample(1.0).unwrap();
        assert!((s.map.offset - 0.25).abs() < 1e-6);
    }
}
