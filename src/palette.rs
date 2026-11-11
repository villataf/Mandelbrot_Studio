//! Colour palettes, presets and turning iteration counts into pixels.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Entries in a colour lookup table.
pub const LUT_SIZE: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorStop {
    /// Position along the gradient, 0..1.
    pub pos: f32,
    pub color: [u8; 3],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Palette {
    pub name: String,
    pub stops: Vec<ColorStop>,
    /// Colour of the points that belong to the set.
    pub interior: [u8; 3],
}

/// How iteration counts are spread over the palette.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Mapping {
    Linear,
    Sqrt,
    Log,
}

impl Mapping {
    pub const ALL: [Mapping; 3] = [Mapping::Linear, Mapping::Sqrt, Mapping::Log];

    pub fn label(self) -> &'static str {
        match self {
            Mapping::Linear => "Lineare",
            Mapping::Sqrt => "Radice quadrata",
            Mapping::Log => "Logaritmica",
        }
    }

    #[inline]
    fn apply(self, mu: f32) -> f32 {
        match self {
            Mapping::Linear => mu / 64.0,
            Mapping::Sqrt => mu.sqrt() / 4.0,
            Mapping::Log => (1.0 + mu).ln() / 1.5,
        }
    }
}

/// Everything that decides the colours of an image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ColorSettings {
    pub palette: Palette,
    /// How many times the palette repeats (higher = more bands).
    pub density: f32,
    /// Shift along the palette, 0..1.
    pub offset: f32,
    pub mapping: Mapping,
}

impl Default for ColorSettings {
    fn default() -> Self {
        ColorSettings {
            palette: presets()[0].clone(),
            density: 1.0,
            offset: 0.0,
            mapping: Mapping::Sqrt,
        }
    }
}

fn hex(c: u32) -> [u8; 3] {
    [(c >> 16) as u8, (c >> 8) as u8, c as u8]
}

fn preset(name: &str, stops: &[(f32, u32)], interior: u32) -> Palette {
    Palette {
        name: name.to_string(),
        stops: stops
            .iter()
            .map(|&(pos, c)| ColorStop { pos, color: hex(c) })
            .collect(),
        interior: hex(interior),
    }
}

/// The built-in palettes.
pub fn presets() -> Vec<Palette> {
    vec![
        preset(
            "Classica",
            &[
                (0.0, 0x000764),
                (0.16, 0x206BCB),
                (0.42, 0xEDFFFF),
                (0.6425, 0xFFAA00),
                (0.8575, 0x000200),
            ],
            0x000000,
        ),
        preset(
            "Fuoco",
            &[
                (0.0, 0x000000),
                (0.25, 0x7A0000),
                (0.5, 0xFF4500),
                (0.75, 0xFFD700),
                (0.9, 0xFFFFE0),
            ],
            0x000000,
        ),
        preset(
            "Ghiaccio",
            &[
                (0.0, 0x001020),
                (0.3, 0x0A4A8A),
                (0.6, 0x8FD3FF),
                (0.8, 0xFFFFFF),
                (0.95, 0x3060A0),
            ],
            0x000000,
        ),
        preset(
            "Arcobaleno",
            &[
                (0.0, 0xFF0000),
                (1.0 / 6.0, 0xFFFF00),
                (2.0 / 6.0, 0x00FF00),
                (3.0 / 6.0, 0x00FFFF),
                (4.0 / 6.0, 0x0000FF),
                (5.0 / 6.0, 0xFF00FF),
            ],
            0x000000,
        ),
        preset(
            "Scala di grigi",
            &[(0.0, 0x000000), (0.5, 0xFFFFFF)],
            0x000000,
        ),
        preset(
            "Oceano",
            &[
                (0.0, 0x001018),
                (0.3, 0x005F73),
                (0.55, 0x0A9396),
                (0.75, 0x94D2BD),
                (0.9, 0xE9D8A6),
            ],
            0x000000,
        ),
        preset(
            "Tramonto",
            &[
                (0.0, 0x1A0533),
                (0.25, 0x6B1D5C),
                (0.5, 0xE84A5F),
                (0.7, 0xFF847C),
                (0.85, 0xFECEA8),
            ],
            0x0B0014,
        ),
        preset(
            "Elettrico",
            &[
                (0.0, 0x000000),
                (0.2, 0x1B00FF),
                (0.4, 0x00E1FF),
                (0.6, 0xFFFFFF),
                (0.8, 0xFF00C8),
            ],
            0x000000,
        ),
        preset(
            "Foresta",
            &[
                (0.0, 0x0B1A0B),
                (0.3, 0x2D6A2D),
                (0.6, 0xA7C957),
                (0.8, 0xF2E8CF),
                (0.92, 0x6A4C2D),
            ],
            0x000000,
        ),
        preset(
            "Oro",
            &[
                (0.0, 0x1A1000),
                (0.35, 0x8A5A00),
                (0.6, 0xFFD34D),
                (0.8, 0xFFF6D0),
                (0.92, 0x5C3A00),
            ],
            0x000000,
        ),
    ]
}

impl Palette {
    /// Stops sorted by position (a palette always has at least one).
    fn sorted_stops(&self) -> Vec<ColorStop> {
        let mut s: Vec<ColorStop> = self
            .stops
            .iter()
            .map(|s| ColorStop {
                pos: s.pos.clamp(0.0, 1.0),
                color: s.color,
            })
            .collect();
        if s.is_empty() {
            s.push(ColorStop {
                pos: 0.0,
                color: [255, 255, 255],
            });
        }
        s.sort_by(|a, b| a.pos.total_cmp(&b.pos));
        s
    }

    /// Colour at `t` (0..1); the gradient wraps around from the last stop to the first.
    pub fn sample_with(stops: &[ColorStop], t: f32) -> [f32; 3] {
        let n = stops.len();
        if n == 1 {
            return stops[0].color.map(|c| c as f32);
        }
        let t = t.rem_euclid(1.0);
        // Find the pair of stops around t, wrapping across 1.0.
        let (a, b, pa, pb) = if t < stops[0].pos {
            (stops[n - 1], stops[0], stops[n - 1].pos - 1.0, stops[0].pos)
        } else if t >= stops[n - 1].pos {
            (stops[n - 1], stops[0], stops[n - 1].pos, stops[0].pos + 1.0)
        } else {
            let i = stops.iter().rposition(|s| s.pos <= t).unwrap_or(0);
            let j = (i + 1).min(n - 1);
            (stops[i], stops[j], stops[i].pos, stops[j].pos)
        };
        let span = pb - pa;
        let f = if span > 1e-6 {
            ((t - pa) / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut out = [0.0; 3];
        for k in 0..3 {
            out[k] = a.color[k] as f32 + (b.color[k] as f32 - a.color[k] as f32) * f;
        }
        out
    }

    pub fn sample(&self, t: f32) -> [f32; 3] {
        Self::sample_with(&self.sorted_stops(), t)
    }

    /// Spread the stops evenly.
    pub fn distribute_evenly(&mut self) {
        let n = self.stops.len().max(1) as f32;
        self.stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
        for (i, s) in self.stops.iter_mut().enumerate() {
            s.pos = i as f32 / n;
        }
    }

    /// Reverse the order of the colours.
    pub fn reverse(&mut self) {
        for s in &mut self.stops {
            s.pos = (1.0 - s.pos).clamp(0.0, 1.0);
        }
        self.stops.sort_by(|a, b| a.pos.total_cmp(&b.pos));
    }
}

/// A precomputed colour table for fast colouring.
#[derive(Clone, Debug)]
pub struct Lut {
    pub colors: Vec<[f32; 3]>,
    pub interior: [f32; 3],
}

impl Lut {
    pub fn new(palette: &Palette) -> Lut {
        let stops = palette.sorted_stops();
        let colors = (0..LUT_SIZE)
            .map(|i| Palette::sample_with(&stops, i as f32 / LUT_SIZE as f32))
            .collect();
        Lut {
            colors,
            interior: palette.interior.map(|c| c as f32),
        }
    }

    /// Mix two tables: `t` = 0 gives `a`, 1 gives `b`.
    pub fn blend(a: &Lut, b: &Lut, t: f32) -> Lut {
        let t = t.clamp(0.0, 1.0);
        let mix = |x: [f32; 3], y: [f32; 3]| {
            [
                x[0] + (y[0] - x[0]) * t,
                x[1] + (y[1] - x[1]) * t,
                x[2] + (y[2] - x[2]) * t,
            ]
        };
        Lut {
            colors: a
                .colors
                .iter()
                .zip(&b.colors)
                .map(|(x, y)| mix(*x, *y))
                .collect(),
            interior: mix(a.interior, b.interior),
        }
    }

    #[inline]
    fn lookup(&self, t: f32) -> [f32; 3] {
        let f = t.rem_euclid(1.0) * LUT_SIZE as f32;
        let i = (f as usize).min(LUT_SIZE - 1);
        let j = (i + 1) % LUT_SIZE;
        let w = f - i as f32;
        let a = self.colors[i];
        let b = self.colors[j];
        [
            a[0] + (b[0] - a[0]) * w,
            a[1] + (b[1] - a[1]) * w,
            a[2] + (b[2] - a[2]) * w,
        ]
    }
}

/// The parts of [`ColorSettings`] needed once the table is built.
#[derive(Clone, Copy, Debug)]
pub struct ColorMap {
    pub density: f32,
    pub offset: f32,
    pub mapping: Mapping,
}

impl From<&ColorSettings> for ColorMap {
    fn from(c: &ColorSettings) -> Self {
        ColorMap {
            density: c.density,
            offset: c.offset,
            mapping: c.mapping,
        }
    }
}

#[inline]
fn color_of(v: f32, lut: &Lut, map: ColorMap) -> [f32; 3] {
    if v < 0.0 || !v.is_finite() {
        lut.interior
    } else {
        lut.lookup(map.mapping.apply(v) * map.density + map.offset)
    }
}

/// Colour iteration values into pixels with `channels` bytes each (3 = RGB, 4 = RGBA).
pub fn colorize(data: &[f32], lut: &Lut, map: ColorMap, channels: usize, out: &mut [u8]) {
    debug_assert_eq!(out.len(), data.len() * channels);
    out.par_chunks_mut(channels * 4096)
        .zip(data.par_chunks(4096))
        .for_each(|(o, d)| {
            for (px, &v) in o.chunks_exact_mut(channels).zip(d) {
                let c = color_of(v, lut, map);
                px[0] = c[0].round().clamp(0.0, 255.0) as u8;
                px[1] = c[1].round().clamp(0.0, 255.0) as u8;
                px[2] = c[2].round().clamp(0.0, 255.0) as u8;
                if channels == 4 {
                    px[3] = 255;
                }
            }
        });
}

/// Colour with `factor`×`factor` supersampling: `data` is `w*factor` × `h*factor`, output is RGB `w`×`h`.
pub fn colorize_supersampled(
    data: &[f32],
    w: usize,
    h: usize,
    factor: usize,
    lut: &Lut,
    map: ColorMap,
) -> Vec<u8> {
    if factor <= 1 {
        let mut out = vec![0u8; w * h * 3];
        colorize(data, lut, map, 3, &mut out);
        return out;
    }
    let big_w = w * factor;
    let inv = 1.0 / (factor * factor) as f32;
    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = [0.0f32; 3];
            for sy in 0..factor {
                let base = (y * factor + sy) * big_w + x * factor;
                for sx in 0..factor {
                    let c = color_of(data[base + sx], lut, map);
                    acc[0] += c[0];
                    acc[1] += c[1];
                    acc[2] += c[2];
                }
            }
            for k in 0..3 {
                row[x * 3 + k] = (acc[k] * inv).round().clamp(0.0, 255.0) as u8;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_wraps_and_hits_stops() {
        let p = preset("t", &[(0.0, 0x000000), (0.5, 0xFFFFFF)], 0);
        assert_eq!(p.sample(0.0), [0.0, 0.0, 0.0]);
        assert_eq!(p.sample(0.5), [255.0, 255.0, 255.0]);
        let q = p.sample(0.75);
        assert!((q[0] - 127.5).abs() < 0.01);
        assert_eq!(p.sample(1.0), p.sample(0.0));
    }

    #[test]
    fn interior_uses_interior_colour() {
        let mut p = presets()[0].clone();
        p.interior = [1, 2, 3];
        let lut = Lut::new(&p);
        let mut out = [0u8; 4];
        colorize(
            &[-1.0],
            &lut,
            ColorMap::from(&ColorSettings::default()),
            4,
            &mut out,
        );
        assert_eq!(out, [1, 2, 3, 255]);
    }

    #[test]
    fn blend_halfway() {
        let a = Lut::new(&preset("a", &[(0.0, 0x000000)], 0));
        let b = Lut::new(&preset("b", &[(0.0, 0xFFFFFF)], 0xFFFFFF));
        let m = Lut::blend(&a, &b, 0.5);
        assert!((m.colors[10][0] - 127.5).abs() < 0.01);
        assert!((m.interior[2] - 127.5).abs() < 0.01);
    }
}
