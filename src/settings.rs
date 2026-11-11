//! Settings, favourite places and custom palettes kept between sessions, and the
//! coordinate files people can save and load.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::compute::gpu::PrecisionChoice;
use crate::compute::Engine;
use crate::export::ImageOptions;
use crate::palette::{ColorSettings, Palette};
use crate::video::VideoSettings;
use crate::view::{f64_exact, parse_number, View};

/// A saved place: centre, zoom, iterations and (optionally) colours.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Location {
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub name: String,
    /// Kept as text so every digit survives.
    pub center_re: String,
    pub center_im: String,
    /// Half of the visible height in the complex plane.
    pub scale: String,
    /// For people reading the file; `scale` is what is used.
    #[serde(default)]
    pub zoom: String,
    pub iterations: u32,
    #[serde(default)]
    pub color: Option<ColorSettings>,
}

impl Location {
    pub const KIND: &'static str = "mandelbrot-studio-coordinate";

    pub fn new(name: &str, view: &View, iterations: u32, color: Option<ColorSettings>) -> Location {
        Location {
            kind: Self::KIND.into(),
            name: name.into(),
            center_re: f64_exact(view.cx),
            center_im: f64_exact(view.cy),
            scale: f64_exact(view.scale),
            zoom: format!("{:e}", view.zoom()),
            iterations,
            color,
        }
    }

    pub fn view(&self) -> Option<View> {
        let mut v = View {
            cx: parse_number(&self.center_re)?,
            cy: parse_number(&self.center_im)?,
            scale: parse_number(&self.scale)?,
        };
        if v.scale <= 0.0 {
            return None;
        }
        v.clamp();
        Some(v)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(path, text)
            .map_err(|e| format!("Impossibile salvare {}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Location, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("Impossibile leggere {}: {e}", path.display()))?;
        let loc: Location = serde_json::from_str(&text)
            .map_err(|e| format!("File di coordinate non valido: {e}"))?;
        loc.view().ok_or("Le coordinate nel file non sono valide")?;
        Ok(loc)
    }

    /// One line of text for the clipboard.
    pub fn to_clipboard_text(&self) -> String {
        format!(
            "re={} im={} scala={} iterazioni={}",
            self.center_re, self.center_im, self.scale, self.iterations
        )
    }

    /// Read text copied with [`Self::to_clipboard_text`] or a pasted coordinate file.
    pub fn from_clipboard_text(text: &str) -> Option<Location> {
        if let Ok(loc) = serde_json::from_str::<Location>(text) {
            return loc.view().is_some().then_some(loc);
        }
        let mut re = None;
        let mut im = None;
        let mut scale = None;
        let mut zoom = None;
        let mut iterations = None;
        for part in text.split(|c: char| c.is_whitespace() || c == ';') {
            if let Some((k, v)) = part.split_once('=').or_else(|| part.split_once(':')) {
                match k.trim().to_ascii_lowercase().as_str() {
                    "re" | "x" | "reale" => re = Some(v.to_string()),
                    "im" | "y" | "immaginaria" => im = Some(v.to_string()),
                    "scala" | "scale" => scale = Some(v.to_string()),
                    "zoom" => zoom = parse_number(v),
                    "iterazioni" | "iterations" | "iter" => iterations = v.trim().parse().ok(),
                    _ => {}
                }
            }
        }
        let scale = match (scale, zoom) {
            (Some(s), _) => s,
            (None, Some(z)) if z > 0.0 => f64_exact(crate::view::BASE_SCALE / z),
            _ => f64_exact(View::default().scale),
        };
        let loc = Location {
            kind: Self::KIND.into(),
            name: String::new(),
            center_re: re?,
            center_im: im?,
            scale,
            zoom: String::new(),
            iterations: iterations.unwrap_or(500),
            color: None,
        };
        loc.view().is_some().then_some(loc)
    }
}

/// Settings kept between sessions.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub engine: Engine,
    pub gpu_adapter: Option<String>,
    pub precision: PrecisionChoice,
    pub ffmpeg_path: String,
    pub auto_iterations: bool,
    pub iterations: u32,
    pub view: View,
    pub color: ColorSettings,
    pub favourites: Vec<Location>,
    pub image: ImageOptions,
    pub image_supersampling: u32,
    pub video: VideoSettings,
    pub last_folder: Option<PathBuf>,
    pub show_help_on_start: bool,
    /// Fly (zoom out, travel, zoom in) when jumping to another position.
    pub animate_moves: bool,
    /// Speed of those flights (1 = normal).
    pub flight_speed: f64,
    pub show_left_panel: bool,
    pub show_right_panel: bool,
    /// Where the program opens: the last position or the starting view.
    pub start_at: StartAt,
    /// Light, dark (night mode) or as the system.
    pub theme: Theme,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StartAt {
    LastPosition,
    DefaultPosition,
}

impl StartAt {
    pub fn label(self) -> &'static str {
        match self {
            StartAt::LastPosition => "Ultima posizione",
            StartAt::DefaultPosition => "Posizione iniziale",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    System,
    Light,
    Night,
}

impl Theme {
    pub fn label(self) -> &'static str {
        match self {
            Theme::System => "Come il sistema",
            Theme::Light => "Chiaro",
            Theme::Night => "Notte",
        }
    }
}

impl Default for AppSettings {
    fn default() -> Self {
        AppSettings {
            engine: Engine::Gpu,
            gpu_adapter: None,
            precision: PrecisionChoice::Auto,
            ffmpeg_path: String::new(),
            auto_iterations: false,
            iterations: 500,
            view: View::default(),
            color: ColorSettings::default(),
            favourites: vec![
                Location::new(
                    "Valle dei cavallucci marini",
                    &View {
                        cx: -0.743_643_887_037_158_7,
                        cy: 0.131_825_904_205_311_97,
                        scale: 2.0e-3,
                    },
                    1500,
                    None,
                ),
                Location::new(
                    "Valle degli elefanti",
                    &View {
                        cx: 0.282_5,
                        cy: 0.01,
                        scale: 0.02,
                    },
                    1000,
                    None,
                ),
                Location::new(
                    "Spirale tripla",
                    &View {
                        cx: -0.088,
                        cy: 0.654,
                        scale: 0.012,
                    },
                    1200,
                    None,
                ),
                Location::new(
                    "Mini-Mandelbrot",
                    &View {
                        cx: -1.754_877_666_246_693,
                        cy: 0.0,
                        scale: 0.025,
                    },
                    1000,
                    None,
                ),
            ],
            image: ImageOptions::default(),
            image_supersampling: 2,
            video: VideoSettings::default(),
            last_folder: None,
            show_help_on_start: true,
            animate_moves: true,
            flight_speed: 1.0,
            show_left_panel: true,
            show_right_panel: true,
            start_at: StartAt::LastPosition,
            theme: Theme::System,
        }
    }
}

/// The folder where settings and palettes live.
/// MANDELBROT_CONFIG_DIR, when set, uses another folder (for tests and portable copies).
pub fn config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("MANDELBROT_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    dirs::config_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("mandelbrot-studio")
}

fn read_json<T: serde::de::DeserializeOwned>(name: &str) -> Option<T> {
    let text = std::fs::read_to_string(config_dir().join(name)).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_json<T: Serialize>(name: &str, value: &T) -> Result<(), String> {
    let dir = config_dir();
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    let tmp = dir.join(format!("{name}.tmp"));
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dir.join(name)).map_err(|e| e.to_string())
}

impl AppSettings {
    pub fn load() -> AppSettings {
        read_json("impostazioni.json").unwrap_or_default()
    }

    pub fn save(&self) -> Result<(), String> {
        write_json("impostazioni.json", self)
    }
}

/// Palettes the person made and saved.
pub fn load_custom_palettes() -> Vec<Palette> {
    read_json("palette.json").unwrap_or_default()
}

pub fn save_custom_palettes(palettes: &[Palette]) -> Result<(), String> {
    write_json("palette.json", &palettes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn location_round_trip_is_exact() {
        let v = View {
            cx: -0.743_643_887_037_158_7,
            cy: 0.131_825_904_205_311_97,
            scale: 3.3e-11,
        };
        let loc = Location::new("prova", &v, 4321, None);
        let json = serde_json::to_string(&loc).unwrap();
        let back: Location = serde_json::from_str(&json).unwrap();
        assert_eq!(back.view().unwrap(), v);
        let pasted = Location::from_clipboard_text(&loc.to_clipboard_text()).unwrap();
        assert_eq!(pasted.view().unwrap(), v);
        assert_eq!(pasted.iterations, 4321);
    }

    #[test]
    fn pasted_zoom_text() {
        let loc = Location::from_clipboard_text("re=-0,75 im=0,1 zoom=1000").unwrap();
        let v = loc.view().unwrap();
        assert!((v.zoom() - 1000.0).abs() < 1e-6);
        assert_eq!(v.cx, -0.75);
    }
}
