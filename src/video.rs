//! Writing animations: raw frames, TIFF sequences, or H.264/H.265 through ffmpeg using the
//! graphics card's video encoder (NVIDIA NVENC, AMD AMF, Intel Quick Sync, Apple
//! VideoToolbox or VA-API on Linux).

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::export::{self, ImageFormat, ImageOptions, TiffCompression};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum VideoFormat {
    Raw,
    TiffSequence,
    H264,
    H265,
}

impl VideoFormat {
    pub const ALL: [VideoFormat; 4] = [
        VideoFormat::H264,
        VideoFormat::H265,
        VideoFormat::Raw,
        VideoFormat::TiffSequence,
    ];

    pub fn label(self) -> &'static str {
        match self {
            VideoFormat::Raw => "RAW (RGB non compresso)",
            VideoFormat::TiffSequence => "Sequenza di immagini TIFF",
            VideoFormat::H264 => "H.264 (MP4)",
            VideoFormat::H265 => "H.265 / HEVC (MP4)",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            VideoFormat::Raw => "rgb",
            VideoFormat::TiffSequence => "tif",
            VideoFormat::H264 | VideoFormat::H265 => "mp4",
        }
    }

    pub fn codec(self) -> Option<Codec> {
        match self {
            VideoFormat::H264 => Some(Codec::H264),
            VideoFormat::H265 => Some(Codec::H265),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    H265,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Nvenc,
    Amf,
    Qsv,
    VideoToolbox,
    Vaapi,
    Software,
}

/// One ffmpeg video encoder.
#[derive(Debug, PartialEq)]
pub struct EncoderSpec {
    pub name: &'static str,
    pub codec: Codec,
    pub family: Family,
    pub label: &'static str,
}

impl EncoderSpec {
    pub fn on_gpu(&self) -> bool {
        self.family != Family::Software
    }
}

/// Every encoder the program knows, hardware first.
pub const ENCODERS: &[EncoderSpec] = &[
    EncoderSpec {
        name: "h264_nvenc",
        codec: Codec::H264,
        family: Family::Nvenc,
        label: "NVIDIA NVENC (GPU)",
    },
    EncoderSpec {
        name: "hevc_nvenc",
        codec: Codec::H265,
        family: Family::Nvenc,
        label: "NVIDIA NVENC (GPU)",
    },
    EncoderSpec {
        name: "h264_amf",
        codec: Codec::H264,
        family: Family::Amf,
        label: "AMD AMF (GPU)",
    },
    EncoderSpec {
        name: "hevc_amf",
        codec: Codec::H265,
        family: Family::Amf,
        label: "AMD AMF (GPU)",
    },
    EncoderSpec {
        name: "h264_qsv",
        codec: Codec::H264,
        family: Family::Qsv,
        label: "Intel Quick Sync (GPU)",
    },
    EncoderSpec {
        name: "hevc_qsv",
        codec: Codec::H265,
        family: Family::Qsv,
        label: "Intel Quick Sync (GPU)",
    },
    EncoderSpec {
        name: "h264_videotoolbox",
        codec: Codec::H264,
        family: Family::VideoToolbox,
        label: "Apple VideoToolbox (GPU)",
    },
    EncoderSpec {
        name: "hevc_videotoolbox",
        codec: Codec::H265,
        family: Family::VideoToolbox,
        label: "Apple VideoToolbox (GPU)",
    },
    EncoderSpec {
        name: "h264_vaapi",
        codec: Codec::H264,
        family: Family::Vaapi,
        label: "VA-API (GPU, Linux)",
    },
    EncoderSpec {
        name: "hevc_vaapi",
        codec: Codec::H265,
        family: Family::Vaapi,
        label: "VA-API (GPU, Linux)",
    },
    EncoderSpec {
        name: "libx264",
        codec: Codec::H264,
        family: Family::Software,
        label: "x264 (solo CPU)",
    },
    EncoderSpec {
        name: "libx265",
        codec: Codec::H265,
        family: Family::Software,
        label: "x265 (solo CPU)",
    },
];

pub fn spec(name: &str) -> Option<&'static EncoderSpec> {
    ENCODERS.iter().find(|e| e.name == name)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateMode {
    ConstantQuality,
    Bitrate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Speed {
    Fast,
    Balanced,
    Quality,
}

impl Speed {
    pub const ALL: [Speed; 3] = [Speed::Fast, Speed::Balanced, Speed::Quality];

    pub fn label(self) -> &'static str {
        match self {
            Speed::Fast => "Veloce",
            Speed::Balanced => "Bilanciata",
            Speed::Quality => "Massima qualità (lenta)",
        }
    }
}

/// Everything about how an animation is written.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VideoSettings {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub format: VideoFormat,
    /// ffmpeg encoder name; empty picks the best graphics-card encoder.
    pub encoder: String,
    pub rate_mode: RateMode,
    /// Constant quality, 0 (best) to 51 (smallest).
    pub quality: u8,
    pub bitrate_mbps: f32,
    pub speed: Speed,
    /// Anti-aliasing: each pixel is the average of factor×factor samples.
    pub supersampling: u32,
    pub tiff_compression: TiffCompression,
    pub vaapi_device: String,
}

impl Default for VideoSettings {
    fn default() -> Self {
        VideoSettings {
            width: 1920,
            height: 1080,
            fps: 30,
            format: VideoFormat::H264,
            encoder: String::new(),
            rate_mode: RateMode::ConstantQuality,
            quality: 20,
            bitrate_mbps: 20.0,
            speed: Speed::Balanced,
            supersampling: 1,
            tiff_compression: TiffCompression::Lzw,
            vaapi_device: "/dev/dri/renderD128".into(),
        }
    }
}

/// Common resolutions for animations.
pub const RESOLUTIONS: &[(&str, u32, u32)] = &[
    ("640×360", 640, 360),
    ("854×480 (480p)", 854, 480),
    ("1280×720 (HD 720p)", 1280, 720),
    ("1920×1080 (Full HD)", 1920, 1080),
    ("2560×1440 (QHD)", 2560, 1440),
    ("3840×2160 (4K UHD)", 3840, 2160),
    ("7680×4320 (8K)", 7680, 4320),
    ("1080×1920 (verticale)", 1080, 1920),
    ("1080×1080 (quadrato)", 1080, 1080),
];

/// A command that never opens a console window on Windows.
pub fn quiet_command(program: &Path) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn runs(program: &Path) -> bool {
    quiet_command(program)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Find ffmpeg: the path chosen in the settings, next to this program, on the PATH, or in the
/// usual install folders.
pub fn find_ffmpeg(custom: &str) -> Option<PathBuf> {
    let exe = if cfg!(windows) {
        "ffmpeg.exe"
    } else {
        "ffmpeg"
    };
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !custom.trim().is_empty() {
        candidates.push(PathBuf::from(custom.trim()));
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
    {
        candidates.push(dir.join(exe));
        candidates.push(dir.join("ffmpeg").join(exe));
        candidates.push(dir.join("ffmpeg").join("bin").join(exe));
    }
    candidates.push(PathBuf::from(exe));
    for p in [
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/usr/bin/ffmpeg",
        "/snap/bin/ffmpeg",
    ] {
        candidates.push(PathBuf::from(p));
    }
    candidates.into_iter().find(|p| runs(p))
}

/// Where the frames come from when building an ffmpeg command line.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Input {
    /// Raw RGB frames written to ffmpeg's standard input.
    Pipe,
    /// A short generated test clip (to check an encoder works).
    TestPattern,
}

fn estimated_bitrate(s: &VideoSettings, codec: Codec) -> f64 {
    let bpp = 0.25 * 2f64.powf((23.0 - s.quality as f64) / 6.0);
    let factor = if codec == Codec::H265 { 0.6 } else { 1.0 };
    (s.width as f64 * s.height as f64 * s.fps as f64 * bpp * factor).max(100_000.0)
}

/// The ffmpeg arguments that encode with `enc` according to `s`.
pub fn ffmpeg_args(
    enc: &EncoderSpec,
    s: &VideoSettings,
    input: Input,
    output: &Path,
) -> Vec<String> {
    let mut a: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
        .map(String::from)
        .to_vec();
    let (w, h) = match input {
        Input::Pipe => (s.width, s.height),
        Input::TestPattern => (320, 240),
    };
    if enc.family == Family::Vaapi {
        a.extend(["-vaapi_device".into(), s.vaapi_device.clone()]);
    }
    match input {
        Input::Pipe => a.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                &format!("{w}x{h}"),
                "-r",
                &s.fps.to_string(),
                "-i",
                "-",
            ]
            .map(String::from),
        ),
        Input::TestPattern => a.extend(
            [
                "-f",
                "lavfi",
                "-i",
                &format!("testsrc2=size={w}x{h}:rate={}", s.fps.max(1)),
                "-frames:v",
                "5",
            ]
            .map(String::from),
        ),
    }
    let pix = match enc.family {
        Family::Qsv | Family::Vaapi => "nv12",
        _ => "yuv420p",
    };
    let mut vf = format!("scale=out_color_matrix=bt709:out_range=tv,format={pix}");
    if enc.family == Family::Vaapi {
        vf.push_str(",hwupload");
    }
    a.extend(["-vf".into(), vf, "-c:v".into(), enc.name.into()]);

    let speed = s.speed;
    let opt = |a: &mut Vec<String>, k: &str, v: &str| {
        a.push(k.into());
        a.push(v.into());
    };
    match enc.family {
        Family::Nvenc => {
            opt(
                &mut a,
                "-preset",
                match speed {
                    Speed::Fast => "p2",
                    Speed::Balanced => "p5",
                    Speed::Quality => "p7",
                },
            );
            opt(&mut a, "-tune", "hq");
        }
        Family::Amf => opt(
            &mut a,
            "-quality",
            match speed {
                Speed::Fast => "speed",
                Speed::Balanced => "balanced",
                Speed::Quality => "quality",
            },
        ),
        Family::Qsv => opt(
            &mut a,
            "-preset",
            match speed {
                Speed::Fast => "veryfast",
                Speed::Balanced => "medium",
                Speed::Quality => "veryslow",
            },
        ),
        Family::Software => opt(
            &mut a,
            "-preset",
            match speed {
                Speed::Fast => "veryfast",
                Speed::Balanced => "medium",
                Speed::Quality => "slow",
            },
        ),
        Family::VideoToolbox | Family::Vaapi => {}
    }

    let q = s.quality.clamp(1, 51).to_string();
    let bitrate = match s.rate_mode {
        RateMode::Bitrate => (s.bitrate_mbps.max(0.1) as f64) * 1.0e6,
        RateMode::ConstantQuality => estimated_bitrate(s, enc.codec),
    };
    let b = format!("{}k", (bitrate / 1000.0).round() as u64);
    let maxrate = format!("{}k", (bitrate * 1.5 / 1000.0).round() as u64);
    let bufsize = format!("{}k", (bitrate * 2.0 / 1000.0).round() as u64);
    match (s.rate_mode, enc.family) {
        (RateMode::ConstantQuality, Family::Nvenc) => {
            opt(&mut a, "-rc", "vbr");
            opt(&mut a, "-cq", &q);
            opt(&mut a, "-b:v", "0");
        }
        (RateMode::ConstantQuality, Family::Amf) => {
            opt(&mut a, "-rc", "cqp");
            opt(&mut a, "-qp_i", &q);
            opt(&mut a, "-qp_p", &q);
            if enc.codec == Codec::H264 {
                opt(&mut a, "-qp_b", &q);
            }
        }
        (RateMode::ConstantQuality, Family::Qsv) => opt(&mut a, "-global_quality", &q),
        (RateMode::ConstantQuality, Family::Vaapi) => {
            opt(&mut a, "-rc_mode", "CQP");
            opt(&mut a, "-qp", &q);
        }
        (RateMode::ConstantQuality, Family::Software) => opt(&mut a, "-crf", &q),
        // VideoToolbox has no constant quality on every Mac: use a bitrate that matches it.
        (RateMode::ConstantQuality, Family::VideoToolbox) => opt(&mut a, "-b:v", &b),
        (RateMode::Bitrate, family) => {
            match family {
                Family::Nvenc => opt(&mut a, "-rc", "vbr"),
                Family::Amf => opt(&mut a, "-rc", "vbr_peak"),
                Family::Vaapi => opt(&mut a, "-rc_mode", "VBR"),
                _ => {}
            }
            opt(&mut a, "-b:v", &b);
            if family != Family::VideoToolbox {
                opt(&mut a, "-maxrate", &maxrate);
                opt(&mut a, "-bufsize", &bufsize);
            }
        }
    }
    opt(&mut a, "-g", &(s.fps.max(1) * 2).to_string());
    for (k, v) in [
        ("-colorspace", "bt709"),
        ("-color_primaries", "bt709"),
        ("-color_trc", "bt709"),
        ("-color_range", "tv"),
    ] {
        opt(&mut a, k, v);
    }
    if enc.codec == Codec::H265 {
        opt(&mut a, "-tag:v", "hvc1");
    }
    match input {
        Input::TestPattern => a.extend(["-f", "null", "-"].map(String::from)),
        Input::Pipe => {
            opt(&mut a, "-movflags", "+faststart");
            a.push(output.to_string_lossy().into_owned());
        }
    }
    a
}

/// Result of trying one encoder.
#[derive(Clone, Debug)]
pub struct EncoderProbe {
    pub name: &'static str,
    pub works: bool,
    pub message: String,
}

fn run_with_timeout(mut cmd: Command, timeout: Duration) -> Result<(), String> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let mut stderr = child.stderr.take();
    let reader = std::thread::spawn(move || {
        let mut s = String::new();
        if let Some(e) = stderr.as_mut() {
            let _ = e.read_to_string(&mut s);
        }
        s
    });
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let text = reader.join().unwrap_or_default();
                return if status.success() {
                    Ok(())
                } else {
                    Err(last_lines(&text, 3))
                };
            }
            Ok(None) if start.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err("tempo scaduto".into());
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(30)),
            Err(e) => return Err(e.to_string()),
        }
    }
}

fn last_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(n)..].join(" | ")
}

/// Try every known encoder with a tiny clip to learn which ones really work on this computer.
pub fn probe_encoders(ffmpeg: &Path, settings: &VideoSettings) -> Vec<EncoderProbe> {
    let mut s = settings.clone();
    s.rate_mode = RateMode::ConstantQuality;
    let handles: Vec<_> = ENCODERS
        .iter()
        .filter(|e| match e.family {
            Family::VideoToolbox => cfg!(target_os = "macos"),
            Family::Vaapi => cfg!(target_os = "linux"),
            Family::Nvenc | Family::Amf | Family::Qsv => !cfg!(target_os = "macos"),
            Family::Software => true,
        })
        .map(|enc| {
            let mut cmd = quiet_command(ffmpeg);
            cmd.args(ffmpeg_args(enc, &s, Input::TestPattern, Path::new("")));
            std::thread::spawn(move || {
                let r = run_with_timeout(cmd, Duration::from_secs(25));
                EncoderProbe {
                    name: enc.name,
                    works: r.is_ok(),
                    message: r.err().unwrap_or_default(),
                }
            })
        })
        .collect();
    handles.into_iter().filter_map(|h| h.join().ok()).collect()
}

/// The encoder to use: the chosen one if it works, otherwise the first working graphics-card
/// encoder for the codec, otherwise the processor encoder.
pub fn pick_encoder(
    codec: Codec,
    working: &[&'static str],
    chosen: &str,
) -> Option<&'static EncoderSpec> {
    if let Some(e) = spec(chosen) {
        if e.codec == codec && working.contains(&e.name) {
            return Some(e);
        }
    }
    ENCODERS
        .iter()
        .filter(|e| e.codec == codec && working.contains(&e.name))
        .min_by_key(|e| !e.on_gpu())
}

/// Where finished frames go.
pub enum Sink {
    Raw {
        file: BufWriter<File>,
        frames: u64,
        path: PathBuf,
    },
    Tiff {
        base: PathBuf,
        index: u32,
        options: ImageOptions,
    },
    Ffmpeg {
        child: Child,
        stdin: Option<ChildStdin>,
        errors: Option<JoinHandle<String>>,
        path: PathBuf,
    },
}

impl Sink {
    pub fn open(
        settings: &VideoSettings,
        path: &Path,
        ffmpeg: Option<&Path>,
        encoder: Option<&'static EncoderSpec>,
    ) -> Result<Sink, String> {
        match settings.format {
            VideoFormat::Raw => {
                let file = File::create(path)
                    .map_err(|e| format!("Impossibile creare {}: {e}", path.display()))?;
                let info = format!(
                    "Video RAW non compresso\r\nFile: {}\r\nRisoluzione: {}x{}\r\nFotogrammi al secondo: {}\r\nFormato pixel: RGB 8 bit (rgb24), righe dall'alto in basso, nessuna intestazione\r\n\r\nPer vederlo con ffmpeg:\r\nffplay -f rawvideo -pixel_format rgb24 -video_size {}x{} -framerate {} \"{}\"\r\n",
                    path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
                    settings.width, settings.height, settings.fps,
                    settings.width, settings.height, settings.fps,
                    path.display()
                );
                std::fs::write(path.with_extension("txt"), info).map_err(|e| e.to_string())?;
                Ok(Sink::Raw {
                    file: BufWriter::with_capacity(8 << 20, file),
                    frames: 0,
                    path: path.to_path_buf(),
                })
            }
            VideoFormat::TiffSequence => Ok(Sink::Tiff {
                base: path.with_extension(""),
                index: 0,
                options: ImageOptions {
                    format: ImageFormat::Tiff,
                    jpeg_quality: 90,
                    tiff_compression: settings.tiff_compression,
                },
            }),
            VideoFormat::H264 | VideoFormat::H265 => {
                let ffmpeg = ffmpeg.ok_or("ffmpeg non trovato: installalo o indica dove si trova nelle impostazioni video")?;
                let enc =
                    encoder.ok_or("Nessun codificatore video disponibile per questo formato")?;
                let mut cmd = quiet_command(ffmpeg);
                cmd.args(ffmpeg_args(enc, settings, Input::Pipe, path));
                let mut child = cmd
                    .stdin(Stdio::piped())
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("Impossibile avviare ffmpeg: {e}"))?;
                let stdin = child.stdin.take();
                let mut stderr = child.stderr.take();
                let errors = std::thread::spawn(move || {
                    let mut s = String::new();
                    if let Some(e) = stderr.as_mut() {
                        let _ = e.read_to_string(&mut s);
                    }
                    s
                });
                Ok(Sink::Ffmpeg {
                    child,
                    stdin,
                    errors: Some(errors),
                    path: path.to_path_buf(),
                })
            }
        }
    }

    pub fn write(&mut self, rgb: &[u8], w: u32, h: u32) -> Result<(), String> {
        match self {
            Sink::Raw { file, frames, .. } => {
                file.write_all(rgb)
                    .map_err(|e| format!("Errore di scrittura: {e}"))?;
                *frames += 1;
                Ok(())
            }
            Sink::Tiff {
                base,
                index,
                options,
            } => {
                *index += 1;
                let name = format!(
                    "{}_{:05}.tif",
                    base.file_name()
                        .map(|n| n.to_string_lossy())
                        .unwrap_or_default(),
                    index
                );
                let p = base.with_file_name(name);
                export::save_rgb(&p, w, h, rgb, options)
            }
            Sink::Ffmpeg {
                stdin,
                child,
                errors,
                ..
            } => {
                let pipe = stdin.as_mut().ok_or("ffmpeg è già chiuso")?;
                if pipe.write_all(rgb).is_err() {
                    stdin.take();
                    let _ = child.wait();
                    let text = errors
                        .take()
                        .and_then(|h| h.join().ok())
                        .unwrap_or_default();
                    return Err(format!("ffmpeg si è fermato: {}", last_lines(&text, 4)));
                }
                Ok(())
            }
        }
    }

    /// Close the output; returns a short description of what was written.
    pub fn finish(self) -> Result<String, String> {
        match self {
            Sink::Raw {
                mut file,
                frames,
                path,
            } => {
                file.flush().map_err(|e| e.to_string())?;
                Ok(format!(
                    "{frames} fotogrammi RAW in {} (dettagli nel file .txt accanto)",
                    path.display()
                ))
            }
            Sink::Tiff { base, index, .. } => Ok(format!(
                "{index} immagini TIFF ({}_00001.tif …)",
                base.display()
            )),
            Sink::Ffmpeg {
                mut child,
                stdin,
                errors,
                path,
            } => {
                drop(stdin);
                let status = child.wait().map_err(|e| e.to_string())?;
                let text = errors.and_then(|h| h.join().ok()).unwrap_or_default();
                if status.success() {
                    Ok(format!("Video salvato in {}", path.display()))
                } else {
                    Err(format!(
                        "ffmpeg ha segnalato un errore: {}",
                        last_lines(&text, 4)
                    ))
                }
            }
        }
    }

    /// Stop without finishing (the partial file is removed for videos).
    pub fn abort(self) {
        if let Sink::Ffmpeg {
            mut child,
            stdin,
            path,
            ..
        } = self
        {
            drop(stdin);
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvenc_quality_args() {
        let s = VideoSettings::default();
        let a = ffmpeg_args(
            spec("hevc_nvenc").unwrap(),
            &s,
            Input::Pipe,
            Path::new("out.mp4"),
        );
        let joined = a.join(" ");
        assert!(joined.contains("-c:v hevc_nvenc"));
        assert!(joined.contains("-cq 20"));
        assert!(joined.contains("-tag:v hvc1"));
        assert!(joined.contains("-s 1920x1080"));
        assert!(joined.ends_with("out.mp4"));
    }

    #[test]
    fn picks_gpu_first() {
        let working = ["libx264", "h264_qsv", "h264_nvenc"];
        assert_eq!(
            pick_encoder(Codec::H264, &working, "").unwrap().name,
            "h264_nvenc"
        );
        assert_eq!(
            pick_encoder(Codec::H264, &working, "h264_qsv")
                .unwrap()
                .name,
            "h264_qsv"
        );
        assert_eq!(
            pick_encoder(Codec::H264, &["libx264"], "").unwrap().name,
            "libx264"
        );
        assert!(pick_encoder(Codec::H265, &working, "").is_none());
    }
}
