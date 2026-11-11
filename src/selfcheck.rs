//! `--prova-esportazione <cartella>`: save a few pictures and a short animation in every
//! format, to check the whole saving chain (calculation, colours, files, ffmpeg) on this
//! computer without opening the window.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::animation::{Animation, Easing, Keyframe};
use crate::compute::gpu::PrecisionChoice;
use crate::compute::{ComputeHub, Engine, FractalParams, GpuState};
use crate::export::{ImageFormat, ImageOptions, TiffCompression};
use crate::jobs::{self, ComputeChoice, JobStatus};
use crate::palette::{presets, ColorMap, ColorSettings, Lut, Mapping};
use crate::video::{self, VideoFormat, VideoSettings};
use crate::view::View;

fn wait(job: &Arc<JobStatus>) -> Result<String, String> {
    while !job.done.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(50));
    }
    job.result.lock().unwrap().take().unwrap_or(Err("nessun risultato".into()))
}

fn size(p: &Path) -> String {
    std::fs::metadata(p).map(|m| format!("{} byte", m.len())).unwrap_or_else(|_| "mancante".into())
}

pub fn run(dir: PathBuf) -> bool {
    if let Err(e) = std::fs::create_dir_all(&dir) {
        println!("Impossibile creare {}: {e}", dir.display());
        return false;
    }
    let hub = ComputeHub::new();
    hub.start_gpu(None, || {});
    while matches!(*hub.gpu.read().unwrap(), GpuState::NotStarted | GpuState::Starting) {
        std::thread::sleep(Duration::from_millis(50));
    }
    let choice = ComputeChoice { engine: Engine::Gpu, precision: PrecisionChoice::Auto };
    let mut ok = true;

    // Pictures.
    let color = ColorSettings::default();
    let lut = Lut::new(&color.palette);
    let params = FractalParams {
        view: View { cx: -0.743_643_887_037_158_7, cy: 0.131_825_904_205_311_97, scale: 2.0e-3 },
        max_iter: 1500,
    };
    for (name, options) in [
        ("immagine_q95.jpg", ImageOptions { format: ImageFormat::Jpeg, jpeg_quality: 95, ..Default::default() }),
        ("immagine_q20.jpg", ImageOptions { format: ImageFormat::Jpeg, jpeg_quality: 20, ..Default::default() }),
        ("immagine_lzw.tif", ImageOptions { format: ImageFormat::Tiff, tiff_compression: TiffCompression::Lzw, ..Default::default() }),
        ("immagine_non_compressa.tif", ImageOptions { format: ImageFormat::Tiff, tiff_compression: TiffCompression::None, ..Default::default() }),
    ] {
        let path = dir.join(name);
        let started = Instant::now();
        let job = jobs::start_image_export(hub.clone(), choice, params, 1920, 1080, 2, lut.clone(), ColorMap::from(&color), options, path.clone());
        match wait(&job) {
            Ok(m) => println!("OK   {name}: {} in {:?} — {m}", size(&path), started.elapsed()),
            Err(e) => {
                ok = false;
                println!("ERR  {name}: {e}");
            }
        }
    }

    // A short animation: zoom from the whole set into the seahorse valley while the palette
    // changes from "Classica" to "Fuoco" and cycles.
    let p = presets();
    let mut fire = ColorSettings { palette: p[1].clone(), density: 2.0, offset: 0.3, mapping: Mapping::Sqrt };
    fire.palette.interior = [10, 0, 0];
    let anim = Animation {
        keyframes: vec![
            Keyframe { name: "inizio".into(), view: View::default(), iterations: 300, color: ColorSettings::default(), duration: 2.0, easing: Easing::Smooth },
            Keyframe { name: "fine".into(), view: params.view, iterations: 1500, color: fire, duration: 1.0, easing: Easing::Smooth },
        ],
        color_cycle_speed: 0.1,
    };
    let ffmpeg = video::find_ffmpeg("");
    let working = ffmpeg
        .as_deref()
        .map(|f| video::probe_encoders(f, &VideoSettings::default()))
        .unwrap_or_default()
        .into_iter()
        .filter(|p| p.works)
        .map(|p| p.name)
        .collect::<Vec<_>>();
    println!("Codificatori funzionanti: {}", working.join(", "));
    for (format, name) in [
        (VideoFormat::H264, "animazione_h264.mp4"),
        (VideoFormat::H265, "animazione_h265.mp4"),
        (VideoFormat::Raw, "animazione.rgb"),
        (VideoFormat::TiffSequence, "fotogramma.tif"),
    ] {
        let settings = VideoSettings { width: 640, height: 360, fps: 30, format, ..Default::default() };
        let encoder = format.codec().and_then(|c| video::pick_encoder(c, &working, ""));
        let path = dir.join(name);
        let started = Instant::now();
        let job = jobs::start_video_export(hub.clone(), choice, anim.clone(), settings, ffmpeg.clone(), working.clone(), path.clone());
        match wait(&job) {
            Ok(m) => println!(
                "OK   {name}{}: {} in {:?} — {m}",
                encoder.map(|e| format!(" [{}]", e.name)).unwrap_or_default(),
                size(&path),
                started.elapsed()
            ),
            Err(e) => {
                ok = false;
                println!("ERR  {name}: {e}");
            }
        }
    }
    ok
}
