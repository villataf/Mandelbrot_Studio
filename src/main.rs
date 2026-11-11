#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

//! Mandelbrot Studio: explore the Mandelbrot set, save pictures and make animations.

mod animation;
mod app;
mod compute;
mod export;
mod flight;
mod jobs;
mod palette;
mod recommend;
mod selfcheck;
mod settings;
mod video;
mod view;
mod workers;

use eframe::egui;

fn main() -> eframe::Result {
    let args: Vec<String> = std::env::args().collect();
    // `--cartella-impostazioni <cartella>`: keep settings and palettes in another folder.
    if let Some(dir) = args.iter().position(|a| a == "--cartella-impostazioni").and_then(|i| args.get(i + 1)) {
        std::env::set_var("MANDELBROT_CONFIG_DIR", dir);
    }
    if args.iter().any(|a| a == "--diagnostica") {
        diagnostics();
        return Ok(());
    }
    if let Some(i) = args.iter().position(|a| a == "--confronto") {
        let num = |k: usize, d: f64| args.get(i + k).and_then(|s| s.parse::<f64>().ok()).unwrap_or(d);
        benchmark(
            view::View { cx: num(1, -0.75), cy: num(2, 0.1), scale: num(3, 1e-3) },
            num(4, 10000.0) as u32,
            num(5, 960.0) as u32,
            num(6, 1060.0) as u32,
            args.get(i + 7).map(std::path::PathBuf::from),
        );
        return Ok(());
    }
    if let Some(i) = args.iter().position(|a| a == "--prova-esportazione") {
        let dir = args.get(i + 1).map(std::path::PathBuf::from).unwrap_or_else(|| "prova-esportazione".into());
        let ok = selfcheck::run(dir);
        std::process::exit(if ok { 0 } else { 1 });
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Mandelbrot Studio")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "Mandelbrot Studio",
        options,
        Box::new(|cc| Ok(Box::new(app::MandelApp::new(cc)))),
    )
}

/// `--confronto <re> <im> <scala> <iterazioni> [larghezza altezza [cartella]]`: time the
/// processor and every graphics-card method on one view and say how much they agree; with
/// a folder, also save each picture there as PNG.
fn benchmark(view: view::View, max_iter: u32, w: u32, h: u32, folder: Option<std::path::PathBuf>) {
    use compute::gpu::{GpuContext, Precision};
    use compute::{cpu, FractalParams, INTERIOR, UNKNOWN};

    let save = |name: &str, data: &[f32]| {
        if let Some(dir) = &folder {
            let colors = palette::ColorSettings::default();
            let rgb = palette::colorize_supersampled(
                data,
                w as usize,
                h as usize,
                1,
                &palette::Lut::new(&colors.palette),
                palette::ColorMap::from(&colors),
            );
            let opts = export::ImageOptions { format: export::ImageFormat::Png, ..Default::default() };
            let _ = std::fs::create_dir_all(dir);
            let _ = export::save_rgb(&dir.join(name), w, h, &rgb, &opts);
        }
    };
    let params = FractalParams { view, max_iter };
    let never = || false;
    let started = std::time::Instant::now();
    let reference = cpu::render(&params, w, h, &never, None).unwrap();
    let count = |d: &[f32], v: f32| d.iter().filter(|x| **x == v).count();
    println!(
        "CPU {w}×{h}, {max_iter} iterazioni: {:?} (interni {}, non decisi {}, esterni {})",
        started.elapsed(),
        count(&reference, INTERIOR),
        count(&reference, UNKNOWN),
        reference.iter().filter(|x| **x >= 0.0).count()
    );
    save("cpu.png", &reference);
    let Ok(ctx) = GpuContext::new(None) else { return };
    for p in Precision::ALL {
        if !ctx.has(p) || p == Precision::Single {
            continue;
        }
        let started = std::time::Instant::now();
        if let Ok(Some(d)) = ctx.render_with(p, &params, w, h, &never, None) {
            save(&format!("gpu_{p:?}.png"), &d);
            let same = d
                .iter()
                .zip(&reference)
                .filter(|(a, b)| (**a < 0.0) == (**b < 0.0) && (**a < 0.0 || (**a - **b).abs() < 0.5))
                .count();
            // Same colour: the default colouring (square root of the count) within 2% of the
            // palette, which is what the eye sees.
            let same_colour = d
                .iter()
                .zip(&reference)
                .filter(|(a, b)| {
                    (**a < 0.0) == (**b < 0.0) && (**a < 0.0 || (a.sqrt() - b.sqrt()).abs() / 4.0 < 0.02)
                })
                .count();
            println!(
                "GPU {}: {:?}, uguale {:.1}%, stesso colore {:.1}% (interni {}, non decisi {})",
                p.label(),
                started.elapsed(),
                same as f64 * 100.0 / d.len() as f64,
                same_colour as f64 * 100.0 / d.len() as f64,
                count(&d, INTERIOR),
                count(&d, UNKNOWN)
            );
        }
    }
}

/// `--diagnostica`: report the graphics cards, their precision and the working video
/// encoders, and compare a GPU image with the processor's.
fn diagnostics() {
    println!(
        "Mandelbrot Studio {} – diagnostica",
        env!("CARGO_PKG_VERSION")
    );
    let adapters = compute::gpu::list_adapters();
    for a in &adapters {
        println!(
            "Scheda: {} [{}] doppia nativa: {}",
            a.key, a.kind, a.supports_f64
        );
    }
    for a in &adapters {
        gpu_report(&a.key);
    }
    if let Some(a) = adapters.first() {
        heavy_report(&a.key);
    }
    match video::find_ffmpeg("") {
        Some(f) => {
            println!("ffmpeg: {}", f.display());
            for p in video::probe_encoders(&f, &video::VideoSettings::default()) {
                println!(
                    "  {:<20} {}",
                    p.name,
                    if p.works {
                        "funziona".to_string()
                    } else {
                        format!("no ({})", p.message)
                    }
                );
            }
        }
        None => println!("ffmpeg non trovato"),
    }
}

/// A million iterations inside a mini-Mandelbrot: the work must be cut into pieces short
/// enough that the system never resets the graphics card.
fn heavy_report(key: &str) {
    use compute::gpu::{GpuContext, Precision};
    use compute::FractalParams;

    let Ok(ctx) = GpuContext::new(Some(key)) else { return };
    let params = FractalParams {
        view: view::View { cx: -1.754_877_666_246_693, cy: 0.0, scale: 0.025 },
        max_iter: 1_000_000,
    };
    for p in [Precision::Perturbation, Precision::DoubleEmulated, Precision::DoubleNative] {
        if !ctx.has(p) {
            continue;
        }
        let never = || false;
        let started = std::time::Instant::now();
        match ctx.render_with(p, &params, 320, 180, &never, None) {
            Ok(Some(data)) => println!(
                "\n[1.000.000 iterazioni, {}] {} 320×180: {:?}, punti interni {}",
                ctx.entry.key,
                p.label(),
                started.elapsed(),
                data.iter().filter(|v| **v < 0.0).count()
            ),
            Ok(None) => {}
            Err(e) => println!("\n[1.000.000 iterazioni] {}: errore {e}", p.label()),
        }
    }
}

fn gpu_report(key: &str) {
    use compute::gpu::{GpuContext, Precision};
    use compute::{cpu, FractalParams};

    println!();
    match GpuContext::new(Some(key)) {
        Ok(ctx) => {
            println!("In uso: {}", ctx.entry.key);
            for n in &ctx.notes {
                println!("  {n}");
            }
            let shallow = FractalParams {
                view: view::View::default(),
                max_iter: 2000,
            };
            let deep = FractalParams {
                view: view::View {
                    cx: -0.743_643_887_037_158_7,
                    cy: 0.131_825_904_205_311_97,
                    scale: 1.0e-9,
                },
                max_iter: 5000,
            };
            for (name, params) in [("vista iniziale", shallow), ("zoom 1,25e9×", deep)] {
                let never = || false;
                let started = std::time::Instant::now();
                let reference = cpu::render(&params, 1920, 1080, &never, None).unwrap();
                println!(
                    "[{name}] CPU 1920×1080, {} iterazioni: {:?}",
                    params.max_iter,
                    started.elapsed()
                );
                for p in Precision::ALL {
                    if !ctx.has(p) {
                        continue;
                    }
                    let started = std::time::Instant::now();
                    match ctx.render_with(p, &params, 1920, 1080, &never, None) {
                        Ok(Some(data)) => {
                            let same = data
                                .iter()
                                .zip(&reference)
                                .filter(|(a, b)| {
                                    (**a < 0.0) == (**b < 0.0)
                                        && (**a < 0.0 || (**a - **b).abs() < 0.5)
                                })
                                .count();
                            println!(
                                "GPU {}: {:?}, uguale alla CPU nel {:.2}% dei pixel",
                                p.label(),
                                started.elapsed(),
                                same as f64 * 100.0 / reference.len() as f64
                            );
                        }
                        Ok(None) => println!("GPU {}: annullato", p.label()),
                        Err(e) => println!("GPU {}: errore {e}", p.label()),
                    }
                }
            }
        }
        Err(e) => println!("GPU non disponibile: {e}"),
    }
}
