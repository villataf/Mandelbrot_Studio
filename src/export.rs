//! Saving pictures as JPEG, TIFF or PNG.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use tiff::encoder::{colortype, compression, TiffEncoder};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TiffCompression {
    None,
    Lzw,
    Deflate,
    PackBits,
}

impl TiffCompression {
    pub const ALL: [TiffCompression; 4] = [
        TiffCompression::None,
        TiffCompression::Lzw,
        TiffCompression::Deflate,
        TiffCompression::PackBits,
    ];

    pub fn label(self) -> &'static str {
        match self {
            TiffCompression::None => "Nessuna compressione",
            TiffCompression::Lzw => "LZW (senza perdita)",
            TiffCompression::Deflate => "Deflate/ZIP (senza perdita)",
            TiffCompression::PackBits => "PackBits (senza perdita)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageFormat {
    Jpeg,
    Tiff,
    Png,
}

impl ImageFormat {
    pub const ALL: [ImageFormat; 3] = [ImageFormat::Jpeg, ImageFormat::Tiff, ImageFormat::Png];

    pub fn label(self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "JPEG",
            ImageFormat::Tiff => "TIFF",
            ImageFormat::Png => "PNG",
        }
    }

    pub fn extension(self) -> &'static str {
        match self {
            ImageFormat::Jpeg => "jpg",
            ImageFormat::Tiff => "tif",
            ImageFormat::Png => "png",
        }
    }
}

/// How a picture is saved.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageOptions {
    pub format: ImageFormat,
    /// JPEG quality, 1 (smallest file) to 100 (best picture).
    pub jpeg_quality: u8,
    pub tiff_compression: TiffCompression,
}

impl Default for ImageOptions {
    fn default() -> Self {
        ImageOptions {
            format: ImageFormat::Jpeg,
            jpeg_quality: 92,
            tiff_compression: TiffCompression::Lzw,
        }
    }
}

fn tiff_with<D: compression::Compression, W: Write + std::io::Seek>(
    enc: &mut TiffEncoder<W>,
    w: u32,
    h: u32,
    rgb: &[u8],
    c: D,
) -> tiff::TiffResult<()> {
    let mut image = enc.new_image_with_compression::<colortype::RGB8, D>(w, h, c)?;
    // Strips of about 64 KiB keep memory low and suit every reader.
    let rows = ((64 * 1024) / (w as usize * 3).max(1)).clamp(1, h as usize) as u32;
    image.rows_per_strip(rows)?;
    image.write_data(rgb)
}

/// Write an RGB picture (`w`×`h`, 3 bytes per pixel) to `path`.
pub fn save_rgb(
    path: &Path,
    w: u32,
    h: u32,
    rgb: &[u8],
    opts: &ImageOptions,
) -> Result<(), String> {
    let file =
        File::create(path).map_err(|e| format!("Impossibile creare {}: {e}", path.display()))?;
    let mut out = BufWriter::new(file);
    match opts.format {
        ImageFormat::Jpeg => {
            if w > 65_535 || h > 65_535 {
                return Err("Il formato JPEG non supporta lati più grandi di 65535 pixel".into());
            }
            let q = opts.jpeg_quality.clamp(1, 100);
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q)
                .encode(rgb, w, h, image::ExtendedColorType::Rgb8)
                .map_err(|e| format!("Errore JPEG: {e}"))?;
        }
        ImageFormat::Png => {
            use image::ImageEncoder;
            image::codecs::png::PngEncoder::new(&mut out)
                .write_image(rgb, w, h, image::ExtendedColorType::Rgb8)
                .map_err(|e| format!("Errore PNG: {e}"))?;
        }
        ImageFormat::Tiff => {
            let mut enc = TiffEncoder::new(&mut out).map_err(|e| format!("Errore TIFF: {e}"))?;
            let r = match opts.tiff_compression {
                TiffCompression::None => tiff_with(&mut enc, w, h, rgb, compression::Uncompressed),
                TiffCompression::Lzw => tiff_with(&mut enc, w, h, rgb, compression::Lzw),
                TiffCompression::Deflate => {
                    tiff_with(&mut enc, w, h, rgb, compression::Deflate::default())
                }
                TiffCompression::PackBits => tiff_with(&mut enc, w, h, rgb, compression::Packbits),
            };
            r.map_err(|e| format!("Errore TIFF: {e}"))?;
        }
    }
    out.flush()
        .map_err(|e| format!("Errore di scrittura: {e}"))?;
    Ok(())
}

/// Make sure `path` ends with the format's extension.
pub fn with_extension(path: &Path, format: ImageFormat) -> std::path::PathBuf {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let ok = match format {
        ImageFormat::Jpeg => ext == "jpg" || ext == "jpeg",
        ImageFormat::Tiff => ext == "tif" || ext == "tiff",
        ImageFormat::Png => ext == "png",
    };
    if ok {
        path.to_path_buf()
    } else {
        path.with_extension(format.extension())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(w: u32, h: u32) -> Vec<u8> {
        (0..w * h * 3).map(|i| (i % 251) as u8).collect()
    }

    #[test]
    fn writes_every_format() {
        let dir = std::env::temp_dir().join(format!("mandel-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rgb = sample(64, 40);
        for (fmt, tc, name) in [
            (ImageFormat::Jpeg, TiffCompression::None, "a.jpg"),
            (ImageFormat::Png, TiffCompression::None, "a.png"),
            (ImageFormat::Tiff, TiffCompression::None, "raw.tif"),
            (ImageFormat::Tiff, TiffCompression::Lzw, "lzw.tif"),
            (ImageFormat::Tiff, TiffCompression::Deflate, "zip.tif"),
            (ImageFormat::Tiff, TiffCompression::PackBits, "pb.tif"),
        ] {
            let p = dir.join(name);
            save_rgb(
                &p,
                64,
                40,
                &rgb,
                &ImageOptions {
                    format: fmt,
                    jpeg_quality: 80,
                    tiff_compression: tc,
                },
            )
            .unwrap();
            assert!(std::fs::metadata(&p).unwrap().len() > 100);
        }
        // Uncompressed TIFF must hold all the pixel bytes; the tiff reader must read LZW back.
        assert!(std::fs::metadata(dir.join("raw.tif")).unwrap().len() >= 64 * 40 * 3);
        let mut dec =
            tiff::decoder::Decoder::new(File::open(dir.join("lzw.tif")).unwrap()).unwrap();
        match dec.read_image().unwrap() {
            tiff::decoder::DecodingResult::U8(v) => assert_eq!(v, rgb),
            _ => panic!("unexpected sample type"),
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn jpeg_quality_changes_size() {
        let dir = std::env::temp_dir().join(format!("mandel-jpeg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rgb = sample(128, 128);
        let size = |q: u8| {
            let p = dir.join(format!("q{q}.jpg"));
            save_rgb(
                &p,
                128,
                128,
                &rgb,
                &ImageOptions {
                    format: ImageFormat::Jpeg,
                    jpeg_quality: q,
                    ..Default::default()
                },
            )
            .unwrap();
            std::fs::metadata(&p).unwrap().len()
        };
        assert!(size(10) < size(95));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
