use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, GenericImageView};
use ort::{
    execution_providers::DirectMLExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Tensor,
};
use rayon::prelude::*;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
    },
    time::SystemTime,
};

const SIDE: usize = 224;
const CHANNEL: usize = SIDE * SIDE;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Debug)]
pub struct Pair {
    pub score: f32,
    pub a: PathBuf,
    pub b: PathBuf,
}
pub enum Event {
    Progress(String),
    Device(String),
    Done(Vec<Pair>),
    Failed(String),
    Deleted(PathBuf),
}
pub enum Command {
    Scan {
        folder: PathBuf,
        model: PathBuf,
        gpu: bool,
        adapter: i32,
        threshold: f32,
        recursive: bool,
    },
    Delete(PathBuf),
}

pub fn find_asset(relative: &str) -> PathBuf {
    for start in [
        std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf)),
        std::env::current_dir().ok(),
    ]
    .into_iter()
    .flatten()
    {
        for parent in start.ancestors() {
            let candidate = parent.join(relative);
            if candidate.is_file() {
                return candidate;
            }
        }
    }
    PathBuf::from(relative)
}

pub fn init_runtime() -> Result<()> {
    let dll = std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| find_asset("onnxruntime.dll"));
    if !dll.is_file() {
        bail!("ONNX Runtime not found. Run package.ps1 to place the native DLLs beside the executable, or set ORT_DYLIB_PATH.");
    }
    ort::init_from(dll.to_string_lossy()).commit()?;
    Ok(())
}

fn create_session(model: &Path, gpu: bool, adapter: i32) -> Result<(Session, String)> {
    if gpu {
        let attempt = (|| -> Result<Session> {
            Ok(Session::builder()?
                .with_memory_pattern(false)?
                .with_parallel_execution(false)?
                .with_execution_providers([DirectMLExecutionProvider::default()
                    .with_device_id(adapter)
                    .build()
                    .error_on_failure()])?
                .commit_from_file(model)?)
        })();
        match attempt {
            Ok(session) => return Ok((session, format!("GPU (DirectML adapter {adapter})"))),
            Err(error) => eprintln!("DirectML unavailable, falling back to CPU: {error:#}"),
        }
    }
    // Keep the existing app's fp16 CPU optimization workaround.
    Ok((
        Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level2)?
            .commit_from_file(model)?,
        if gpu {
            "CPU (DirectML unavailable)".into()
        } else {
            "CPU".into()
        },
    ))
}

fn preprocess(path: &Path) -> Result<Vec<f32>> {
    let image = image::open(path)
        .with_context(|| format!("Cannot decode {}", path.display()))?
        .to_rgb8();
    let (w, h) = image.dimensions();
    let scale = 256.0 / w.min(h) as f64;
    let width = ((w as f64 * scale).round_ties_even() as u32).max(256);
    let height = ((h as f64 * scale).round_ties_even() as u32).max(256);
    let resized = image::imageops::resize(&image, width, height, FilterType::CatmullRom);
    let crop = image::imageops::crop_imm(&resized, (width - 224) / 2, (height - 224) / 2, 224, 224);
    let mut data = vec![0.0; 3 * CHANNEL];
    for (x, y, pixel) in crop.pixels() {
        for c in 0..3 {
            data[c * CHANNEL + y as usize * SIDE + x as usize] =
                (pixel[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    Ok(data)
}

fn embed(session: &mut Session, tensors: Vec<Vec<f32>>) -> Result<Vec<Vec<f32>>> {
    let count = tensors.len();
    let data: Vec<f32> = tensors.into_iter().flatten().collect();
    let input = Tensor::from_array(([count, 3, SIDE, SIDE], data))?;
    let outputs = session.run(ort::inputs!["pixel_values" => input])?;
    let (shape, values) = outputs["last_hidden_state"].try_extract_tensor::<f32>()?;
    if shape.len() != 3 || shape[0] != count as i64 || shape[1] < 1 || shape[2] < 1 {
        bail!("Unexpected DINOv2 output shape: {shape:?}");
    }
    let channels = shape[2] as usize;
    let stride = shape[1] as usize * channels;
    (0..count)
        .map(|i| {
            let cls = &values[i * stride..i * stride + channels];
            let norm = cls.iter().map(|v| v * v).sum::<f32>().sqrt();
            if !norm.is_finite() || norm <= 0.0 {
                bail!("Model returned an invalid embedding");
            }
            Ok(cls.iter().map(|v| v / norm).collect())
        })
        .collect()
}

pub fn similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn list_images(
    folder: &Path,
    recursive: bool,
    cancel: &AtomicBool,
    files: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in
        std::fs::read_dir(folder).with_context(|| format!("Cannot read {}", folder.display()))?
    {
        if cancel.load(Ordering::Relaxed) {
            bail!("Scan cancelled");
        }
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_dir() && recursive {
            list_images(&entry.path(), true, cancel, files)?;
        }
        if kind.is_file()
            && entry
                .path()
                .extension()
                .and_then(|v| v.to_str())
                .is_some_and(|v| {
                    matches!(
                        v.to_ascii_lowercase().as_str(),
                        "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif"
                    )
                })
        {
            files.push(entry.path());
        }
    }
    Ok(())
}

struct Cached {
    modified: Option<SystemTime>,
    size: u64,
    vector: Option<Vec<f32>>,
}
#[derive(Default)]
pub struct Engine {
    session: Option<Session>,
    key: Option<(PathBuf, bool, i32)>,
    cache: HashMap<PathBuf, Cached>,
}
impl Engine {
    pub fn execute(
        &mut self,
        command: Command,
        tx: &Sender<Event>,
        cancel: &AtomicBool,
    ) -> Result<()> {
        match command {
            Command::Delete(path) => {
                trash::delete(&path).with_context(|| {
                    format!("Could not move {} to the Recycle Bin", path.display())
                })?;
                self.cache.remove(&path);
                let _ = tx.send(Event::Deleted(path));
            }
            Command::Scan {
                folder,
                model,
                gpu,
                adapter,
                threshold,
                recursive,
            } => {
                let mut files = Vec::new();
                list_images(&folder, recursive, cancel, &mut files)?;
                files.sort();
                if files.len() < 2 {
                    bail!("Need at least two images in this folder");
                }
                let key = (model.clone(), gpu, adapter);
                if self.key.as_ref() != Some(&key) {
                    let _ = tx.send(Event::Progress("Loading DINOv2 model…".into()));
                    let (session, device) = create_session(&model, gpu, adapter)?;
                    self.session = Some(session);
                    self.key = Some(key);
                    self.cache.clear();
                    let _ = tx.send(Event::Device(device));
                }
                let pending: Vec<_> = files
                    .iter()
                    .filter(|p| {
                        let meta = std::fs::metadata(p).ok();
                        !self.cache.get(*p).zip(meta).is_some_and(|(cached, meta)| {
                            cached.size == meta.len() && cached.modified == meta.modified().ok()
                        })
                    })
                    .cloned()
                    .collect();
                for (batch_index, batch) in pending.chunks(32).enumerate() {
                    if cancel.load(Ordering::Relaxed) {
                        bail!("Scan cancelled");
                    }
                    let _ = tx.send(Event::Progress(format!(
                        "Analysing images {}–{} of {}",
                        batch_index * 32 + 1,
                        (batch_index * 32 + batch.len()).min(pending.len()),
                        pending.len()
                    )));
                    let processed: Vec<_> = batch.par_iter().map(|p| preprocess(p).ok()).collect();
                    let valid: Vec<_> = processed
                        .iter()
                        .enumerate()
                        .filter_map(|(i, p)| p.as_ref().map(|_| i))
                        .collect();
                    let mut vectors = if valid.is_empty() {
                        Vec::new()
                    } else {
                        embed(
                            self.session.as_mut().unwrap(),
                            processed.into_iter().flatten().collect(),
                        )?
                    }
                    .into_iter();
                    for (i, path) in batch.iter().enumerate() {
                        let vector = if valid.contains(&i) {
                            vectors.next()
                        } else {
                            None
                        };
                        let meta = std::fs::metadata(path)?;
                        self.cache.insert(
                            path.clone(),
                            Cached {
                                size: meta.len(),
                                modified: meta.modified().ok(),
                                vector,
                            },
                        );
                    }
                }
                let _ = tx.send(Event::Progress("Comparing images…".into()));
                let mut pairs = Vec::new();
                for i in 0..files.len() {
                    if cancel.load(Ordering::Relaxed) {
                        bail!("Scan cancelled");
                    }
                    for j in i + 1..files.len() {
                        if let (Some(a), Some(b)) =
                            (&self.cache[&files[i]].vector, &self.cache[&files[j]].vector)
                        {
                            let score = similarity(a, b);
                            if score >= threshold {
                                pairs.push(Pair {
                                    score,
                                    a: files[i].clone(),
                                    b: files[j].clone(),
                                });
                            }
                        }
                    }
                }
                pairs.sort_by(|a, b| b.score.total_cmp(&a.score));
                let skipped = files
                    .iter()
                    .filter(|p| self.cache[*p].vector.is_none())
                    .count();
                let _ = tx.send(Event::Progress(format!(
                    "Found {} pairs in {} images ({skipped} unreadable).",
                    pairs.len(),
                    files.len()
                )));
                let _ = tx.send(Event::Done(pairs));
            }
        }
        Ok(())
    }
}

pub fn smoke_test() -> Result<()> {
    init_runtime()?;
    let model = find_asset("Model/model_fp16.onnx");
    let path =
        std::env::temp_dir().join(format!("duplicate-photo-smoke-{}.png", std::process::id()));
    image::RgbImage::from_pixel(320, 180, image::Rgb([220, 100, 30])).save(&path)?;
    let tensor = preprocess(&path);
    let _ = std::fs::remove_file(&path);
    let tensor = tensor?;
    for gpu in [false, true] {
        let (mut session, device) = create_session(&model, gpu, 0)?;
        let vectors = embed(&mut session, vec![tensor.clone(), tensor.clone()])?;
        let score = similarity(&vectors[0], &vectors[1]);
        anyhow::ensure!(
            (score - 1.0).abs() < 0.001,
            "Identical image similarity was {score}"
        );
        println!(
            "PASS {device}: {} dimensions, identical image similarity {score}",
            vectors[0].len()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preprocessing_has_correct_channels_and_crop() {
        let path =
            std::env::temp_dir().join(format!("duplicate-photo-unit-{}.png", std::process::id()));
        image::RgbImage::from_pixel(80, 160, image::Rgb([255, 128, 0]))
            .save(&path)
            .unwrap();
        let data = preprocess(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(data.len(), 3 * CHANNEL);
        for c in 0..3 {
            let value = ([255.0, 128.0, 0.0][c] / 255.0 - MEAN[c]) / STD[c];
            assert!(data[c * CHANNEL..(c + 1) * CHANNEL]
                .iter()
                .all(|v| (v - value).abs() < 0.0001));
        }
    }

    #[test]
    fn preprocessing_takes_the_center_instead_of_stretching() {
        let path =
            std::env::temp_dir().join(format!("duplicate-photo-crop-{}.png", std::process::id()));
        let image = image::RgbImage::from_fn(256, 512, |_, y| {
            if (144..368).contains(&y) {
                image::Rgb([0, 0, 255])
            } else {
                image::Rgb([255, 0, 0])
            }
        });
        image.save(&path).unwrap();
        let data = preprocess(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        for c in 0..3 {
            let expected = ([0.0, 0.0, 1.0][c] - MEAN[c]) / STD[c];
            assert!(data[c * CHANNEL..(c + 1) * CHANNEL]
                .iter()
                .all(|v| (v - expected).abs() < 0.0001));
        }
    }
}
