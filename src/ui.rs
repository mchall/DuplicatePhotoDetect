use crate::engine::{self, Command, Engine, Event, Pair};
use eframe::egui::{self, Color32, TextureHandle};
use std::{
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    time::{Duration, Instant},
};

type Preview = (PathBuf, Result<(TextureHandle, String), String>);

pub struct App {
    folder: String,
    model: String,
    threshold: f32,
    gpu: bool,
    adapter: i32,
    recursive: bool,
    pairs: Vec<Pair>,
    selected: usize,
    busy: bool,
    status: String,
    error: Option<String>,
    device: String,
    commands: Sender<Command>,
    events: Receiver<Event>,
    cancel: Arc<AtomicBool>,
    previews: Vec<Preview>,
    started: Option<Instant>,
    elapsed: Duration,
}
impl Default for App {
    fn default() -> Self {
        let (commands, rx) = mpsc::channel();
        let (tx, events) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        std::thread::spawn(move || {
            let mut engine = Engine::default();
            let mut initialized = false;
            while let Ok(command) = rx.recv() {
                let result = (|| {
                    if !initialized {
                        engine::init_runtime()?;
                        initialized = true;
                    }
                    engine.execute(command, &tx, &worker_cancel)
                })();
                if let Err(error) = result {
                    let _ = tx.send(Event::Failed(format!("{error:#}")));
                }
            }
        });
        Self {
            folder: directories::UserDirs::new()
                .and_then(|d| d.picture_dir().map(|p| p.to_string_lossy().into_owned()))
                .unwrap_or_default(),
            model: engine::find_asset("Model/model_fp16.onnx")
                .to_string_lossy()
                .into_owned(),
            threshold: 0.85,
            gpu: true,
            adapter: 0,
            recursive: false,
            pairs: vec![],
            selected: 0,
            busy: false,
            status: "Choose a folder and scan for duplicate photos.".into(),
            error: None,
            device: "Not loaded".into(),
            commands,
            events,
            cancel,
            previews: vec![],
            started: None,
            elapsed: Duration::ZERO,
        }
    }
}

impl App {
    fn scan(&mut self) {
        if !Path::new(&self.folder).is_dir() {
            self.error = Some("Select an existing folder.".into());
            return;
        }
        if !Path::new(&self.model).is_file() {
            self.error = Some("Select the DINOv2 model_fp16.onnx file.".into());
            return;
        }
        self.cancel.store(false, Ordering::Relaxed);
        self.error = None;
        self.pairs.clear();
        self.previews.clear();
        self.selected = 0;
        self.busy = true;
        self.started = Some(Instant::now());
        if self
            .commands
            .send(Command::Scan {
                folder: self.folder.clone().into(),
                model: self.model.clone().into(),
                gpu: self.gpu,
                adapter: self.adapter,
                threshold: self.threshold,
                recursive: self.recursive,
            })
            .is_err()
        {
            self.busy = false;
            self.error = Some("Scan worker stopped. Restart the application.".into());
        }
    }

    fn poll(&mut self) {
        while let Ok(event) = self.events.try_recv() {
            match event {
                Event::Progress(status) => self.status = status,
                Event::Device(device) => self.device = device,
                Event::Done(pairs) => {
                    self.pairs = pairs;
                    self.selected = 0;
                    self.finish();
                }
                Event::Failed(error) => {
                    self.error = Some(error);
                    self.finish();
                }
                Event::Deleted(path) => {
                    let previous = self.pairs.get(self.selected).cloned();
                    self.pairs.retain(|p| p.a != path && p.b != path);
                    self.selected = previous
                        .and_then(|old| {
                            self.pairs.iter().position(|p| p.a == old.a && p.b == old.b)
                        })
                        .unwrap_or(self.selected.min(self.pairs.len().saturating_sub(1)));
                    self.previews.clear();
                    self.busy = false;
                    self.status = format!("{} pairs remaining.", self.pairs.len());
                }
            }
        }
    }
    fn finish(&mut self) {
        self.busy = false;
        if let Some(start) = self.started.take() {
            self.elapsed = start.elapsed();
        }
    }

    fn preview(&mut self, ctx: &egui::Context, path: &Path) {
        if self.previews.iter().any(|(p, _)| p == path) {
            return;
        }
        let result = (|| -> anyhow::Result<_> {
            let image = image::open(path)?;
            let metadata = std::fs::metadata(path)?;
            let created = metadata
                .created()
                .ok()
                .map(|t| {
                    chrono::DateTime::<chrono::Local>::from(t)
                        .format("%Y-%m-%d %H:%M")
                        .to_string()
                })
                .unwrap_or_else(|| "unavailable".into());
            let details = format!(
                "{} × {} px · {:.1} KB · Created: {created}",
                image.width(),
                image.height(),
                metadata.len() as f64 / 1024.0
            );
            let thumbnail = image.thumbnail(1000, 800).to_rgba8();
            let color = egui::ColorImage::from_rgba_unmultiplied(
                [thumbnail.width() as usize, thumbnail.height() as usize],
                thumbnail.as_raw(),
            );
            Ok((
                ctx.load_texture(path.to_string_lossy(), color, egui::TextureOptions::LINEAR),
                details,
            ))
        })()
        .map_err(|e| e.to_string());
        self.previews.push((path.to_path_buf(), result));
    }

    fn delete(&mut self, path: PathBuf) {
        let confirmed = rfd::MessageDialog::new()
            .set_title("Delete photo?")
            .set_description(format!(
                "{}\n\nMove this photo to the Recycle Bin?",
                path.display()
            ))
            .set_buttons(rfd::MessageButtons::YesNo)
            .set_level(rfd::MessageLevel::Warning)
            .show();
        if confirmed == rfd::MessageDialogResult::Yes {
            self.error = None;
            self.busy = true;
            if self.commands.send(Command::Delete(path)).is_err() {
                self.busy = false;
                self.error = Some("Worker stopped. Restart the application.".into());
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll();
        if self.busy {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Duplicate Photo Detect");
            ui.label("Compare local photos using DINOv2 embeddings. Deleted photos go to the Recycle Bin.");
            ui.add_space(8.0);
            ui.add_enabled_ui(!self.busy, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Folder:");
                    ui.add(egui::TextEdit::singleline(&mut self.folder).desired_width((ui.available_width()-230.0).max(100.0)));
                    if ui.button("Browse…").clicked() {
                        if let Some(folder) = rfd::FileDialog::new().set_directory(&self.folder).pick_folder() { self.folder = folder.to_string_lossy().into_owned(); }
                    }
                    if ui.button("Scan folder").clicked() { self.scan(); }
                });
                ui.horizontal(|ui| {
                    ui.label("Threshold:"); ui.add(egui::DragValue::new(&mut self.threshold).range(0.01..=1.0).speed(0.01));
                    ui.checkbox(&mut self.recursive,"Include subfolders");
                    ui.checkbox(&mut self.gpu,"Use GPU (DirectML)");
                    ui.label("Adapter:"); ui.add(egui::DragValue::new(&mut self.adapter).range(0..=32));
                });
                ui.collapsing("Model settings", |ui| {
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut self.model).desired_width((ui.available_width()-110.0).max(100.0)));
                        if ui.button("Select model…").clicked() {
                            if let Some(model) = rfd::FileDialog::new().add_filter("ONNX model", &["onnx"]).pick_file() { self.model = model.to_string_lossy().into_owned(); }
                        }
                    });
                });
            });
            ui.horizontal(|ui| {
                if self.busy { ui.spinner(); if ui.button("Cancel scan").clicked() { self.cancel.store(true,Ordering::Relaxed); } }
                ui.label(&self.status);
                ui.label(format!(" · {:.1}s · {}", self.started.map(|t|t.elapsed()).unwrap_or(self.elapsed).as_secs_f32(), self.device));
            });
            if let Some(error) = &self.error { ui.colored_label(Color32::LIGHT_RED,error); }
            ui.separator();
            if let Some(pair) = self.pairs.get(self.selected).cloned() {
                ui.horizontal(|ui| {
                    if ui.add_enabled(!self.busy && self.selected>0,egui::Button::new("◀ Previous")).clicked() { self.selected-=1; }
                    ui.label(format!("Pair {} of {} · Similarity {:.3}",self.selected+1,self.pairs.len(),pair.score));
                    if ui.add_enabled(!self.busy && self.selected+1<self.pairs.len(),egui::Button::new("Next ▶")).clicked() { self.selected+=1; }
                });
                // Resolve the selected pair again after navigation.
                let pair = self.pairs[self.selected].clone();
                self.previews.retain(|(p,_)| p == &pair.a || p == &pair.b);
                self.preview(ctx,&pair.a); self.preview(ctx,&pair.b);
                let mut deletion = None;
                let image_height = (ui.available_height()*0.53).clamp(140.0,420.0);
                ui.columns(2, |columns| {
                    for (column,path) in columns.iter_mut().zip([&pair.a,&pair.b]) {
                        column.label(path.file_name().unwrap_or_default().to_string_lossy()).on_hover_text(path.display().to_string());
                        if let Some((_, preview)) = self.previews.iter().find(|(p,_)| p == path) {
                            if let Ok((_, details)) = preview {
                                column.small(details);
                            }
                            column.vertical_centered(|ui| {
                                match preview {
                                    Ok((texture, _)) => {
                                        ui.add(egui::Image::new(texture).fit_to_exact_size(egui::vec2(ui.available_width(),image_height)).maintain_aspect_ratio(true));
                                    },
                                    Err(error) => { ui.colored_label(Color32::LIGHT_RED,error); }
                                }
                                if ui.add_enabled(!self.busy,egui::Button::new("Delete")).clicked() { deletion = Some(path.clone()); }
                            });
                        }
                    }
                });
                if let Some(path) = deletion { self.delete(path); }
                ui.separator();
                egui::ScrollArea::vertical().auto_shrink([false,false]).show_rows(ui,22.0,self.pairs.len(),|ui,rows| {
                    for index in rows {
                        let pair = &self.pairs[index];
                        let label = format!("{:.3}    {}    ↔    {}",pair.score,pair.a.file_name().unwrap_or_default().to_string_lossy(),pair.b.file_name().unwrap_or_default().to_string_lossy());
                        if ui.selectable_label(index==self.selected,label).clicked() && !self.busy { self.selected = index; }
                    }
                });
            }
        });
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
