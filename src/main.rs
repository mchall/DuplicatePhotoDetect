mod engine;
mod ui;

fn main() -> anyhow::Result<()> {
    if std::env::args().any(|a| a == "--smoke-test") {
        return engine::smoke_test();
    }
    eframe::run_native(
        "Duplicate Photo Detect",
        eframe::NativeOptions {
            viewport: eframe::egui::ViewportBuilder::default().with_inner_size([1200.0, 850.0]),
            ..Default::default()
        },
        Box::new(|_| Ok(Box::new(ui::App::default()))),
    )
    .map_err(|e| anyhow::anyhow!(e.to_string()))
}
