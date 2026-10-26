# Duplicate Photo Detect

Native Windows desktop application built with egui/eframe and the `ort` ONNX Runtime binding. 

## Features

- Folder picker, adjustable threshold, optional subfolder scanning (off by default, matching the current Blazor app).
- DirectML GPU adapter selection with CPU fallback.
- Batch inference (32 images), parallel image preprocessing, in-memory embeddings cached by path, size, and modification date.
- Side-by-side thumbnails with dimensions, size, and filesystem creation date.
- Previous/next pair navigation and virtualized pair list.
- Confirmed Recycle Bin deletion, no success notification or full-size preview.
- Background scanning, progress, elapsed time, and cooperative cancellation between batches and comparison rows. An inference already in progress finishes before cancellation takes effect.

The same DINOv2 preprocessing and normalized CLS similarity are used: resize shortest edge to 256, center crop 224×224, ImageNet RGB normalization, and cosine similarity. Image decoders and resampling differ from SkiaSharp, so scores can vary slightly; thresholds should be checked against representative photos.

## Verification

`cargo test --locked` checks preprocessing. After packaging, run:

```powershell
./dist/DuplicatePhotoDetect/duplicate-photo-detect.exe --smoke-test
```

This processes a generated image through the real model on CPU and requested DirectML adapter 0 (with fallback), checks normalized 384-dimensional embeddings and identical-image similarity, and prints the actual selected device. It does not delete photos.