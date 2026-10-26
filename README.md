# Duplicate Photo Detect — Rust desktop version

Native Windows desktop application built with egui/eframe and the `ort` ONNX Runtime binding. The Blazor application remains separate and unchanged by this port.

## Build and package

Install stable Rust (MSVC target) and Visual Studio C++ build tools, then run from this directory:

```powershell
cargo test --locked
./package.ps1
```

The script reuses the existing .NET application's ONNX Runtime 1.24.4 DirectML libraries and `../Model/model_fp16.onnx`. If necessary, pass `-RuntimeDirectory` and `-ModelPath` explicitly. The runtime directory must contain `onnxruntime.dll`, `onnxruntime_providers_shared.dll`, and `DirectML.dll`. The Rust `ort` dependency is pinned to a version supporting this runtime; do not independently upgrade the binding or DLLs without testing compatibility.

Distribute `dist/DuplicatePhotoDetect-win-x64.zip`. Extract all files before launching `duplicate-photo-detect.exe`. No .NET installation, browser, local server, or wwwroot is needed. Windows may require the Microsoft Visual C++ 2015–2022 x64 Redistributable.

For development, copy the three runtime DLLs beside `target/debug/duplicate-photo-detect.exe`, or set `ORT_DYLIB_PATH` to the full runtime DLL path. The model is located relative to the executable or working directory and their parents, or selected through Model settings.

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

Scans currently compare all image pairs (quadratic in image count), as in the Blazor implementation. The port does not promise faster inference than the .NET app because both use ONNX Runtime.
