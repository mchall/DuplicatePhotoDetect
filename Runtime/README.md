# Bundled Windows x64 runtime

These files were copied from the working release of the previous .NET application, now located in `DuplicatePhotoDetect2`, to make this Rust repository independent of that checkout and its NuGet cache.

| File | Source package |
| --- | --- |
| `win-x64/onnxruntime.dll` | Microsoft.ML.OnnxRuntime.DirectML 1.24.4 |
| `win-x64/onnxruntime_providers_shared.dll` | Microsoft.ML.OnnxRuntime.DirectML 1.24.4 |
| `win-x64/DirectML.dll` | Microsoft.AI.DirectML 1.15.4 |

`build.rs` copies these DLLs beside Cargo-built executables. `package.ps1` includes them in the downloadable release. Keep the source DLLs committed to this repository so a checkout can build without the old project. Runtime overrides for packaged releases are supported through `package.ps1 -RuntimeDirectory`; the running application also accepts `ORT_DYLIB_PATH`.

ONNX Runtime and third-party license notices are in `licenses/`. The `ort` crate is pinned to `2.0.0-rc.10`, supporting the bundled ONNX Runtime API. Test CPU and DirectML using `--smoke-test` after changing runtime versions.
