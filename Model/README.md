# DINOv2 model

`model_fp16.onnx` and `preprocessor_config.json` were copied from the working Blazor application in `DuplicatePhotoDetect2/Model`.

Original model download used by that application:
https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model_fp16.onnx

The application uses the `pixel_values` input and the CLS token of `last_hidden_state`, producing normalized 384-dimensional embeddings. Preprocessing is implemented in `src/engine.rs`; `preprocessor_config.json` is included as reference and isn't loaded at runtime.

Keep this model in the repository. `package.ps1` copies it into the release's `Model` folder; users can choose a different model path in the app, but its input/output contract must match DINOv2 small.
