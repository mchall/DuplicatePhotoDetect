param(
    [string]$RuntimeDirectory = (Join-Path $PSScriptRoot 'Runtime/win-x64'),
    [string]$ModelPath = (Join-Path $PSScriptRoot 'Model/model_fp16.onnx'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'dist')
)
$ErrorActionPreference = 'Stop'
foreach ($name in @('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll')) {
    if (!(Test-Path -LiteralPath (Join-Path $RuntimeDirectory $name) -PathType Leaf)) { throw "Missing $name in $RuntimeDirectory" }
}
if (!(Test-Path -LiteralPath $ModelPath -PathType Leaf)) { throw "Missing model: $ModelPath" }
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (!$cargo) { $cargoPath = Join-Path $env:USERPROFILE '.cargo/bin/cargo.exe' } else { $cargoPath = $cargo.Source }
& $cargoPath build --manifest-path (Join-Path $PSScriptRoot 'Cargo.toml') --release --locked
if ($LASTEXITCODE -ne 0) { throw 'Rust release build failed' }
$destination = Join-Path $OutputDirectory 'DuplicatePhotoDetect'
New-Item -ItemType Directory -Force (Join-Path $destination 'Model') | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'target/release/duplicate-photo-detect.exe') -Destination $destination
foreach ($name in @('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll')) {
    $source = Join-Path $RuntimeDirectory $name
    if (!(Test-Path $source)) { throw "Missing $name. Pass a complete DirectML runtime directory." }
    Copy-Item -LiteralPath $source -Destination $destination
}
Copy-Item -LiteralPath $ModelPath -Destination (Join-Path $destination 'Model/model_fp16.onnx')
$preprocessor = Join-Path (Split-Path -Parent $ModelPath) 'preprocessor_config.json'
if (Test-Path -LiteralPath $preprocessor) { Copy-Item -LiteralPath $preprocessor -Destination (Join-Path $destination 'Model') }
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Runtime/licenses') -Destination $destination -Recurse -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'LICENSE') -Destination $destination
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination $destination
Compress-Archive -Path (Join-Path $destination '*') -DestinationPath (Join-Path $OutputDirectory 'DuplicatePhotoDetect-win-x64.zip') -Force
Write-Host "Release: $destination"
