param(
    [string]$RuntimeDirectory,
    [string]$ModelPath = (Join-Path $PSScriptRoot '../Model/model_fp16.onnx'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'dist')
)
$ErrorActionPreference = 'Stop'
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (!$cargo) { $cargoPath = Join-Path $env:USERPROFILE '.cargo/bin/cargo.exe' } else { $cargoPath = $cargo.Source }
& $cargoPath build --manifest-path (Join-Path $PSScriptRoot 'Cargo.toml') --release --locked
if ($LASTEXITCODE -ne 0) { throw 'Rust release build failed' }
if (!$RuntimeDirectory) {
    $candidates = @(
        (Join-Path $PSScriptRoot '../bin/Release/net8.0/win-x64/publish'),
        (Join-Path $PSScriptRoot '../bin/Release/net8.0/win-x64'),
        (Join-Path $env:USERPROFILE '.nuget/packages/microsoft.ml.onnxruntime.directml/1.24.4/runtimes/win-x64/native')
    )
    $RuntimeDirectory = $candidates | Where-Object { Test-Path (Join-Path $_ 'onnxruntime.dll') } | Select-Object -First 1
}
if (!$RuntimeDirectory) { throw 'Pass -RuntimeDirectory pointing to the ONNX Runtime 1.24.4 DirectML native DLLs.' }
$destination = Join-Path $OutputDirectory 'DuplicatePhotoDetect'
New-Item -ItemType Directory -Force (Join-Path $destination 'Model') | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'target/release/duplicate-photo-detect.exe') -Destination $destination
foreach ($name in @('onnxruntime.dll', 'onnxruntime_providers_shared.dll', 'DirectML.dll')) {
    $source = Join-Path $RuntimeDirectory $name
    if (!(Test-Path $source)) {
        $source = Join-Path $env:USERPROFILE ".nuget/packages/microsoft.ai.directml/1.15.4/bin/x64-win/$name"
    }
    if (!(Test-Path $source)) { throw "Missing $name. Pass a complete DirectML runtime directory." }
    Copy-Item -LiteralPath $source -Destination $destination
}
Copy-Item -LiteralPath $ModelPath -Destination (Join-Path $destination 'Model/model_fp16.onnx')
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'README.md') -Destination $destination
Compress-Archive -Path (Join-Path $destination '*') -DestinationPath (Join-Path $OutputDirectory 'DuplicatePhotoDetect-win-x64.zip') -Force
Write-Host "Release: $destination"
