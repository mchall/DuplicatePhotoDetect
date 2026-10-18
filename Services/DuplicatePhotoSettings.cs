namespace DuplicatePhotoDetect.Services;

/// <summary>
/// The "DuplicatePhotoDetect" section of appsettings.json
/// </summary>
public class DuplicatePhotoSettings
{
    public const string SectionName = "DuplicatePhotoDetect";

    /// <summary>
    /// The DINOv2 model file. Leave empty to look for Downloads\DinoV2\model_fp16.onnx in the working directory
    /// (or the app's folder) and their parents
    /// </summary>
    public string ModelPath { get; set; } = string.Empty;

    /// <summary>
    /// Runs the model on the GPU with DirectML, falling back to the CPU if the GPU can't be used
    /// </summary>
    public bool UseGpu { get; set; } = true;

    /// <summary>
    /// The DirectML GPU adapter to use: 0 is usually the integrated GPU and 1 the dedicated one
    /// </summary>
    public int GpuDeviceId { get; set; } = 1;

    /// <summary>
    /// The folder the scan box starts with. Leave empty to use My Pictures folder
    /// </summary>
    public string DefaultFolder { get; set; } = string.Empty;

    /// <summary>
    /// Fills in the paths left empty by searching the working directory, the app's folder and their parents
    /// </summary>
    public void ResolvePaths()
    {
        if (string.IsNullOrWhiteSpace(ModelPath))
            ModelPath = FindUp(Path.Combine("Model", "model_fp16.onnx"), File.Exists) ?? Path.Combine("Model", "model_fp16.onnx");

        if (string.IsNullOrWhiteSpace(DefaultFolder))
            DefaultFolder = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures);
    }

    private static string? FindUp(string relativePath, Func<string, bool> exists)
    {
        foreach (var start in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            for (var folder = new DirectoryInfo(start); folder != null; folder = folder.Parent)
            {
                var path = Path.Combine(folder.FullName, relativePath);
                if (exists(path))
                    return path;
            }
        }

        return null;
    }
}
