using Microsoft.ML.OnnxRuntime;

namespace DuplicatePhotoDetect.Services;

/// <summary>
/// Compares photos and finds duplicates with DINOv2 embeddings, caching embeddings so repeat scans are fast.
/// The model is loaded the first time it's used, and requests run one at a time off the UI thread
/// (the embedding cache isn't thread safe).
/// </summary>
public class DuplicatePhotoService(DuplicatePhotoSettings settings, ILogger<DuplicatePhotoService> logger)
{
    public const string ModelUrl = "https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model_fp16.onnx";

    private readonly SemaphoreSlim _lock = new(1, 1);
    private readonly Dictionary<string, float[]?> _embeddingCache = new();
    private ImageEmbedder? _embedder;

    /// <summary>
    /// Where the model runs, e.g. "GPU (DirectML adapter 1)" or "CPU". Null until the model is loaded
    /// </summary>
    public string? Device { get; private set; }

    public Task<List<(float Similarity, string A, string B)>> FindDuplicatesAsync(IReadOnlyList<string> files, float threshold, IProgress<string> progress)
    {
        return RunAsync(embedder =>
        {
            // Embed the images not already cached in batches, which is much faster on the GPU than one at a time
            var uncached = files.Where(o => !_embeddingCache.ContainsKey(o)).ToList();

            for (int batchStart = 0; batchStart < uncached.Count; batchStart += ImageEmbedder.BatchSize)
            {
                var batch = uncached.Skip(batchStart).Take(ImageEmbedder.BatchSize).ToList();
                progress.Report($"Analysing images {batchStart + 1} - {batchStart + batch.Count} of {uncached.Count}");

                // Files that aren't valid images come back as null and are skipped when comparing
                var batchEmbeddings = embedder.EmbedBatch(batch);
                for (int i = 0; i < batch.Count; i++)
                    _embeddingCache[batch[i]] = batchEmbeddings[i];
            }

            var embeddings = files.Select(o => _embeddingCache[o]).ToList();

            progress.Report("Comparing images...");

            var pairs = new List<(float Similarity, string A, string B)>();

            for (int i = 0; i < files.Count; i++)
            {
                for (int j = i + 1; j < files.Count; j++)
                {
                    if (embeddings[i] == null || embeddings[j] == null)
                        continue;

                    var similarity = ImageEmbedder.Similarity(embeddings[i]!, embeddings[j]!);
                    if (similarity >= threshold)
                        pairs.Add((similarity, files[i], files[j]));
                }
            }

            return pairs.OrderByDescending(o => o.Similarity).ToList();
        });
    }

    /// <summary>
    /// Moves the photo to the Recycle Bin and forgets its embedding
    /// </summary>
    public async Task DeleteAsync(string path)
    {
        // Under the lock, so it can't happen in the middle of a scan using the cache
        await _lock.WaitAsync();
        try
        {
            RecycleBin.Delete(path);
            _embeddingCache.Remove(path);
        }
        finally
        {
            _lock.Release();
        }
    }

    /// <summary>
    /// Rough guide based on testing with listing photos, calibrate on your own data
    /// </summary>
    public static string Describe(float similarity)
    {
        if (similarity >= 0.95f)
            return "Duplicate (same photo, possibly resized, cropped or edited)";

        if (similarity >= 0.85f)
            return "Very similar (probably the same scene from a slightly different shot)";

        if (similarity >= 0.6f)
            return "Similar (same kind of room or view)";

        return "Different";
    }

    private async Task<TResult> RunAsync<TResult>(Func<ImageEmbedder, TResult> work)
    {
        await _lock.WaitAsync();
        try
        {
            // Loaded here rather than with Lazy<T>, so a failed load (e.g. a missing model file) is retried next time
            return await Task.Run(() => work(_embedder ??= new ImageEmbedder(CreateSession())));
        }
        finally
        {
            _lock.Release();
        }
    }

    private InferenceSession CreateSession()
    {
        if (!File.Exists(settings.ModelPath))
        {
            throw new FileNotFoundException(
                $"The DINOv2 model wasn't found at {Path.GetFullPath(settings.ModelPath)}. Download {ModelUrl} " +
                $"to Models\\model_fp16.onnx, or set {DuplicatePhotoSettings.SectionName}:ModelPath in appsettings.json.");
        }

        if (settings.UseGpu)
        {
            var deviceId = settings.GpuDeviceId;
            try
            {
                using var options = new Microsoft.ML.OnnxRuntime.SessionOptions();
                options.EnableMemoryPattern = false;
                options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
                options.AppendExecutionProvider_DML(deviceId);

                var session = new InferenceSession(settings.ModelPath, options);
                Device = $"GPU (DirectML adapter {deviceId})";
                return session;
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "Couldn't use DirectML adapter {DeviceId}, using the CPU instead", deviceId);
            }
        }

        // ORT_ENABLE_ALL (the default) fails to load the fp16 model on the CPU in ONNX Runtime 1.24
        // ("Attempting to get index by a name which does not exist: InsertedPrecisionFreeCast_..."), EXTENDED works
        using var cpuOptions = new Microsoft.ML.OnnxRuntime.SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED };
        var cpuSession = new InferenceSession(settings.ModelPath, cpuOptions);
        Device = "CPU";
        return cpuSession;
    }
}
