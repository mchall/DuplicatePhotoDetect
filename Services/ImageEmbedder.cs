using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace DuplicatePhotoDetect.Services;

/// <summary>
/// Turns an image into an embedding (a vector of numbers describing what it looks like) using DINOv2.
/// Similar looking images have embeddings that point in a similar direction.
/// </summary>
public class ImageEmbedder
{
    /// <summary>
    /// Images run through the model together. Each model run has a fixed overhead on the GPU,
    /// so batches of 32 were ~5x faster than one image at a time (~10x with the fp16 model)
    /// </summary>
    public const int BatchSize = 32;

    private const int ResizeShortestEdge = 256;
    private const int CropSize = 224;
    private const int ImageLength = 3 * CropSize * CropSize;
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std = { 0.229f, 0.224f, 0.225f };

    private readonly InferenceSession _session;

    public ImageEmbedder(InferenceSession session)
    {
        _session = session;
    }

    /// <summary>
    /// Returns the image's embedding, normalized to length 1 so that the dot product of two embeddings is their cosine similarity
    /// </summary>
    public float[] Embed(string imageLocation)
    {
        return EmbedBatch(new[] { imageLocation })[0]
            ?? throw new InvalidOperationException($"Could not load image: {imageLocation}");
    }

    /// <summary>
    /// Embeds several images, running them through the model in batches.
    /// Returns null for any file that couldn't be loaded as an image.
    /// </summary>
    public float[]?[] EmbedBatch(IReadOnlyList<string> imageLocations)
    {
        var embeddings = new float[]?[imageLocations.Count];

        for (int batchStart = 0; batchStart < imageLocations.Count; batchStart += BatchSize)
        {
            var batchCount = Math.Min(BatchSize, imageLocations.Count - batchStart);

            // Load and resize the images in parallel (CPU work), skipping any that fail
            var pixels = new float[]?[batchCount];
            Parallel.For(0, batchCount, i =>
            {
                try
                {
                    pixels[i] = Preprocess(imageLocations[batchStart + i]);
                }
                catch (Exception)
                {
                    pixels[i] = null;
                }
            });

            var validIndexes = Enumerable.Range(0, batchCount).Where(i => pixels[i] != null).ToList();
            if (validIndexes.Count == 0)
                continue;

            // Stack the images into one [batch, 3, 224, 224] tensor
            var buffer = new float[validIndexes.Count * ImageLength];
            for (int i = 0; i < validIndexes.Count; i++)
                pixels[validIndexes[i]]!.CopyTo(buffer, i * ImageLength);

            var input = new DenseTensor<float>(buffer, new[] { validIndexes.Count, 3, CropSize, CropSize });
            var inputName = _session.InputMetadata.Keys.First(); // "pixel_values"
            using var results = _session.Run(new[] { NamedOnnxValue.CreateFromTensor(inputName, input) });

            // last_hidden_state is [batch, tokens, 384]. The first token is the CLS token that summarizes the whole image
            // (the same as the "pooler_output" the Hugging Face model returns)
            var hiddenState = results.First(o => o.Name == "last_hidden_state").AsTensor<float>();

            for (int i = 0; i < validIndexes.Count; i++)
            {
                var embedding = Enumerable.Range(0, hiddenState.Dimensions[2])
                    .Select(k => hiddenState[i, 0, k])
                    .ToArray();

                embeddings[batchStart + validIndexes[i]] = Normalize(embedding);
            }
        }

        return embeddings;
    }

    /// <summary>
    /// Cosine similarity of two normalized embeddings: 1 = identical, 0 = unrelated
    /// </summary>
    public static float Similarity(float[] a, float[] b)
    {
        float dot = 0;
        for (int i = 0; i < a.Length; i++)
            dot += a[i] * b[i];

        return dot;
    }

    /// <summary>
    /// Loads the image as normalized pixel values in [3, 224, 224] order
    /// </summary>
    private static float[] Preprocess(string imageLocation)
    {
        using var img = Image.Load<Rgb24>(imageLocation);

        // Resize so the shortest edge is 256 (up or down), then take the centre 224x224 (rather than stretching the image)
        var scale = (float)ResizeShortestEdge / Math.Min(img.Width, img.Height);
        var width = Math.Max(ResizeShortestEdge, (int)MathF.Round(img.Width * scale));
        var height = Math.Max(ResizeShortestEdge, (int)MathF.Round(img.Height * scale));
        img.Mutate(x => x.Resize(width, height, KnownResamplers.Bicubic));
        img.Mutate(x => x.Crop(new Rectangle((img.Width - CropSize) / 2, (img.Height - CropSize) / 2, CropSize, CropSize)));

        const int channelLength = CropSize * CropSize;
        var pixels = new float[ImageLength];
        img.ProcessPixelRows(rows =>
        {
            for (int y = 0; y < CropSize; y++)
            {
                var row = rows.GetRowSpan(y);
                for (int x = 0; x < CropSize; x++)
                {
                    var offset = y * CropSize + x;
                    pixels[offset] = (row[x].R / 255f - Mean[0]) / Std[0];
                    pixels[channelLength + offset] = (row[x].G / 255f - Mean[1]) / Std[1];
                    pixels[2 * channelLength + offset] = (row[x].B / 255f - Mean[2]) / Std[2];
                }
            }
        });

        return pixels;
    }

    private static float[] Normalize(float[] embedding)
    {
        var length = MathF.Sqrt(embedding.Sum(o => o * o));
        return embedding.Select(o => o / length).ToArray();
    }
}
