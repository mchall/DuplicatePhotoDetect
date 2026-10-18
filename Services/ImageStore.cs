using System.Collections.Concurrent;

namespace DuplicatePhotoDetect.Services;

/// <summary>
/// Gives images on the server a URL so the browser can show them.
/// Only files registered here can be served, by id rather than by path.
/// </summary>
public class ImageStore
{
    public static readonly string[] ImageExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp" };

    private readonly ConcurrentDictionary<string, string> _pathsById = new();
    private readonly ConcurrentDictionary<string, string> _idsByPath = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// A URL the browser can use to show the image file
    /// </summary>
    public string GetUrl(string path)
    {
        var id = _idsByPath.GetOrAdd(Path.GetFullPath(path), fullPath =>
        {
            var newId = Guid.NewGuid().ToString("N");
            _pathsById[newId] = fullPath;
            return newId;
        });

        return $"images/{id}{Path.GetExtension(path).ToLowerInvariant()}";
    }

    public string? GetPath(string id)
    {
        return _pathsById.TryGetValue(Path.GetFileNameWithoutExtension(id), out var path) ? path : null;
    }

    /// <summary>
    /// All the images in the folder
    /// </summary>
    public static List<string> FindImages(string folder)
    {
        return Directory.EnumerateFiles(folder, "*", SearchOption.TopDirectoryOnly)
            .Where(o => ImageExtensions.Contains(Path.GetExtension(o).ToLowerInvariant()))
            .ToList();
    }
}
