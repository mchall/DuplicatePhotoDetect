namespace DuplicatePhotoDetect.Services;

public static class Format
{
    public static string Elapsed(TimeSpan elapsed)
    {
        return elapsed.TotalSeconds < 1 ? $"{elapsed.TotalMilliseconds:0} ms" : $"{elapsed.TotalSeconds:0.0} s";
    }
}
