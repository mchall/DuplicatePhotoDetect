using DuplicatePhotoDetect.Components;
using DuplicatePhotoDetect.Services;
using Microsoft.AspNetCore.StaticFiles;
using MudBlazor.Services;
using System.Diagnostics;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddMudServices();

var settings = builder.Configuration.GetSection(DuplicatePhotoSettings.SectionName).Get<DuplicatePhotoSettings>() ?? new DuplicatePhotoSettings();
settings.ResolvePaths();

builder.Services.AddSingleton(settings);
builder.Services.AddSingleton<ImageStore>();
builder.Services.AddSingleton<DuplicatePhotoService>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
    app.UseExceptionHandler("/Error", createScopeForErrors: true);

app.UseStaticFiles();
app.UseAntiforgery();

// Serves images from scanned folders (only files registered with the ImageStore)
var contentTypes = new FileExtensionContentTypeProvider();
app.MapGet("/images/{id}", (string id, ImageStore store) =>
{
    var path = store.GetPath(id);
    if (path == null || !File.Exists(path))
        return Results.NotFound();

    return Results.File(path, contentTypes.TryGetContentType(path, out var contentType) ? contentType : "application/octet-stream");
});

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Logger.LogInformation("DINOv2 model: {ModelPath}", Path.GetFullPath(settings.ModelPath));

await app.StartAsync();

try
{
    Process.Start(new ProcessStartInfo
    {
        FileName = app.Urls.First(),
        UseShellExecute = true
    });
}
catch (Exception ex)
{
    app.Logger.LogWarning(ex, "Could not open the browser automatically.");
}

await app.WaitForShutdownAsync();
