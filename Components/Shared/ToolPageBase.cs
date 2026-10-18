using Microsoft.AspNetCore.Components;
using DuplicatePhotoDetect.Services;
using System.Diagnostics;

namespace DuplicatePhotoDetect.Components.Shared;

/// <summary>
/// Busy / error / elapsed time handling for a page that runs the model
/// </summary>
public abstract class ToolPageBase : ComponentBase
{
    private Stopwatch? _stopwatch;

    protected bool Busy { get; private set; }
    protected string? Error { get; set; }
    protected string? Status { get; set; }
    protected string? Elapsed { get; set; }

    /// <summary>
    /// Runs the work, showing it's busy, how long it took, and any error
    /// </summary>
    protected async Task RunAsync(string busyStatus, Func<Task> work)
    {
        Busy = true;
        Error = null;
        Status = busyStatus;
        Elapsed = null;
        _stopwatch = Stopwatch.StartNew();
        StateHasChanged();

        try
        {
            await work();
            Elapsed = Format.Elapsed(_stopwatch.Elapsed);
        }
        catch (Exception ex)
        {
            Status = null;
            Error = ex.Message;
        }
        finally
        {
            Busy = false;
        }
    }

    /// <summary>
    /// Progress for long running work: updates the status and elapsed time on the page
    /// </summary>
    protected IProgress<string> CreateProgress()
    {
        // Progress posts back to the page's synchronization context, so it's safe to re-render here
        return new Progress<string>(message =>
        {
            if (!Busy)
                return;

            Status = message;
            Elapsed = _stopwatch == null ? null : Format.Elapsed(_stopwatch.Elapsed);
            StateHasChanged();
        });
    }
}
