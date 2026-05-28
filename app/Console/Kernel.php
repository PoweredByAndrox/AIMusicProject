<?php

namespace App\Console;

use App\Jobs\TrackAnalysisJob;
use App\Models\Track;
use App\Support\AppStoragePath;
use Illuminate\Console\Scheduling\Schedule;
use Illuminate\Foundation\Console\Kernel as ConsoleKernel;

class Kernel extends ConsoleKernel
{
    /**
     * Define the application's command schedule.
     */
    protected function schedule(Schedule $schedule): void
    {
        // Check for pending tracks every minute and queue them for analysis
        $schedule->call(function () {
            $pendingTracks = Track::where('analysis_status', 'pending')
                ->whereNotNull('file_path')
                ->take(10)
                ->get();
            
            foreach ($pendingTracks as $track) {
                TrackAnalysisJob::dispatch($track);
            }
        })->everyMinute()->name('queue-pending-tracks')->withoutOverlapping();

        // Batch-enqueue segment analysis for the library (in small chunks).
        // This is intentionally conservative to avoid GPU overload.
        $schedule->command('segments:enqueue-missing --limit=20')
            ->everyFiveMinutes()
            ->name('enqueue-missing-segments')
            ->withoutOverlapping();

        // Periodic coverage snapshot for monitoring segment embedding fill.
        $schedule->command('segments:coverage --json')
            ->hourly()
            ->name('segments-coverage')
            ->withoutOverlapping()
            ->appendOutputTo(AppStoragePath::log('segments_coverage.log'));

        // Backfill embeddings for existing segments (pattern_examples) in small batches.
        // This avoids re-segmenting tracks and preserves user feedback.
        $schedule->command('segments:backfill-embeddings --limit=200 --chunk=50')
            ->everyFiveMinutes()
            ->name('backfill-segment-embeddings')
            ->withoutOverlapping()
            ->appendOutputTo(AppStoragePath::log('segments_backfill.log'));
    }

    /**
     * Register the commands for the application.
     */
    protected function commands(): void
    {
        $this->load(__DIR__.'/Commands');

        require base_path('routes/console.php');
    }
}
