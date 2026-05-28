<?php

namespace App\Console\Commands;

use App\Jobs\TrackAnalysisJob;
use App\Models\Track;
use Illuminate\Console\Command;

class AnalyzePendingTracks extends Command
{
    protected $signature = 'tracks:analyze-pending {--limit=50}';
    protected $description = 'Queue pending tracks for ML analysis';

    public function handle(): int
    {
        $limit = (int) $this->option('limit');
        
        $tracks = Track::where('analysis_status', 'pending')
            ->limit($limit)
            ->get();

        $this->info("Found {$tracks->count()} pending tracks");

        foreach ($tracks as $track) {
            TrackAnalysisJob::dispatch($track);
            $this->line("Queued: {$track->id} - {$track->title}");
        }

        $this->info("Done! Queued {$tracks->count()} tracks for analysis.");
        
        return 0;
    }
}
