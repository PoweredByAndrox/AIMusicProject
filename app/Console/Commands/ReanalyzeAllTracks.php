<?php

namespace App\Console\Commands;

use App\Jobs\TrackAnalysisJob;
use App\Models\Track;
use Illuminate\Console\Command;

class ReanalyzeAllTracks extends Command
{
    protected $signature = 'tracks:reanalyze 
                            {--all : Reanalyze all completed tracks}
                            {--pending : Queue all pending tracks (without resetting status)}
                            {--failed : Reanalyze all failed tracks}
                            {--default-only : Only reanalyze tracks with default values (BPM=120, Key=C)}
                            {--limit=0 : Limit number of tracks (0 = no limit)}
                            {--force : Skip confirmation}';
    
    protected $description = 'Queue tracks for re-analysis (reset status and reanalyze)';

    public function handle(): int
    {
        $all = $this->option('all');
        $pending = $this->option('pending');
        $failed = $this->option('failed');
        $defaultOnly = $this->option('default-only');
        $limit = (int) $this->option('limit');

        if (!$all && !$defaultOnly && !$pending && !$failed) {
            $this->error('Please specify --all, --pending, --failed, or --default-only');
            return 1;
        }

        $query = Track::query();

        if ($pending) {
            $this->info('Finding all pending tracks...');
            $query->where('analysis_status', 'pending');
        } elseif ($failed) {
            $this->info('Finding all failed tracks...');
            $query->where('analysis_status', 'failed');
        } elseif ($defaultOnly) {
            // Tracks with default/fallback values (likely failed analysis)
            $query->where(function ($q) {
                $q->where('bpm', 120)
                  ->where('key', 'C')
                  ->where('genre', 'Electronic');
            });
            $this->info('Finding tracks with default values (BPM=120, Key=C, Genre=Electronic)...');
        } else {
            $this->info('Finding all completed tracks...');
            $query->where('analysis_status', 'completed');
        }

        if ($limit > 0) {
            $query->limit($limit);
        }

        $tracks = $query->get();

        if ($tracks->isEmpty()) {
            $this->info('No tracks found to reanalyze.');
            return 0;
        }

        $this->info("Found {$tracks->count()} tracks to reanalyze");

        if (!$this->option('force') && !$this->confirm("Queue {$tracks->count()} tracks for reanalysis?")) {
            $this->info('Cancelled.');
            return 0;
        }

        $bar = $this->output->createProgressBar($tracks->count());
        $bar->start();

        foreach ($tracks as $track) {
            // Reset analysis status to pending (skip for --pending option)
            if (!$pending) {
                $track->update(['analysis_status' => 'pending']);
            }
            
            // Push directly to the ML worker queue from this command.
            // The old async Laravel dispatch path reported success even when
            // no separate app queue worker was processing TrackAnalysisJob.
            TrackAnalysisJob::dispatchSync($track);
            
            $bar->advance();
        }

        $bar->finish();
        $this->newLine();
        $this->info("Done! Queued {$tracks->count()} tracks for reanalysis.");
        
        return 0;
    }
}
