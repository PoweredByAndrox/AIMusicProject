<?php

namespace App\Console\Commands;

use App\Models\Track;
use App\Services\MLQueueService;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Redis;

class EnqueueMissingSegmentAnalysis extends Command
{
    private const MAX_SEGMENT_ANALYSIS_DURATION = 900;

    protected $signature = 'segments:enqueue-missing {--limit=25} {--dry-run}';
    protected $description = 'Enqueue analyze_segments tasks for analyzed tracks that do not have any segments yet';

    public function handle(): int
    {
        $limit = max(0, (int) $this->option('limit'));
        $dryRun = (bool) $this->option('dry-run');

        if ($limit === 0) {
            $this->info('limit=0, nothing to do');
            return 0;
        }

        // Only enqueue for tracks that are already analyzed and have embeddings.
        // We intentionally target tracks with NO pattern_examples to avoid re-running
        // segmentation on tracks that may already have user feedback.
        $candidates = Track::query()
            ->analyzed()
            ->whereNotNull('embedding_path')
            ->whereNotNull('file_path')
            ->where('duration', '>=', 30)
            ->where('duration', '<=', self::MAX_SEGMENT_ANALYSIS_DURATION)
            ->doesntHave('patternExamples')
            ->orderByRaw('CASE WHEN preference_score IS NULL THEN 1 ELSE 0 END ASC')
            ->orderByDesc('preference_score')
            ->orderBy('id')
            ->limit($limit * 10)
            ->get(['id', 'title', 'artist']);

        $this->info("Found {$candidates->count()} candidate tracks (will enqueue up to {$limit})");

        $enqueued = 0;
        $skippedQueued = 0;

        foreach ($candidates as $track) {
            if ($enqueued >= $limit) {
                break;
            }

            $statusKey = "segment_analysis_status:{$track->id}";
            $alreadyQueued = false;
            try {
                $alreadyQueued = (int) Redis::connection()->exists($statusKey) > 0;
            } catch (\Throwable $e) {
                $alreadyQueued = false;
            }

            if ($alreadyQueued) {
                $skippedQueued++;
                continue;
            }

            if (!$dryRun) {
                $taskId = MLQueueService::pushSegmentAnalysis((int) $track->id);
                $this->line("Enqueued track {$track->id}: {$track->artist} - {$track->title} (task_id={$taskId})");
            } else {
                $this->line("Would enqueue track {$track->id}: {$track->artist} - {$track->title}");
            }

            $enqueued++;
        }

        $this->info(
            ($dryRun ? 'Dry run complete.' : 'Done.') .
            " enqueued={$enqueued}, skipped_already_queued={$skippedQueued}"
        );

        return 0;
    }
}
