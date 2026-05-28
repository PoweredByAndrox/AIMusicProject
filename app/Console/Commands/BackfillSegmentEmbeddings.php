<?php

namespace App\Console\Commands;

use App\Services\MLQueueService;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\DB;

class BackfillSegmentEmbeddings extends Command
{
    protected $signature = 'segments:backfill-embeddings {--limit=200} {--chunk=50} {--dry-run}';
    protected $description = 'Backfill pattern_examples.embedding_path for existing segments (no re-segmentation)';

    public function handle(): int
    {
        $limit = max(0, (int) $this->option('limit'));
        $chunk = max(1, (int) $this->option('chunk'));
        $dryRun = (bool) $this->option('dry-run');

        if ($limit === 0) {
            $this->info('limit=0, nothing to do');
            return 0;
        }

        // Join tracks to avoid repeatedly enqueuing examples whose audio is missing.
        $ids = DB::table('pattern_examples as pe')
            ->join('tracks as t', 't.id', '=', 'pe.track_id')
            ->whereNull('pe.embedding_path')
            ->whereNotNull('pe.start_time')
            ->whereNotNull('pe.end_time')
            ->whereNotNull('t.file_path')
            ->orderBy('pe.id')
            ->limit($limit)
            ->pluck('pe.id')
            ->map(fn ($v) => (int) $v)
            ->toArray();

        if (count($ids) === 0) {
            $this->info('No pattern_examples need embedding backfill.');
            return 0;
        }

        $this->info('Found ' . count($ids) . " pattern_examples without embeddings (enqueue chunk_size={$chunk})");

        $enqueuedTasks = 0;
        foreach (array_chunk($ids, $chunk) as $chunkIds) {
            if ($dryRun) {
                $this->line('Would enqueue backfill for pattern_example_ids=' . implode(',', $chunkIds));
                $enqueuedTasks++;
                continue;
            }

            MLQueueService::push([
                'type' => 'backfill_pattern_example_embeddings',
                'pattern_example_ids' => $chunkIds,
                'created_at' => now()->toIso8601String(),
            ]);

            $this->line('Enqueued backfill for ' . count($chunkIds) . ' pattern_examples');
            $enqueuedTasks++;
        }

        $this->info(($dryRun ? 'Dry run complete.' : 'Done.') . " tasks_enqueued={$enqueuedTasks}");

        return 0;
    }
}
