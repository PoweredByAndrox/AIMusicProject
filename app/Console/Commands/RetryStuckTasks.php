<?php

namespace App\Console\Commands;

use App\Models\QueueTask;
use App\Models\Track;
use App\Services\MLQueueService;
use Carbon\Carbon;
use Illuminate\Console\Command;

class RetryStuckTasks extends Command
{
    protected $signature = 'queue:retry-stuck
                            {--minutes=5 : Consider tasks stuck if older than X minutes}
                            {--dry-run : Show what would be retried without actually retrying}';

    protected $description = 'Find and retry stuck queue tasks (pending/processing for too long)';

    public function handle(): int
    {
        $minutes = (int) $this->option('minutes');
        $dryRun = $this->option('dry-run');
        $cutoff = Carbon::now()->subMinutes($minutes);

        $stuckTasks = QueueTask::whereIn('status', [QueueTask::STATUS_PENDING, QueueTask::STATUS_PROCESSING])
            ->where('created_at', '<', $cutoff)
            ->get();

        if ($stuckTasks->isEmpty()) {
            $this->info('No stuck tasks found.');
            return 0;
        }

        $this->info("Found {$stuckTasks->count()} stuck tasks (older than {$minutes} minutes)");

        if ($dryRun) {
            $this->table(
                ['ID', 'Type', 'Status', 'Created At', 'Track ID'],
                $stuckTasks->map(fn ($task) => [
                    $task->id,
                    $task->task_type,
                    $task->status,
                    $task->created_at,
                    $task->payload['track_id'] ?? 'N/A',
                ])
            );
            $this->info('Redis queue length: ' . MLQueueService::queueLength());
            return 0;
        }

        $retried = 0;
        $skipped = 0;

        $bar = $this->output->createProgressBar($stuckTasks->count());
        $bar->start();

        foreach ($stuckTasks as $task) {
            $payload = $task->payload;

            if (($task->task_type === 'track_analysis' || $task->task_type === 'analysis') && isset($payload['track_id'])) {
                $track = Track::find($payload['track_id']);

                if (!$track) {
                    $task->update([
                        'status' => QueueTask::STATUS_FAILED,
                        'error_message' => 'Track not found',
                        'completed_at' => now(),
                    ]);
                    $skipped++;
                    $bar->advance();
                    continue;
                }

                if ($track->analysis_status === QueueTask::STATUS_COMPLETED) {
                    $task->update([
                        'status' => QueueTask::STATUS_COMPLETED,
                        'error_message' => null,
                        'completed_at' => now(),
                    ]);
                    $skipped++;
                    $bar->advance();
                    continue;
                }

                $task->update([
                    'status' => QueueTask::STATUS_PENDING,
                    'error_message' => null,
                    'completed_at' => null,
                ]);
                $track->update(['analysis_status' => QueueTask::STATUS_PENDING]);
                $retried++;
            } else {
                $skipped++;
            }

            $bar->advance();
        }

        $bar->finish();
        $this->newLine();
        $this->info("DB reset: retried={$retried}, skipped={$skipped}");

        $analysisPayloads = [];
        QueueTask::where('status', QueueTask::STATUS_PENDING)
            ->whereIn('task_type', ['analysis', 'track_analysis'])
            ->orderBy('created_at')
            ->chunk(500, function ($chunk) use (&$analysisPayloads) {
                foreach ($chunk as $task) {
                    $payload = $task->payload;
                    if (!isset($payload['track_id'])) {
                        continue;
                    }

                    $track = Track::find($payload['track_id']);
                    if (!$track) {
                        continue;
                    }

                    $analysisPayloads[] = [
                        'type' => 'track_analysis',
                        'task_id' => $task->id,
                        'track_id' => $track->id,
                        'audio_path' => '/app/storage/' . $track->file_path,
                        'created_at' => now()->toIso8601String(),
                    ];
                }
            });

        $analysisResult = MLQueueService::reconcilePendingTrackAnalysisTasks($analysisPayloads);
        $this->info(
            'Track analysis reconciliation: prioritized=' . count($analysisResult['prioritized_task_ids'])
            . ', removed_stale=' . count($analysisResult['removed_stale_task_ids'])
            . ', queue_length=' . MLQueueService::queueLength()
        );

        return 0;
    }
}
