<?php

namespace App\Jobs;

use App\Models\Track;
use App\Models\QueueTask;
use App\Services\MLQueueService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;

/**
 * TrackAnalysisJob
 * 
 * Queues a track for ML analysis by the Python worker.
 * Requirements: 1.4, 13.2
 */
class TrackAnalysisJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $tries = 3;
    public $backoff = 60;

    public function __construct(
        public Track $track
    ) {
        $this->onQueue('analysis');
    }

    public function handle(): void
    {
        // Track stays in 'pending' status until ML worker picks it up
        // ML worker will update status to 'processing' when it starts

        // Create queue task record for monitoring
        $queueTask = QueueTask::create([
            'task_type' => QueueTask::TYPE_ANALYSIS,
            'payload' => [
                'track_id' => $this->track->id,
                'audio_path' => $this->track->file_path,
            ],
            'status' => QueueTask::STATUS_PENDING,
        ]);

        // Push to ML worker queue using centralized service
        MLQueueService::pushTrackAnalysis(
            $queueTask->id,
            $this->track->id,
            '/app/storage/' . $this->track->file_path
        );
    }

    public function failed(\Throwable $exception): void
    {
        $this->track->update(['analysis_status' => 'failed']);
    }
}
