<?php

namespace App\Jobs;

use App\Models\Fragment;
use App\Models\QueueTask;
use App\Services\MLQueueService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Storage;

/**
 * FragmentAnalysisJob
 * 
 * Queues a fragment for embedding extraction by the Python worker.
 * Requirements: 6.6
 */
class FragmentAnalysisJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $tries = 3;
    public $backoff = 60;

    public function __construct(
        public Fragment $fragment
    ) {
        $this->onQueue('analysis');
    }

    public function handle(): void
    {
        $this->fragment->update(['analysis_status' => 'processing']);

        $track = $this->fragment->track;

        $queueTask = QueueTask::create([
            'task_type' => QueueTask::TYPE_FRAGMENT_ANALYSIS,
            'payload' => [
                'fragment_id' => $this->fragment->id,
                'track_id' => $track->id,
            ],
            'status' => QueueTask::STATUS_PENDING,
        ]);

        MLQueueService::pushFragmentAnalysis(
            $queueTask->id,
            $this->fragment->id,
            $track->id,
            '/app/storage/' . $track->file_path,
            $this->fragment->start_time,
            $this->fragment->end_time
        );
    }

    public function failed(\Throwable $exception): void
    {
        $this->fragment->update(['analysis_status' => 'failed']);
    }
}
