<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Models\QueueTask;
use App\Services\MLQueueService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Redis;
use Symfony\Component\HttpFoundation\StreamedResponse;

/**
 * QueueStatusController - Server-Sent Events for real-time queue status.
 * 
 * Replaces polling with SSE for efficient real-time updates.
 */
class QueueStatusController extends Controller
{
    /**
     * Enforce admin access even if routes are misconfigured later.
     */
    private function requireAdmin(Request $request): ?JsonResponse
    {
        $user = $request->user();

        if (!$user) {
            return response()->json(['message' => 'Unauthenticated.'], 401);
        }

        if (!method_exists($user, 'isAdmin') || !$user->isAdmin()) {
            return response()->json(['message' => 'Forbidden. Admin access required.'], 403);
        }

        return null;
    }

    /**
     * Get list of failed tasks with error details.
     */
    public function failedTasks(Request $request): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $tasks = QueueTask::where('status', 'failed')
            ->orderBy('created_at', 'desc')
            ->take(100)
            ->get()
            ->map(function ($task) {
                $trackId = $task->payload['track_id'] ?? null;
                $track = $trackId ? Track::find($trackId) : null;
                
                return [
                    'id' => $task->id,
                    'task_type' => $task->task_type,
                    'track_id' => $trackId,
                    'track_title' => $track?->title,
                    'track_artist' => $track?->artist,
                    'error_message' => $task->error_message,
                    'created_at' => $task->created_at->toIso8601String(),
                    'completed_at' => $task->completed_at?->toIso8601String(),
                ];
            });
        
        return response()->json([
            'tasks' => $tasks,
            'total' => QueueTask::where('status', 'failed')->count(),
        ]);
    }
    
    /**
     * Retry a single failed task.
     */
    public function retryTask(Request $request, int $taskId): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $task = QueueTask::findOrFail($taskId);
        
        if ($task->status !== 'failed') {
            return response()->json(['error' => 'Task is not in failed status'], 400);
        }
        
        $trackId = $task->payload['track_id'] ?? null;
        if (!$trackId) {
            return response()->json(['error' => 'Task has no track_id'], 400);
        }
        
        $track = Track::find($trackId);
        if (!$track) {
            return response()->json(['error' => 'Track not found'], 404);
        }
        
        // Reset task status
        $task->update([
            'status' => 'pending',
            'error_message' => null,
            'completed_at' => null,
        ]);
        
        // Reset track status
        $track->update(['analysis_status' => 'pending']);
        
        // Re-push to ML worker queue
        MLQueueService::pushTrackAnalysis(
            $task->id,
            $track->id,
            '/app/storage/' . $track->file_path
        );
        
        return response()->json([
            'success' => true,
            'message' => "Task {$taskId} queued for retry",
        ]);
    }
    
    /**
     * Retry all failed tasks.
     */
    public function retryAllFailed(Request $request): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $failedTasks = QueueTask::where('status', 'failed')
            ->whereNotNull('payload->track_id')
            ->get();
        
        $retried = 0;
        $skipped = 0;
        
        foreach ($failedTasks as $task) {
            $trackId = $task->payload['track_id'] ?? null;
            if (!$trackId) {
                $skipped++;
                continue;
            }
            
            $track = Track::find($trackId);
            if (!$track) {
                $skipped++;
                continue;
            }
            
            // Reset task status
            $task->update([
                'status' => 'pending',
                'error_message' => null,
                'completed_at' => null,
            ]);
            
            // Reset track status
            $track->update(['analysis_status' => 'pending']);
            
            // Re-push to ML worker queue
            MLQueueService::pushTrackAnalysis(
                $task->id,
                $track->id,
                '/app/storage/' . $track->file_path
            );
            
            $retried++;
        }
        
        return response()->json([
            'success' => true,
            'retried' => $retried,
            'skipped' => $skipped,
            'message' => "Retried {$retried} tasks, skipped {$skipped}",
        ]);
    }
    
    /**
     * Delete a failed task (mark track as failed permanently).
     */
    public function deleteTask(Request $request, int $taskId): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $task = QueueTask::findOrFail($taskId);
        $task->delete();
        
        return response()->json([
            'success' => true,
            'message' => "Task {$taskId} deleted",
        ]);
    }
    
    /**
     * Clear all failed tasks.
     */
    public function clearAllFailed(Request $request): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $count = QueueTask::where('status', 'failed')->delete();
        
        return response()->json([
            'success' => true,
            'deleted' => $count,
            'message' => "Deleted {$count} failed tasks",
        ]);
    }
    /**
     * SSE endpoint for queue status updates.
     * 
     * Sends updates every 3 seconds via Server-Sent Events.
     */
    public function stream(Request $request): StreamedResponse|JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        return response()->stream(function () {
            // Disable output buffering
            if (ob_get_level()) {
                ob_end_clean();
            }
            
            // Send initial data immediately
            $this->sendEvent($this->getQueueData());
            
            // Keep connection alive and send updates
            $lastData = null;
            $iterations = 0;
            $maxIterations = 600; // ~30 minutes max connection
            
            while ($iterations < $maxIterations) {
                $iterations++;
                
                // Get current data
                $data = $this->getQueueData();
                
                // Only send if data changed or every 10 iterations (30 sec heartbeat)
                if ($data !== $lastData || $iterations % 10 === 0) {
                    $this->sendEvent($data);
                    $lastData = $data;
                }
                
                // Check if client disconnected
                if (connection_aborted()) {
                    break;
                }
                
                // Wait 3 seconds
                sleep(3);
            }
        }, 200, [
            'Content-Type' => 'text/event-stream',
            'Cache-Control' => 'no-cache',
            'Connection' => 'keep-alive',
            'X-Accel-Buffering' => 'no', // Disable nginx buffering
        ]);
    }
    
    /**
     * Send SSE event.
     */
    private function sendEvent(array $data): void
    {
        echo "data: " . json_encode($data) . "\n\n";
        flush();
    }
    
    /**
     * Get current queue status data.
     */
    private function getQueueData(): array
    {
        // Get Redis queue lengths
        // Laravel queues use prefix + "queues:" + queue_name format
        $youtubeQueue = 0;
        $analysisQueue = 0;
        $segmentQueue = 0;
        $otherWorkerQueue = 0;
        
        try {
            // Laravel queue format: {prefix}queues:{queue_name}
            // The prefix is already applied by Redis facade, so we use raw keys
            $youtubeQueue = Redis::connection()->llen('queues:youtube') ?? 0;

            $mlQueueItems = Redis::connection()->lrange(MLQueueService::QUEUE_NAME, 0, -1) ?? [];
            foreach ($mlQueueItems as $item) {
                $decoded = json_decode($item, true);
                $taskType = is_array($decoded) ? ($decoded['type'] ?? null) : null;

                if ($taskType === 'track_analysis') {
                    $analysisQueue++;
                    continue;
                }

                if ($taskType === 'analyze_segments') {
                    $segmentQueue++;
                    continue;
                }

                $otherWorkerQueue++;
            }
            
            // DEBUG: Log queue data
            \Log::info('QueueStatus SSE data', [
                'youtube_queue' => $youtubeQueue,
                'analysis_queue' => $analysisQueue,
                'segment_queue' => $segmentQueue,
                'other_worker_queue' => $otherWorkerQueue,
            ]);
        } catch (\Exception $e) {
            // Redis may not be available
            \Log::error('QueueStatus Redis error', ['error' => $e->getMessage()]);
        }
        
        // Get track counts by status
        $pending = Track::where('analysis_status', 'pending')->count();
        $processing = Track::where('analysis_status', 'processing')->count();
        $completed = Track::where('analysis_status', 'completed')->count();
        $failed = Track::where('analysis_status', 'failed')->count();
        
        // Get recent tracks
        $recent = Track::orderBy('created_at', 'desc')
            ->take(10)
            ->get(['id', 'title', 'artist', 'analysis_status', 'created_at', 'source'])
            ->map(fn($t) => [
                'id' => $t->id,
                'title' => $t->title,
                'artist' => $t->artist,
                'analysis_status' => $t->analysis_status,
                'created_at' => $t->created_at->toIso8601String(),
                'source' => $t->source,
            ])
            ->toArray();
        
        return [
            'queue' => [
                'youtube' => $youtubeQueue,
                'analysis' => $analysisQueue,
                'segments' => $segmentQueue,
                'other' => $otherWorkerQueue,
                'default' => 0,
                'total' => $youtubeQueue + $analysisQueue + $segmentQueue + $otherWorkerQueue,
            ],
            'tracks' => [
                'pending' => $pending,
                'processing' => $processing,
                'completed' => $completed,
                'failed' => $failed,
            ],
            'recent' => $recent,
            'timestamp' => now()->toIso8601String(),
        ];
    }
}
