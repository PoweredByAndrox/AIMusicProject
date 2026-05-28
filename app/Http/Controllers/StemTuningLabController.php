<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Models\QueueTask;
use App\Services\StemSourceBuildService;
use App\Services\StemTuningLabRegistry;
use App\Services\StemTuningRenderService;
use App\Services\StemTuningWaveformService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Redis;
use RuntimeException;

class StemTuningLabController extends Controller
{
    public function __construct(
        private readonly StemTuningLabRegistry $registry,
        private readonly StemSourceBuildService $stemBuildService,
        private readonly StemTuningRenderService $renderService,
        private readonly StemTuningWaveformService $waveforms,
    ) {}

    public function show(Track $track): JsonResponse
    {
        $session = $this->registry->sessionForTrack($track);
        if ($session === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'No stem sources were found for this track yet. Build stems first, then open the tuning lab.',
                'buildable' => $this->stemBuildService->canBuildTrack($track),
            ], 404);
        }

        return response()->json([
            'track' => $track->toArray(),
            'lab' => $session,
        ]);
    }

    public function sourceWaveform(Track $track, string $sourceKey): JsonResponse
    {
        $source = $this->registry->resolveSource($track, $sourceKey);
        if ($source === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Tuning source not found',
            ], 404);
        }

        try {
            $payload = $this->waveforms->getWaveformPayload(
                'track_' . $track->id,
                $sourceKey,
                $source['path'],
                (float) $track->duration,
            );
        } catch (RuntimeException $e) {
            return response()->json([
                'error' => 'Waveform unavailable',
                'message' => $e->getMessage(),
            ], 500);
        }

        return response()->json($payload, 200, [
            'Cache-Control' => 'public, max-age=86400',
        ]);
    }

    public function sourceStream(Request $request, Track $track, string $sourceKey)
    {
        $source = $this->registry->resolveSource($track, $sourceKey);
        if ($source === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Tuning source not found',
            ], 404);
        }

        $path = $source['path'];
        $mimeType = mime_content_type($path) ?: 'audio/flac';
        return $this->streamFile($request, $path, $mimeType);
    }

    public function render(Request $request, Track $track): JsonResponse
    {
        $validated = $request->validate([
            'preset' => ['required', 'array'],
        ]);

        try {
            $task = $this->renderService->enqueue($track, $validated['preset']);
        } catch (RuntimeException $e) {
            return response()->json([
                'error' => 'Render unavailable',
                'message' => $e->getMessage(),
            ], 422);
        }

        return response()->json([
            'task_id' => $task->id,
            'status' => $task->status,
            'task_type' => $task->task_type,
            'message' => 'Stem tuning render queued',
        ], 202);
    }

    public function buildStems(Track $track): JsonResponse
    {
        try {
            $task = $this->stemBuildService->enqueue($track);
        } catch (RuntimeException $e) {
            return response()->json([
                'error' => 'Build unavailable',
                'message' => $e->getMessage(),
            ], 422);
        }

        return response()->json([
            'task_id' => $task->id,
            'status' => $task->status,
            'task_type' => $task->task_type,
            'message' => 'Stem source build queued',
        ], 202);
    }

    public function renderStatus(Track $track, int $taskId): JsonResponse
    {
        $task = QueueTask::query()
            ->whereKey($taskId)
            ->where('task_type', QueueTask::TYPE_STEM_TUNING_RENDER)
            ->first();

        if ($task === null || (int) ($task->payload['track_id'] ?? 0) !== (int) $track->id) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Stem tuning render task not found',
            ], 404);
        }

        $redisPayload = $this->readStemTuningRenderStatus($taskId);

        return response()->json([
            'task_id' => $task->id,
            'status' => (string) ($redisPayload['status'] ?? $task->status),
            'progress' => (int) ($redisPayload['progress'] ?? (
                in_array($task->status, [QueueTask::STATUS_COMPLETED, QueueTask::STATUS_CANCELLED], true) ? 100 : 0
            )),
            'message' => (string) ($redisPayload['message'] ?? match ($task->status) {
                QueueTask::STATUS_COMPLETED => 'Container render completed',
                QueueTask::STATUS_CANCELLED => 'Container render cancelled',
                default => 'Waiting for worker status',
            }),
            'error_message' => $task->error_message,
            'completed_at' => optional($task->completed_at)?->toIso8601String(),
            'logs' => $this->normalizeStemTuningLogs($redisPayload['logs'] ?? []),
            'payload' => [
                'track_id' => (int) ($task->payload['track_id'] ?? 0),
                'render_id' => (string) ($task->payload['render_id'] ?? ''),
            ],
        ]);
    }

    public function cancelRender(Request $request, Track $track, int $taskId): JsonResponse
    {
        $validated = $request->validate([
            'reason' => ['nullable', 'string', 'max:500'],
        ]);

        try {
            $result = $this->renderService->cancel(
                $track,
                $taskId,
                (string) ($validated['reason'] ?? 'Superseded by newer tuning preset changes'),
            );
        } catch (RuntimeException $e) {
            return response()->json([
                'error' => 'Cancel unavailable',
                'message' => $e->getMessage(),
            ], 422);
        }

        $task = $result['task'];
        $redisPayload = $this->readStemTuningRenderStatus($task->id);
        $queueCleanup = $result['queue_cleanup'];

        return response()->json([
            'task_id' => $task->id,
            'status' => (string) ($redisPayload['status'] ?? $task->status),
            'progress' => (int) ($redisPayload['progress'] ?? ($task->status === QueueTask::STATUS_CANCELLED ? 100 : 0)),
            'message' => (string) ($redisPayload['message'] ?? ($task->status === QueueTask::STATUS_CANCELLED ? 'Container render cancelled' : 'Cancellation requested')),
            'error_message' => $task->error_message,
            'completed_at' => optional($task->completed_at)?->toIso8601String(),
            'logs' => $this->normalizeStemTuningLogs($redisPayload['logs'] ?? []),
            'queue_cleanup' => $queueCleanup,
            'payload' => [
                'track_id' => (int) ($task->payload['track_id'] ?? 0),
                'render_id' => (string) ($task->payload['render_id'] ?? ''),
            ],
        ]);
    }

    public function buildStatus(Track $track, int $taskId): JsonResponse
    {
        $task = QueueTask::query()
            ->whereKey($taskId)
            ->where('task_type', QueueTask::TYPE_STEM_SOURCE_BUILD)
            ->first();

        if ($task === null || (int) ($task->payload['track_id'] ?? 0) !== (int) $track->id) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Stem source build task not found',
            ], 404);
        }

        return response()->json([
            'task_id' => $task->id,
            'status' => $task->status,
            'error_message' => $task->error_message,
            'completed_at' => optional($task->completed_at)?->toIso8601String(),
            'payload' => [
                'track_id' => (int) ($task->payload['track_id'] ?? 0),
                'build_id' => (string) ($task->payload['build_id'] ?? ''),
            ],
        ]);
    }

    private function streamFile(Request $request, string $path, string $mimeType)
    {
        $internalPath = $this->internalFilePath($path);
        if ($internalPath !== null) {
            return response('', 200, [
                'Content-Type' => $mimeType,
                'X-Accel-Redirect' => $internalPath,
                'X-Accel-Buffering' => 'no',
            ]);
        }

        $size = filesize($path);
        $headers = [
            'Content-Type' => $mimeType,
            'Content-Length' => $size,
            'Accept-Ranges' => 'bytes',
            'Cache-Control' => 'public, max-age=86400',
        ];

        if ($request->hasHeader('Range')) {
            $range = $request->header('Range');
            preg_match('/bytes=(\d+)-(\d*)/', (string) $range, $matches);

            $start = (int) ($matches[1] ?? 0);
            $end = isset($matches[2]) && $matches[2] !== '' ? (int) $matches[2] : $size - 1;
            $length = $end - $start + 1;

            $headers['Content-Range'] = "bytes {$start}-{$end}/{$size}";
            $headers['Content-Length'] = $length;

            return response()->stream(function () use ($path, $start, $length) {
                $handle = fopen($path, 'rb');
                if ($handle === false) {
                    return;
                }
                fseek($handle, $start);
                echo fread($handle, $length);
                fclose($handle);
            }, 206, $headers);
        }

        return response()->stream(function () use ($path) {
            readfile($path);
        }, 200, $headers);
    }

    private function internalFilePath(string $path): ?string
    {
        $normalizedPath = str_replace('\\', '/', realpath($path) ?: $path);
        $storageRoot = str_replace('\\', '/', realpath(storage_path('app')) ?: storage_path('app'));
        $artifactsRoot = str_replace('\\', '/', realpath(base_path('lab_artifacts')) ?: base_path('lab_artifacts'));

        foreach ([
            $storageRoot => '/internal-files/storage',
            $artifactsRoot => '/internal-files/lab_artifacts',
        ] as $prefix => $internalPrefix) {
            if ($prefix === '') {
                continue;
            }

            if ($normalizedPath === $prefix || str_starts_with($normalizedPath, $prefix . '/')) {
                $relativePath = ltrim(substr($normalizedPath, strlen($prefix)), '/');
                return $internalPrefix . '/' . $relativePath;
            }
        }

        return null;
    }

    /**
     * @return array<string, mixed>
     */
    private function readStemTuningRenderStatus(int $taskId): array
    {
        $raw = Redis::connection()->get('stem_tuning_render_status:' . $taskId);
        if (!is_string($raw) || trim($raw) === '') {
            return [];
        }

        $decoded = json_decode($raw, true);
        return is_array($decoded) ? $decoded : [];
    }

    /**
     * @param mixed $candidate
     * @return array<int, array{timestamp:string,level:string,progress:int,message:string,stage:?string}>
     */
    private function normalizeStemTuningLogs(mixed $candidate): array
    {
        if (!is_array($candidate)) {
            return [];
        }

        $logs = [];
        foreach ($candidate as $entry) {
            if (!is_array($entry)) {
                continue;
            }

            $logs[] = [
                'timestamp' => (string) ($entry['timestamp'] ?? ''),
                'level' => (string) ($entry['level'] ?? 'info'),
                'progress' => (int) ($entry['progress'] ?? 0),
                'message' => (string) ($entry['message'] ?? ''),
                'stage' => isset($entry['stage']) ? (string) $entry['stage'] : null,
            ];
        }

        return array_values($logs);
    }
}
