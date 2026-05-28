<?php

namespace App\Services;

use Illuminate\Support\Facades\Redis;

/**
 * MLQueueService - Centralized ML Worker queue management
 * 
 * SINGLE SOURCE OF TRUTH for ML worker queue name.
 * All code that pushes to ML worker queue MUST use this service.
 * 
 * IMPORTANT: Laravel's Redis facade automatically adds the prefix from config.
 * We use the BASE queue name here, and Laravel adds the prefix.
 * ML worker must listen on: {REDIS_PREFIX}{QUEUE_NAME}
 */
class MLQueueService
{
    private const STEM_TUNING_RENDER_STATUS_TTL_SECONDS = 3600;

    /**
     * Task types that should not wait behind the long-running segment backlog.
     *
     * @var array<int, string>
     */
    private const PRIORITY_TASK_TYPES = [
        'track_analysis',
        'fragment_analysis',
        'model_training',
        'embedding_recommendations',
        'similarity_search',
        'stem_source_build',
        'querysep_long_tail_build',
        'stem_tuning_render',
    ];

    /**
     * The base queue name (WITHOUT Redis prefix).
     * Laravel's Redis facade will add the prefix automatically.
     * ML worker must listen on: ai_music_recommendations_database_ml_worker:tasks
     */
    public const QUEUE_NAME = 'ml_worker:tasks';
    
    /**
     * Get the full queue name with Redis prefix (for reference/logging only).
     * DO NOT use this for Redis operations - Laravel adds prefix automatically.
     */
    public static function getFullQueueName(): string
    {
        return config('database.redis.options.prefix', '') . self::QUEUE_NAME;
    }

    /**
     * Flush (delete) the entire ML worker queue in Redis.
     * Used before reconciliation to prevent duplicates.
     *
     * @return int Number of items that were in the queue before flush.
     */
    public static function flushQueue(): int
    {
        $conn = Redis::connection();
        $count = (int) $conn->llen(self::QUEUE_NAME);
        $conn->del(self::QUEUE_NAME);
        return $count;
    }

    /**
     * Get current queue length.
     */
    public static function queueLength(): int
    {
        return (int) Redis::connection()->llen(self::QUEUE_NAME);
    }

    /**
     * Atomic reconciliation: flush queue and re-push all given payloads
     * inside a single MULTI/EXEC transaction.
     *
     * @param  array[]  $payloads  Array of payload arrays to push.
     * @return array{flushed: int, pushed: int}
     */
    public static function reconcileQueue(array $payloads): array
    {
        $conn = Redis::connection();
        $flushed = (int) $conn->llen(self::QUEUE_NAME);

        $encoded = array_map('json_encode', $payloads);

        $conn->multi();
        $conn->del(self::QUEUE_NAME);
        foreach ($encoded as $json) {
            $conn->rpush(self::QUEUE_NAME, $json);
        }
        $conn->exec();

        return ['flushed' => $flushed, 'pushed' => count($payloads)];
    }

    /**
     * Push a task to the ML worker queue.
     * Uses base queue name - Laravel adds prefix automatically.
     */
    public static function push(array $payload): void
    {
        // Use base queue name - Laravel's Redis facade adds prefix automatically
        Redis::connection()->rpush(self::QUEUE_NAME, json_encode($payload));
    }

    /**
     * Push a task to the front of the ML worker queue.
     */
    public static function pushFirst(array $payload): void
    {
        Redis::connection()->lpush(self::QUEUE_NAME, json_encode($payload));
    }

    /**
     * Push a task and then promote all priority task types ahead of the
     * segment backlog while preserving FIFO order inside each partition.
     *
     * @return array{queue_length:int,promoted:int}
     */
    public static function pushPriority(array $payload): array
    {
        self::push($payload);

        return self::prioritizeQueuedTaskTypes(self::PRIORITY_TASK_TYPES);
    }
    
    /**
     * Push a track analysis task.
     */
    public static function pushTrackAnalysis(int $taskId, int $trackId, string $audioPath): void
    {
        self::pushFirst([
            'type' => 'track_analysis',
            'task_id' => $taskId,
            'track_id' => $trackId,
            'audio_path' => $audioPath,
            'created_at' => now()->toIso8601String(),
        ]);
    }
    
    /**
     * Push a fragment analysis task.
     */
    public static function pushFragmentAnalysis(
        int $taskId,
        int $fragmentId,
        int $trackId,
        string $audioPath,
        float $startTime,
        float $endTime
    ): void {
        self::pushPriority([
            'type' => 'fragment_analysis',
            'task_id' => $taskId,
            'fragment_id' => $fragmentId,
            'track_id' => $trackId,
            'audio_path' => $audioPath,
            'start_time' => $startTime,
            'end_time' => $endTime,
            'created_at' => now()->toIso8601String(),
        ]);
    }
    
    /**
     * Push a model training task.
     */
    public static function pushModelTraining(int $taskId, ?int $userId = null): void
    {
        self::pushPriority([
            'type' => 'model_training',
            'task_id' => $taskId,
            'user_id' => $userId,
            'created_at' => now()->toIso8601String(),
        ]);
    }

    /**
     * Push a stem tuning render task.
     */
    public static function pushStemTuningRender(
        int $taskId,
        int $trackId,
        string $renderId,
        string $sourceStemManifestPath,
        string $outputDir,
        string $latestPointerPath,
        array $preset,
        ?string $rawReferencePath = null,
        ?string $sourceAudioPath = null,
    ): array {
        return self::pushPriority([
            'type' => 'stem_tuning_render',
            'task_id' => $taskId,
            'lab_track_id' => $trackId,
            'render_id' => $renderId,
            'source_stem_manifest_path' => $sourceStemManifestPath,
            'output_dir' => $outputDir,
            'latest_pointer_path' => $latestPointerPath,
            'preset' => $preset,
            'raw_reference_path' => $rawReferencePath,
            'source_audio_path' => $sourceAudioPath,
            'created_at' => now()->toIso8601String(),
        ]);
    }

    /**
     * Request cancellation of an in-flight stem tuning render task.
     */
    public static function requestStemTuningRenderCancellation(int $taskId, string $reason): void
    {
        Redis::connection()->setex(
            self::stemTuningRenderControlKey($taskId),
            self::STEM_TUNING_RENDER_STATUS_TTL_SECONDS,
            json_encode([
                'cancel_requested' => true,
                'reason' => trim($reason) !== '' ? $reason : 'Stem tuning render cancelled',
                'requested_at' => now()->toIso8601String(),
                'task_id' => $taskId,
            ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES),
        );
    }

    /**
     * Clear the control payload for a stem tuning render task.
     */
    public static function clearStemTuningRenderCancellation(int $taskId): void
    {
        Redis::connection()->del(self::stemTuningRenderControlKey($taskId));
    }

    /**
     * @return array<string, mixed>|null
     */
    public static function readStemTuningRenderCancellation(int $taskId): ?array
    {
        $raw = Redis::connection()->get(self::stemTuningRenderControlKey($taskId));
        if (!is_string($raw) || trim($raw) === '') {
            return null;
        }

        $decoded = json_decode($raw, true);
        return is_array($decoded) ? $decoded : null;
    }

    /**
     * Push a stem source build task.
     */
    public static function pushStemSourceBuild(
        int $taskId,
        int $trackId,
        string $buildId,
        string $audioPath,
        string $outputDir,
        string $playbackManifestPath,
        string $latestPointerPath,
        string $modelName,
    ): array {
        return self::pushPriority([
            'type' => 'stem_source_build',
            'task_id' => $taskId,
            'track_id' => $trackId,
            'build_id' => $buildId,
            'audio_path' => $audioPath,
            'output_dir' => $outputDir,
            'playback_manifest_path' => $playbackManifestPath,
            'latest_pointer_path' => $latestPointerPath,
            'model_name' => $modelName,
            'created_at' => now()->toIso8601String(),
        ]);
    }

    /**
     * Push a long-tail query separation task.
     *
     * The Python worker enforces banquet_byoq and fails fast if querysep output
     * is not visible through the supplied shared path mappings.
     *
     * @param array<int, string> $targetStems
     * @param array<int, array{local_prefix:string,querysep_prefix:string}> $pathMappings
     * @param array<string, string> $buildFromPaths
     */
    public static function pushQuerysepLongTailBuild(
        int $taskId,
        int $trackId,
        string $buildId,
        string $audioPath,
        string $sourceStemManifestPath,
        string $outputDir,
        array $targetStems,
        ?string $querysepWorkerUrl = null,
        ?string $querysepInputPath = null,
        ?string $querysepOutputRoot = null,
        array $pathMappings = [],
        array $buildFromPaths = [],
        string $backend = 'banquet_byoq',
    ): array {
        $payload = [
            'type' => 'querysep_long_tail_build',
            'task_id' => $taskId,
            'track_id' => $trackId,
            'build_id' => $buildId,
            'audio_path' => $audioPath,
            'source_stem_manifest_path' => $sourceStemManifestPath,
            'output_dir' => $outputDir,
            'target_stems' => array_values($targetStems),
            'querysep_worker_url' => $querysepWorkerUrl,
            'querysep_input_path' => $querysepInputPath,
            'querysep_output_root' => $querysepOutputRoot,
            'path_mappings' => array_values($pathMappings),
            'backend' => $backend,
            'dry_run' => false,
            'created_at' => now()->toIso8601String(),
        ];

        if ($buildFromPaths !== []) {
            $payload['build_from_paths'] = $buildFromPaths;
        }

        return self::pushPriority($payload);
    }

    /**
     * Remove queued Redis items that match a concrete task_id while preserving
     * the order of all other pending ML worker tasks.
     *
     * @return array{removed:int,remaining:int}
     */
    public static function removeQueuedTaskById(int $taskId): array
    {
        $conn = Redis::connection();
        $items = $conn->lrange(self::QUEUE_NAME, 0, -1) ?? [];

        $kept = [];
        $removed = 0;

        foreach ($items as $item) {
            $decoded = json_decode($item, true);
            $queuedTaskId = is_array($decoded) ? ($decoded['task_id'] ?? null) : null;

            if ((string) $queuedTaskId === (string) $taskId) {
                $removed++;
                continue;
            }

            $kept[] = $item;
        }

        $conn->multi();
        $conn->del(self::QUEUE_NAME);
        foreach ($kept as $item) {
            $conn->rpush(self::QUEUE_NAME, $item);
        }
        $conn->exec();

        return [
            'removed' => $removed,
            'remaining' => count($kept),
        ];
    }

    private static function stemTuningRenderControlKey(int $taskId): string
    {
        return 'stem_tuning_render_control:' . $taskId;
    }

    /**
     * Read queued model_training task ids directly from Redis in queue order.
     *
     * @return array<int, int>
     */
    public static function getQueuedTrainingTaskIds(): array
    {
        $items = Redis::connection()->lrange(self::QUEUE_NAME, 0, -1) ?? [];
        $taskIds = [];

        foreach ($items as $item) {
            $decoded = json_decode($item, true);
            $taskType = is_array($decoded) ? ($decoded['type'] ?? null) : null;
            $taskId = is_array($decoded) ? ($decoded['task_id'] ?? null) : null;

            if ($taskType !== 'model_training' || !is_numeric($taskId)) {
                continue;
            }

            $taskIds[] = (int) $taskId;
        }

        return $taskIds;
    }

    /**
     * Reconcile only queued model_training items against the current set of
     * pending training tasks from the database. Non-training queue items keep
     * their relative order and remain untouched.
     *
     * @param array<int, array{type:string,task_id:int,user_id:int|null,created_at:string}> $pendingPayloads
     * @return array{
     *   pending_db_task_ids: array<int, int>,
     *   queued_before_task_ids: array<int, int>,
     *   kept_pending_task_ids: array<int, int>,
     *   removed_stale_task_ids: array<int, int>,
     *   added_missing_task_ids: array<int, int>,
     *   remaining_training_task_ids: array<int, int>
     * }
     */
    public static function reconcilePendingTrainingTasks(array $pendingPayloads): array
    {
        $conn = Redis::connection();
        $items = $conn->lrange(self::QUEUE_NAME, 0, -1) ?? [];

        $pendingById = [];
        $pendingDbTaskIds = [];
        foreach ($pendingPayloads as $payload) {
            $taskId = (int) ($payload['task_id'] ?? 0);
            if ($taskId <= 0) {
                continue;
            }

            $pendingById[(string) $taskId] = $payload;
            $pendingDbTaskIds[] = $taskId;
        }

        $keptQueueItems = [];
        $queuedBeforeTaskIds = [];
        $keptPendingTaskIds = [];
        $removedStaleTaskIds = [];

        foreach ($items as $item) {
            $decoded = json_decode($item, true);

            if (!is_array($decoded) || (($decoded['type'] ?? null) !== 'model_training')) {
                $keptQueueItems[] = $item;
                continue;
            }

            $taskId = isset($decoded['task_id']) ? (int) $decoded['task_id'] : 0;
            if ($taskId <= 0) {
                $keptQueueItems[] = $item;
                continue;
            }

            $queuedBeforeTaskIds[] = $taskId;
            $taskKey = (string) $taskId;

            if (array_key_exists($taskKey, $pendingById)) {
                $keptQueueItems[] = $item;
                $keptPendingTaskIds[] = $taskId;
                continue;
            }

            $removedStaleTaskIds[] = $taskId;
        }

        $addedMissingTaskIds = [];
        foreach ($pendingById as $taskKey => $payload) {
            $taskId = (int) $taskKey;
            if (in_array($taskId, $keptPendingTaskIds, true)) {
                continue;
            }

            $keptQueueItems[] = json_encode($payload);
            $addedMissingTaskIds[] = $taskId;
        }

        $remainingTrainingTaskIds = array_values(array_map(
            static fn (array $payload): int => (int) $payload['task_id'],
            $pendingPayloads
        ));

        $conn->multi();
        $conn->del(self::QUEUE_NAME);
        foreach ($keptQueueItems as $item) {
            $conn->rpush(self::QUEUE_NAME, $item);
        }
        $conn->exec();

        return [
            'pending_db_task_ids' => array_values(array_unique($pendingDbTaskIds)),
            'queued_before_task_ids' => array_values(array_unique($queuedBeforeTaskIds)),
            'kept_pending_task_ids' => array_values(array_unique($keptPendingTaskIds)),
            'removed_stale_task_ids' => array_values(array_unique($removedStaleTaskIds)),
            'added_missing_task_ids' => array_values(array_unique($addedMissingTaskIds)),
            'remaining_training_task_ids' => array_values(array_unique($remainingTrainingTaskIds)),
        ];
    }

    /**
     * Rebuild only track_analysis items so every pending analysis task is present
     * exactly once and placed ahead of lower-priority worker tasks.
     *
     * @param array<int, array{type:string,task_id:int,track_id:int,audio_path:string,created_at:string}> $pendingPayloads
     * @return array{
     *   pending_db_task_ids: array<int, int>,
     *   queued_before_task_ids: array<int, int>,
     *   removed_stale_task_ids: array<int, int>,
     *   prioritized_task_ids: array<int, int>,
     *   remaining_track_analysis_task_ids: array<int, int>
     * }
     */
    public static function reconcilePendingTrackAnalysisTasks(array $pendingPayloads): array
    {
        $conn = Redis::connection();
        $items = $conn->lrange(self::QUEUE_NAME, 0, -1) ?? [];

        $pendingById = [];
        $pendingDbTaskIds = [];
        foreach ($pendingPayloads as $payload) {
            $taskId = (int) ($payload['task_id'] ?? 0);
            if ($taskId <= 0) {
                continue;
            }

            $pendingById[(string) $taskId] = $payload;
            $pendingDbTaskIds[] = $taskId;
        }

        $nonTrackQueueItems = [];
        $queuedBeforeTaskIds = [];
        $removedStaleTaskIds = [];

        foreach ($items as $item) {
            $decoded = json_decode($item, true);

            if (!is_array($decoded) || (($decoded['type'] ?? null) !== 'track_analysis')) {
                $nonTrackQueueItems[] = $item;
                continue;
            }

            $taskId = isset($decoded['task_id']) ? (int) $decoded['task_id'] : 0;
            if ($taskId <= 0) {
                continue;
            }

            $queuedBeforeTaskIds[] = $taskId;
            if (!array_key_exists((string) $taskId, $pendingById)) {
                $removedStaleTaskIds[] = $taskId;
            }
        }

        $prioritizedTaskIds = [];
        $prioritizedTrackItems = [];
        foreach ($pendingPayloads as $payload) {
            $taskId = (int) ($payload['task_id'] ?? 0);
            if ($taskId <= 0) {
                continue;
            }

            $prioritizedTaskIds[] = $taskId;
            $prioritizedTrackItems[] = json_encode($payload);
        }

        $rebuiltQueueItems = array_merge($prioritizedTrackItems, $nonTrackQueueItems);

        $conn->multi();
        $conn->del(self::QUEUE_NAME);
        foreach ($rebuiltQueueItems as $item) {
            $conn->rpush(self::QUEUE_NAME, $item);
        }
        $conn->exec();

        return [
            'pending_db_task_ids' => array_values(array_unique($pendingDbTaskIds)),
            'queued_before_task_ids' => array_values(array_unique($queuedBeforeTaskIds)),
            'removed_stale_task_ids' => array_values(array_unique($removedStaleTaskIds)),
            'prioritized_task_ids' => array_values(array_unique($prioritizedTaskIds)),
            'remaining_track_analysis_task_ids' => array_values(array_unique($prioritizedTaskIds)),
        ];
    }

    /**
     * Promote selected queued task types ahead of lower-priority backlog items
     * while preserving the current FIFO order inside both partitions.
     *
     * @param  array<int, string>  $taskTypes
     * @return array{queue_length:int,promoted:int}
     */
    public static function prioritizeQueuedTaskTypes(array $taskTypes): array
    {
        $conn = Redis::connection();
        $items = $conn->lrange(self::QUEUE_NAME, 0, -1) ?? [];
        $queueLength = count($items);

        if ($queueLength === 0) {
            return [
                'queue_length' => 0,
                'promoted' => 0,
            ];
        }

        $priorityLookup = [];
        foreach ($taskTypes as $taskType) {
            if (!is_string($taskType) || $taskType === '') {
                continue;
            }

            $priorityLookup[$taskType] = true;
        }

        if ($priorityLookup === []) {
            return [
                'queue_length' => $queueLength,
                'promoted' => 0,
            ];
        }

        $priorityItems = [];
        $otherItems = [];

        foreach ($items as $item) {
            $decoded = json_decode($item, true);
            $taskType = is_array($decoded) ? ($decoded['type'] ?? null) : null;

            if (is_string($taskType) && isset($priorityLookup[$taskType])) {
                $priorityItems[] = $item;
                continue;
            }

            $otherItems[] = $item;
        }

        $promoted = count($priorityItems);
        if ($promoted === 0 || $promoted === $queueLength) {
            return [
                'queue_length' => $queueLength,
                'promoted' => $promoted,
            ];
        }

        $conn->multi();
        $conn->del(self::QUEUE_NAME);
        foreach (array_merge($priorityItems, $otherItems) as $item) {
            $conn->rpush(self::QUEUE_NAME, $item);
        }
        $conn->exec();

        return [
            'queue_length' => $queueLength,
            'promoted' => $promoted,
        ];
    }
    
    /**
     * Push a DJ analysis task.
     */
    public static function pushDjAnalysis(int $taskId, int $trackId, string $audioPath): void
    {
        self::push([
            'type' => 'dj_analysis',
            'task_id' => $taskId,
            'track_id' => $trackId,
            'audio_path' => $audioPath,
            'created_at' => now()->toIso8601String(),
        ]);
    }
    
    /**
     * Push a segment analysis task.
     * Returns task_id for status tracking.
     */
    public static function pushSegmentAnalysis(int $trackId): string
    {
        return self::queueSegmentAnalysis($trackId, false);
    }

    /**
     * Push a user-requested segment analysis task ahead of the background backlog
     * while preserving FIFO order between manual requests.
     */
    public static function pushPrioritySegmentAnalysis(int $trackId): string
    {
        return self::queueSegmentAnalysis($trackId, true);
    }

    /**
     * Read the first queued segment-analysis task for a track without mutating the queue.
     *
     * @return array{task_id:string,priority:string|null,duplicates_removed:int,promoted:bool}|null
     */
    public static function getQueuedSegmentAnalysisTask(int $trackId): ?array
    {
        return self::inspectQueuedSegmentAnalysisTask($trackId, false, false);
    }

    /**
     * Collapse duplicate queued segment-analysis tasks for a track and promote the
     * surviving queued task to manual priority when requested by the user.
     *
     * @return array{task_id:string,priority:string|null,duplicates_removed:int,promoted:bool}|null
     */
    public static function promoteAndDeduplicateQueuedSegmentAnalysisTask(int $trackId): ?array
    {
        return self::inspectQueuedSegmentAnalysisTask($trackId, true, true);
    }

    /**
     * Promote user-requested segment-analysis tasks ahead of the rest of the ML queue
     * while preserving the current FIFO order inside both partitions.
     *
     * @return array{queue_length:int,promoted:int}
     */
    public static function prioritizeQueuedManualSegmentAnalysisTasks(): array
    {
        $conn = Redis::connection();
        $items = $conn->lrange(self::QUEUE_NAME, 0, -1) ?? [];
        $queueLength = count($items);

        if ($queueLength === 0) {
            return [
                'queue_length' => 0,
                'promoted' => 0,
            ];
        }

        $manualSegmentItems = [];
        $otherItems = [];

        foreach ($items as $item) {
            $decoded = json_decode($item, true);
            $taskType = is_array($decoded) ? ($decoded['type'] ?? null) : null;
            $taskPriority = is_array($decoded) ? ($decoded['priority'] ?? null) : null;

            if ($taskType === 'analyze_segments' && $taskPriority === 'manual') {
                $manualSegmentItems[] = $item;
                continue;
            }

            $otherItems[] = $item;
        }

        $promoted = count($manualSegmentItems);
        if ($promoted === 0 || $promoted === $queueLength) {
            return [
                'queue_length' => $queueLength,
                'promoted' => $promoted,
            ];
        }

        $conn->multi();
        $conn->del(self::QUEUE_NAME);
        foreach (array_merge($manualSegmentItems, $otherItems) as $item) {
            $conn->rpush(self::QUEUE_NAME, $item);
        }
        $conn->exec();

        return [
            'queue_length' => $queueLength,
            'promoted' => $promoted,
        ];
    }

    /**
     * @return array{task_id:string,priority:string|null,duplicates_removed:int,promoted:bool}|null
     */
    private static function inspectQueuedSegmentAnalysisTask(int $trackId, bool $promoteToManual, bool $applyMutations): ?array
    {
        $conn = Redis::connection();
        $items = $conn->lrange(self::QUEUE_NAME, 0, -1) ?? [];

        if ($items === []) {
            return null;
        }

        $keptItems = [];
        $match = null;
        $mutated = false;

        foreach ($items as $item) {
            $decoded = json_decode($item, true);
            $taskType = is_array($decoded) ? ($decoded['type'] ?? null) : null;
            $queuedTrackId = is_array($decoded) ? (int) ($decoded['track_id'] ?? 0) : 0;
            $taskId = is_array($decoded) ? ($decoded['task_id'] ?? null) : null;

            if ($taskType !== 'analyze_segments' || $queuedTrackId !== $trackId || !is_string($taskId) || $taskId === '') {
                $keptItems[] = $item;
                continue;
            }

            if ($match !== null) {
                $match['duplicates_removed']++;
                $mutated = true;
                continue;
            }

            $priority = is_array($decoded) ? ($decoded['priority'] ?? null) : null;
            $promoted = false;

            if ($promoteToManual && $priority !== 'manual') {
                $decoded['priority'] = 'manual';
                $priority = 'manual';
                $item = json_encode($decoded);
                $promoted = true;
                $mutated = true;
            }

            $match = [
                'task_id' => $taskId,
                'priority' => is_string($priority) ? $priority : null,
                'duplicates_removed' => 0,
                'promoted' => $promoted,
            ];

            $keptItems[] = $item;
        }

        if ($match === null) {
            return null;
        }

        if ($applyMutations && $mutated) {
            $conn->multi();
            $conn->del(self::QUEUE_NAME);
            foreach ($keptItems as $queuedItem) {
                $conn->rpush(self::QUEUE_NAME, $queuedItem);
            }
            $conn->exec();

            if ($match['priority'] === 'manual') {
                self::prioritizeQueuedManualSegmentAnalysisTasks();
            }
        }

        return $match;
    }

    /**
     * Push a segment analysis task and optionally mark it as manual priority.
     * Returns task_id for status tracking.
     */
    private static function queueSegmentAnalysis(int $trackId, bool $prioritize): string
    {
        $taskId = 'segment_analysis_' . $trackId . '_' . time();
        $createdAt = now()->toIso8601String();

        // Set initial status in Redis
        Redis::connection()->setex(
            "segment_analysis_status:{$trackId}",
            300, // 5 minutes TTL
            json_encode([
                'status' => 'queued',
                'task_id' => $taskId,
                'track_id' => $trackId,
                'progress' => 0,
                'message' => 'Waiting in queue...',
                'created_at' => $createdAt,
            ])
        );

        $payload = [
            'type' => 'analyze_segments',
            'task_id' => $taskId,
            'track_id' => $trackId,
            'created_at' => $createdAt,
        ];

        if ($prioritize) {
            $payload['priority'] = 'manual';
        }

        self::push($payload);
        if ($prioritize) {
            self::prioritizeQueuedManualSegmentAnalysisTasks();
        }

        return $taskId;
    }
    
    /**
     * Get segment analysis status for a track.
     */
    public static function getSegmentAnalysisStatus(int $trackId): ?array
    {
        $status = Redis::connection()->get("segment_analysis_status:{$trackId}");
        return $status ? json_decode($status, true) : null;
    }
    
    /**
     * Update segment analysis status.
     */
    public static function updateSegmentAnalysisStatus(int $trackId, array $data): void
    {
        $current = self::getSegmentAnalysisStatus($trackId) ?? [];
        $updated = array_merge($current, $data, ['updated_at' => now()->toIso8601String()]);
        
        Redis::connection()->setex(
            "segment_analysis_status:{$trackId}",
            300, // 5 minutes TTL
            json_encode($updated)
        );
    }
}
