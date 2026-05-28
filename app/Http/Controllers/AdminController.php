<?php

namespace App\Http\Controllers;

use App\Models\Fragment;
use App\Models\ListeningHistory;
use App\Models\Playlist;
use App\Models\PreferenceModelStatus;
use App\Models\QueueTask;
use App\Models\Track;
use App\Models\TrackRating;
use App\Models\User;
use App\Models\UserPatternExampleRating;
use App\Models\UserTrackLike;
use App\Models\UserTrackScore;
use App\Support\AppStoragePath;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\DB;
use App\Services\MLQueueService;
use App\Services\PreferenceTrainingService;

class AdminController extends Controller
{
    private const RADIO_DIAGNOSTICS_RECENT_EVENT_LIMIT = 40;
    private const RADIO_DIAGNOSTICS_LOG_FILE_LIMIT = 3;
    private const RADIO_DIAGNOSTICS_LOG_TAIL_LINES = 250;

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

    public function overview(Request $request): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $latestModelStatus = PreferenceModelStatus::query()->latest('updated_at')->first();
        $trainingQueueCounts = $this->getTrainingQueueCounts();
        $trainingQueueDrift = $this->getTrainingQueueDrift();
        $trainingScopeSummary = $this->getTrainingScopeSummary();
        $userModels = $this->getUserModelOverview();
        $recentTrainingRuns = $this->getRecentTrainingRuns();
        $recentTrainingTasks = $this->getRecentTrainingTasks();
        $radioDiagnostics = $this->getRadioDiagnosticsOverview();

        return response()->json([
            'users' => [
                'total' => User::count(),
                'admins' => User::where('role', User::ROLE_ADMIN)->count(),
                'members' => User::where('role', User::ROLE_USER)->count(),
            ],
            'tracks' => [
                'total' => Track::count(),
                'completed' => Track::where('analysis_status', 'completed')->count(),
                'pending' => Track::where('analysis_status', 'pending')->count(),
                'processing' => Track::where('analysis_status', 'processing')->count(),
                'failed' => Track::where('analysis_status', 'failed')->count(),
            ],
            'ratings' => [
                'total' => TrackRating::count(),
                'high' => TrackRating::where('rating', '>=', 4)->count(),
                'low' => TrackRating::where('rating', '<=', 2)->count(),
            ],
            'playlists' => [
                'total' => Playlist::count(),
                'ai_generated' => Playlist::where('is_ai_generated', true)->count(),
            ],
            'queue' => [
                'pending' => QueueTask::where('status', QueueTask::STATUS_PENDING)->count(),
                'processing' => QueueTask::where('status', QueueTask::STATUS_PROCESSING)->count(),
                'completed' => QueueTask::where('status', QueueTask::STATUS_COMPLETED)->count(),
                'failed' => QueueTask::where('status', QueueTask::STATUS_FAILED)->count(),
            ],
            'model' => [
                'status' => $latestModelStatus?->status ?? 'not_trained',
                'trained_at' => $latestModelStatus?->trained_at?->toIso8601String(),
                'model_path' => $latestModelStatus?->model_path,
                'likes_count' => $latestModelStatus?->likes_count ?? 0,
                'trained_users' => PreferenceModelStatus::query()
                    ->whereNotNull('user_id')
                    ->where('status', PreferenceModelStatus::STATUS_TRAINED)
                    ->count(),
                'training_users' => PreferenceModelStatus::query()
                    ->whereNotNull('user_id')
                    ->where('status', PreferenceModelStatus::STATUS_TRAINING)
                    ->count(),
                'stale_training_users' => PreferenceModelStatus::query()
                    ->whereNotNull('user_id')
                    ->where('status', PreferenceModelStatus::STATUS_TRAINING)
                    ->where('updated_at', '<', now()->subMinutes(10))
                    ->count(),
            ],
            'training_queue' => $trainingQueueCounts,
            'training_queue_drift' => $trainingQueueDrift,
            'training_scope_summary' => $trainingScopeSummary,
            'user_models' => $userModels,
            'recent_training_runs' => $recentTrainingRuns,
            'recent_training_tasks' => $recentTrainingTasks,
            'radio_diagnostics' => $radioDiagnostics,
        ]);
    }

    public function retrainUserModel(Request $request, User $user): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $stats = PreferenceTrainingService::getStats($user->id);
        if (($stats['total_positive'] ?? 0) < 1) {
            return response()->json([
                'error' => 'Insufficient data',
                'message' => "User {$user->id} does not have enough positive feedback to retrain.",
                'stats' => $stats,
            ], 400);
        }

        $status = PreferenceModelStatus::forUser($user->id)->first();
        if ($status && $status->status === PreferenceModelStatus::STATUS_TRAINING) {
            return response()->json([
                'error' => 'Already training',
                'message' => "User {$user->id} already has an active training task. Retry or cancel the pending task first.",
            ], 409);
        }

        if ($this->hasActiveTrainingTaskForScope($user->id)) {
            return response()->json([
                'error' => 'Active training task exists',
                'message' => "User {$user->id} already has a pending or processing training task in the queue.",
            ], 409);
        }

        $latestBeforeId = $this->getLatestTrainingTaskIdForUser($user->id);
        PreferenceTrainingService::triggerTraining($user->id);
        $latestTask = $this->getLatestTrainingTaskForUser($user->id);

        if (!$latestTask || $latestTask->id === $latestBeforeId) {
            return response()->json([
                'error' => 'Training not queued',
                'message' => "Failed to queue a new training task for user {$user->id}.",
            ], 409);
        }

        return response()->json([
            'success' => true,
            'message' => "Queued retraining for user {$user->id}",
            'task_id' => $latestTask->id,
            'user_id' => $user->id,
            'stats' => $stats,
        ], 202);
    }

    public function retryTrainingTask(Request $request, int $taskId): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $task = $this->getTrainingTaskOrFail($taskId);
        $userId = $this->extractTrainingTaskUserId($task);

        if ($task->status !== QueueTask::STATUS_FAILED) {
            return response()->json([
                'error' => 'Unsupported status',
                'message' => "Only failed training tasks can be retried safely. Task {$task->id} is currently {$task->status}.",
            ], 400);
        }

        if ($this->hasActiveTrainingTaskForScope($userId, $task->id)) {
            return response()->json([
                'error' => 'Active training task exists',
                'message' => "Task {$task->id} cannot be retried while another training task for the same user scope is pending or processing.",
            ], 409);
        }

        $queueCleanup = MLQueueService::removeQueuedTaskById($task->id);

        $task->update([
            'status' => QueueTask::STATUS_PENDING,
            'error_message' => null,
            'completed_at' => null,
        ]);

        $this->markModelStatusQueued($userId);
        MLQueueService::pushModelTraining($task->id, $userId);

        return response()->json([
            'success' => true,
            'message' => "Queued retry for training task {$task->id}",
            'task_id' => $task->id,
            'user_id' => $userId,
            'queue_cleanup' => $queueCleanup,
        ]);
    }

    public function cancelTrainingTask(Request $request, int $taskId): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $task = $this->getTrainingTaskOrFail($taskId);

        if ($task->status !== QueueTask::STATUS_PENDING) {
            return response()->json([
                'error' => 'Unsafe cancel',
                'message' => "Only pending training tasks can be cancelled safely. Task {$task->id} is currently {$task->status}.",
            ], 400);
        }

        $userId = $this->extractTrainingTaskUserId($task);
        $queueCleanup = MLQueueService::removeQueuedTaskById($task->id);

        $task->update([
            'status' => QueueTask::STATUS_FAILED,
            'error_message' => 'Cancelled by admin',
            'completed_at' => now(),
        ]);

        $this->restoreModelStatusIfNoActiveTraining($userId, $task->id);

        return response()->json([
            'success' => true,
            'message' => "Cancelled pending training task {$task->id}",
            'task_id' => $task->id,
            'user_id' => $userId,
            'queue_cleanup' => $queueCleanup,
        ]);
    }

    public function repairTrainingQueue(Request $request): JsonResponse
    {
        if ($response = $this->requireAdmin($request)) {
            return $response;
        }

        $pendingTasks = QueueTask::query()
            ->where('task_type', QueueTask::TYPE_TRAINING)
            ->where('status', QueueTask::STATUS_PENDING)
            ->orderBy('created_at')
            ->orderBy('id')
            ->get();

        $pendingPayloads = [];
        foreach ($pendingTasks as $task) {
            $userId = $this->extractTrainingTaskUserId($task);
            $pendingPayloads[] = [
                'type' => 'model_training',
                'task_id' => $task->id,
                'user_id' => $userId,
                'created_at' => $task->created_at?->toIso8601String() ?? now()->toIso8601String(),
            ];

            $this->markModelStatusQueued($userId);
        }

        $result = MLQueueService::reconcilePendingTrainingTasks($pendingPayloads);

        return response()->json([
            'success' => true,
            'message' => sprintf(
                'Training queue repaired: kept=%d, added=%d, removed=%d',
                count($result['kept_pending_task_ids']),
                count($result['added_missing_task_ids']),
                count($result['removed_stale_task_ids']),
            ),
            'repair' => $result,
        ]);
    }

    /**
     * @return array{pending:int,processing:int,completed:int,failed:int}
     */
    private function getTrainingQueueCounts(): array
    {
        return [
            'pending' => QueueTask::query()->ofType(QueueTask::TYPE_TRAINING)->withStatus(QueueTask::STATUS_PENDING)->count(),
            'processing' => QueueTask::query()->ofType(QueueTask::TYPE_TRAINING)->withStatus(QueueTask::STATUS_PROCESSING)->count(),
            'completed' => QueueTask::query()->ofType(QueueTask::TYPE_TRAINING)->withStatus(QueueTask::STATUS_COMPLETED)->count(),
            'failed' => QueueTask::query()->ofType(QueueTask::TYPE_TRAINING)->withStatus(QueueTask::STATUS_FAILED)->count(),
        ];
    }

    /**
     * @return array{
     *   healthy: bool,
     *   pending_db_count: int,
     *   queued_redis_count: int,
     *   missing_from_redis_count: int,
     *   stale_in_redis_count: int,
     *   pending_db_task_ids: array<int, int>,
     *   queued_redis_task_ids: array<int, int>,
     *   missing_from_redis_task_ids: array<int, int>,
     *   stale_in_redis_task_ids: array<int, int>
     * }
     */
    private function getTrainingQueueDrift(): array
    {
        $pendingDbTaskIds = QueueTask::query()
            ->where('task_type', QueueTask::TYPE_TRAINING)
            ->where('status', QueueTask::STATUS_PENDING)
            ->orderBy('created_at')
            ->orderBy('id')
            ->pluck('id')
            ->map(fn ($id) => (int) $id)
            ->values()
            ->all();

        $queuedRedisTaskIds = MLQueueService::getQueuedTrainingTaskIds();

        $pendingSet = array_fill_keys(array_map('strval', $pendingDbTaskIds), true);
        $queuedSet = array_fill_keys(array_map('strval', $queuedRedisTaskIds), true);

        $missingFromRedisTaskIds = array_values(array_filter(
            $pendingDbTaskIds,
            static fn (int $taskId): bool => !isset($queuedSet[(string) $taskId])
        ));
        $staleInRedisTaskIds = array_values(array_filter(
            array_values(array_unique($queuedRedisTaskIds)),
            static fn (int $taskId): bool => !isset($pendingSet[(string) $taskId])
        ));

        return [
            'healthy' => count($missingFromRedisTaskIds) === 0
                && count($staleInRedisTaskIds) === 0
                && count($pendingDbTaskIds) === count($queuedRedisTaskIds),
            'pending_db_count' => count($pendingDbTaskIds),
            'queued_redis_count' => count($queuedRedisTaskIds),
            'missing_from_redis_count' => count($missingFromRedisTaskIds),
            'stale_in_redis_count' => count($staleInRedisTaskIds),
            'pending_db_task_ids' => $pendingDbTaskIds,
            'queued_redis_task_ids' => $queuedRedisTaskIds,
            'missing_from_redis_task_ids' => $missingFromRedisTaskIds,
            'stale_in_redis_task_ids' => $staleInRedisTaskIds,
        ];
    }

    /**
     * @return array{
     *   user_scoped: array{
     *     tasks_pending:int,
     *     tasks_processing:int,
     *     tasks_completed:int,
     *     tasks_failed:int,
     *     runs_total:int,
     *     latest_run_id:int|null,
     *     latest_run_at:string|null
     *   },
     *   global_legacy: array{
     *     tasks_pending:int,
     *     tasks_processing:int,
     *     tasks_completed:int,
     *     tasks_failed:int,
     *     runs_total:int,
     *     latest_run_id:int|null,
     *     latest_run_at:string|null
     *   }
     * }
     */
    private function getTrainingScopeSummary(): array
    {
        $summary = [
            'user_scoped' => [
                'tasks_pending' => 0,
                'tasks_processing' => 0,
                'tasks_completed' => 0,
                'tasks_failed' => 0,
                'runs_total' => 0,
                'latest_run_id' => null,
                'latest_run_at' => null,
            ],
            'global_legacy' => [
                'tasks_pending' => 0,
                'tasks_processing' => 0,
                'tasks_completed' => 0,
                'tasks_failed' => 0,
                'runs_total' => 0,
                'latest_run_id' => null,
                'latest_run_at' => null,
            ],
        ];

        $taskRows = DB::table('queue_tasks')
            ->selectRaw("
                CASE
                    WHEN jsonb_exists(payload::jsonb, 'user_id') AND NULLIF(payload->>'user_id', '') IS NOT NULL THEN 'user_scoped'
                    ELSE 'global_legacy'
                END as scope,
                status,
                COUNT(*) as aggregate
            ")
            ->where('task_type', QueueTask::TYPE_TRAINING)
            ->groupByRaw("
                CASE
                    WHEN jsonb_exists(payload::jsonb, 'user_id') AND NULLIF(payload->>'user_id', '') IS NOT NULL THEN 'user_scoped'
                    ELSE 'global_legacy'
                END,
                status
            ")
            ->get();

        foreach ($taskRows as $row) {
            $scope = ($row->scope === 'user_scoped') ? 'user_scoped' : 'global_legacy';
            $statusKey = match ($row->status) {
                QueueTask::STATUS_PENDING => 'tasks_pending',
                QueueTask::STATUS_PROCESSING => 'tasks_processing',
                QueueTask::STATUS_COMPLETED => 'tasks_completed',
                QueueTask::STATUS_FAILED => 'tasks_failed',
                default => null,
            };

            if ($statusKey === null) {
                continue;
            }

            $summary[$scope][$statusKey] = (int) $row->aggregate;
        }

        $runRows = DB::table('preference_model_training_runs')
            ->selectRaw("
                CASE
                    WHEN user_id IS NULL THEN 'global_legacy'
                    ELSE 'user_scoped'
                END as scope,
                COUNT(*) as runs_total,
                MAX(id) as latest_run_id,
                MAX(COALESCE(finished_at, created_at)) as latest_run_at
            ")
            ->groupByRaw("
                CASE
                    WHEN user_id IS NULL THEN 'global_legacy'
                    ELSE 'user_scoped'
                END
            ")
            ->get();

        foreach ($runRows as $row) {
            $scope = ($row->scope === 'user_scoped') ? 'user_scoped' : 'global_legacy';
            $summary[$scope]['runs_total'] = (int) $row->runs_total;
            $summary[$scope]['latest_run_id'] = $row->latest_run_id === null ? null : (int) $row->latest_run_id;
            $summary[$scope]['latest_run_at'] = $row->latest_run_at;
        }

        return $summary;
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    private function getUserModelOverview(): array
    {
        $users = User::query()
            ->orderByRaw("CASE WHEN role = ? THEN 0 ELSE 1 END", [User::ROLE_ADMIN])
            ->orderBy('id')
            ->get(['id', 'name', 'email', 'role']);

        $modelStatuses = PreferenceModelStatus::query()
            ->whereNotNull('user_id')
            ->get()
            ->keyBy('user_id');

        $scoreCounts = $this->pluckGroupedCounts(UserTrackScore::query(), 'user_id');
        $likedTrackCounts = $this->pluckGroupedCounts(UserTrackLike::query(), 'user_id');
        $likedFragmentCounts = $this->pluckGroupedCounts(Fragment::query()->where('is_liked', true), 'user_id');
        $dislikedFragmentCounts = $this->pluckGroupedCounts(Fragment::query()->where('is_liked', false), 'user_id');
        $segmentCounts = $this->pluckGroupedCounts(UserPatternExampleRating::query(), 'user_id');

        $frequentlyPlayedCounts = DB::query()
            ->fromSub(
                ListeningHistory::query()
                    ->select('user_id', 'track_id')
                    ->selectRaw('COUNT(*) as play_count')
                    ->whereNotNull('user_id')
                    ->groupBy('user_id', 'track_id')
                    ->havingRaw('COUNT(*) >= 2'),
                'frequent_tracks'
            )
            ->select('user_id')
            ->selectRaw('COUNT(*) as aggregate')
            ->groupBy('user_id')
            ->pluck('aggregate', 'user_id');

        $trainingTaskRows = DB::table('queue_tasks')
            ->selectRaw("NULLIF(payload->>'user_id', '')::bigint as user_id, status, COUNT(*) as aggregate")
            ->where('task_type', QueueTask::TYPE_TRAINING)
            ->whereRaw("jsonb_exists(payload::jsonb, 'user_id')")
            ->groupByRaw("NULLIF(payload->>'user_id', '')::bigint, status")
            ->get();

        /** @var array<int, array<string, int>> $trainingTaskMap */
        $trainingTaskMap = [];
        foreach ($trainingTaskRows as $row) {
            if ($row->user_id === null) {
                continue;
            }

            $userId = (int) $row->user_id;
            $trainingTaskMap[$userId] ??= [
                QueueTask::STATUS_PENDING => 0,
                QueueTask::STATUS_PROCESSING => 0,
                QueueTask::STATUS_COMPLETED => 0,
                QueueTask::STATUS_FAILED => 0,
            ];
            $trainingTaskMap[$userId][$row->status] = (int) $row->aggregate;
        }

        $latestRunRows = DB::table('preference_model_training_runs as r')
            ->leftJoin('queue_tasks as q', 'q.id', '=', 'r.queue_task_id')
            ->whereNotNull('r.user_id')
            ->select([
                'r.id',
                'r.user_id',
                'r.queue_task_id',
                'r.model_path',
                'r.started_at',
                'r.finished_at',
                'r.final_loss',
                'r.positives_total',
                'r.explicit_negatives_total',
                'q.status as queue_task_status',
            ])
            ->orderByDesc('r.id')
            ->get()
            ->groupBy('user_id')
            ->map(fn (Collection $rows) => $rows->first());

        return $users->map(function (User $user) use (
            $modelStatuses,
            $scoreCounts,
            $likedTrackCounts,
            $likedFragmentCounts,
            $dislikedFragmentCounts,
            $segmentCounts,
            $frequentlyPlayedCounts,
            $trainingTaskMap,
            $latestRunRows
        ) {
            /** @var PreferenceModelStatus|null $status */
            $status = $modelStatuses->get($user->id);
            $taskCounts = $trainingTaskMap[$user->id] ?? [
                QueueTask::STATUS_PENDING => 0,
                QueueTask::STATUS_PROCESSING => 0,
                QueueTask::STATUS_COMPLETED => 0,
                QueueTask::STATUS_FAILED => 0,
            ];
            $latestRun = $latestRunRows->get($user->id);

            return [
                'user_id' => $user->id,
                'name' => $user->name,
                'email' => $user->email,
                'role' => $user->role,
                'model_status' => $status?->status ?? 'not_trained',
                'model_trained_at' => $status?->trained_at?->toIso8601String(),
                'model_path' => $status?->model_path,
                'likes_count' => $status?->likes_count ?? 0,
                'score_count' => (int) ($scoreCounts[$user->id] ?? 0),
                'liked_tracks' => (int) ($likedTrackCounts[$user->id] ?? 0),
                'liked_fragments' => (int) ($likedFragmentCounts[$user->id] ?? 0),
                'disliked_fragments' => (int) ($dislikedFragmentCounts[$user->id] ?? 0),
                'frequently_played_tracks' => (int) ($frequentlyPlayedCounts[$user->id] ?? 0),
                'rated_segments' => (int) ($segmentCounts[$user->id] ?? 0),
                'training_tasks_pending' => $taskCounts[QueueTask::STATUS_PENDING],
                'training_tasks_processing' => $taskCounts[QueueTask::STATUS_PROCESSING],
                'training_tasks_completed' => $taskCounts[QueueTask::STATUS_COMPLETED],
                'training_tasks_failed' => $taskCounts[QueueTask::STATUS_FAILED],
                'latest_training_run' => $latestRun === null ? null : [
                    'id' => (int) $latestRun->id,
                    'queue_task_id' => $latestRun->queue_task_id === null ? null : (int) $latestRun->queue_task_id,
                    'queue_task_status' => $latestRun->queue_task_status,
                    'model_path' => $latestRun->model_path,
                    'started_at' => $latestRun->started_at,
                    'finished_at' => $latestRun->finished_at,
                    'final_loss' => $latestRun->final_loss === null ? null : (float) $latestRun->final_loss,
                    'positives_total' => (int) $latestRun->positives_total,
                    'explicit_negatives_total' => (int) $latestRun->explicit_negatives_total,
                ],
            ];
        })->all();
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    private function getRecentTrainingRuns(): array
    {
        return DB::table('preference_model_training_runs as r')
            ->leftJoin('users as u', 'u.id', '=', 'r.user_id')
            ->leftJoin('queue_tasks as q', 'q.id', '=', 'r.queue_task_id')
            ->select([
                'r.id',
                'r.user_id',
                'u.name as user_name',
                'u.email as user_email',
                'r.queue_task_id',
                'q.status as queue_task_status',
                'r.model_path',
                'r.final_loss',
                'r.positives_total',
                'r.positives_tracks',
                'r.positives_fragments',
                'r.positives_segments',
                'r.explicit_negatives_total',
                'r.started_at',
                'r.finished_at',
                'r.created_at',
            ])
            ->orderByDesc('r.id')
            ->limit(8)
            ->get()
            ->map(function ($row) {
                return [
                    'id' => (int) $row->id,
                    'scope' => $row->user_id === null ? 'global_legacy' : 'user_scoped',
                    'user_id' => $row->user_id === null ? null : (int) $row->user_id,
                    'user_name' => $row->user_name,
                    'user_email' => $row->user_email,
                    'queue_task_id' => $row->queue_task_id === null ? null : (int) $row->queue_task_id,
                    'queue_task_status' => $row->queue_task_status,
                    'model_path' => $row->model_path,
                    'final_loss' => $row->final_loss === null ? null : (float) $row->final_loss,
                    'positives_total' => (int) $row->positives_total,
                    'positives_tracks' => (int) $row->positives_tracks,
                    'positives_fragments' => (int) $row->positives_fragments,
                    'positives_segments' => (int) $row->positives_segments,
                    'explicit_negatives_total' => (int) $row->explicit_negatives_total,
                    'started_at' => $row->started_at,
                    'finished_at' => $row->finished_at,
                    'created_at' => $row->created_at,
                ];
            })
            ->all();
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    private function getRecentTrainingTasks(): array
    {
        return DB::table('queue_tasks as q')
            ->leftJoin('users as u', function ($join) {
                $join->on('u.id', '=', DB::raw("NULLIF(q.payload->>'user_id', '')::bigint"));
            })
            ->where('q.task_type', QueueTask::TYPE_TRAINING)
            ->select([
                'q.id',
                'q.status',
                'q.error_message',
                'q.created_at',
                'q.completed_at',
                DB::raw("NULLIF(q.payload->>'user_id', '')::bigint as user_id"),
                'u.name as user_name',
                'u.email as user_email',
            ])
            ->orderByDesc('q.id')
            ->limit(8)
            ->get()
            ->map(function ($row) {
                return [
                    'id' => (int) $row->id,
                    'scope' => $row->user_id === null ? 'global_legacy' : 'user_scoped',
                    'user_id' => $row->user_id === null ? null : (int) $row->user_id,
                    'user_name' => $row->user_name,
                    'user_email' => $row->user_email,
                    'status' => $row->status,
                    'error_message' => $row->error_message,
                    'created_at' => $row->created_at,
                    'completed_at' => $row->completed_at,
                ];
            })
            ->all();
    }

    private function getTrainingTaskOrFail(int $taskId): QueueTask
    {
        return QueueTask::query()
            ->where('id', $taskId)
            ->where('task_type', QueueTask::TYPE_TRAINING)
            ->firstOrFail();
    }

    private function extractTrainingTaskUserId(QueueTask $task): ?int
    {
        $userId = $task->payload['user_id'] ?? null;

        if ($userId === null || $userId === '') {
            return null;
        }

        return (int) $userId;
    }

    private function getLatestTrainingTaskForUser(?int $userId): ?QueueTask
    {
        $query = QueueTask::query()->where('task_type', QueueTask::TYPE_TRAINING);

        if ($userId === null) {
            $query->whereRaw("NOT jsonb_exists(payload::jsonb, 'user_id') OR NULLIF(payload->>'user_id', '') IS NULL");
        } else {
            $query->whereRaw("NULLIF(payload->>'user_id', '')::bigint = ?", [$userId]);
        }

        /** @var QueueTask|null $task */
        $task = $query->orderByDesc('id')->first();
        return $task;
    }

    private function getLatestTrainingTaskIdForUser(?int $userId): ?int
    {
        return $this->getLatestTrainingTaskForUser($userId)?->id;
    }

    private function markModelStatusQueued(?int $userId): void
    {
        $status = PreferenceModelStatus::forUser($userId)->first();

        if (!$status) {
            PreferenceModelStatus::create([
                'user_id' => $userId,
                'status' => PreferenceModelStatus::STATUS_TRAINING,
                'likes_count' => 0,
            ]);
            return;
        }

        $status->update([
            'status' => PreferenceModelStatus::STATUS_TRAINING,
        ]);
    }

    private function restoreModelStatusIfNoActiveTraining(?int $userId, int $excludedTaskId): void
    {
        if ($this->hasActiveTrainingTaskForScope($userId, $excludedTaskId)) {
            return;
        }

        $status = PreferenceModelStatus::forUser($userId)->first();
        if (!$status || $status->status !== PreferenceModelStatus::STATUS_TRAINING) {
            return;
        }

        $nextStatus = ($status->model_path || $status->trained_at)
            ? PreferenceModelStatus::STATUS_TRAINED
            : PreferenceModelStatus::STATUS_NOT_TRAINED;

        $status->update([
            'status' => $nextStatus,
        ]);
    }

    private function hasActiveTrainingTaskForScope(?int $userId, ?int $excludedTaskId = null): bool
    {
        return QueueTask::query()
            ->where('task_type', QueueTask::TYPE_TRAINING)
            ->when($excludedTaskId !== null, fn ($query) => $query->where('id', '!=', $excludedTaskId))
            ->whereIn('status', [QueueTask::STATUS_PENDING, QueueTask::STATUS_PROCESSING])
            ->when(
                $userId === null,
                fn ($query) => $query->whereRaw("NOT jsonb_exists(payload::jsonb, 'user_id') OR NULLIF(payload->>'user_id', '') IS NULL"),
                fn ($query) => $query->whereRaw("NULLIF(payload->>'user_id', '')::bigint = ?", [$userId])
            )
            ->exists();
    }

    /**
     * @param \Illuminate\Database\Eloquent\Builder<\Illuminate\Database\Eloquent\Model> $query
     * @return \Illuminate\Support\Collection<int, int>
     */
    private function pluckGroupedCounts($query, string $groupBy): Collection
    {
        return $query
            ->whereNotNull($groupBy)
            ->select($groupBy)
            ->selectRaw('COUNT(*) as aggregate')
            ->groupBy($groupBy)
            ->pluck('aggregate', $groupBy);
    }

    /**
     * @return array<string, mixed>
     */
    private function getRadioDiagnosticsOverview(): array
    {
        $logFiles = $this->getRecentRadioDiagnosticsLogFiles();
        $records = collect();

        foreach ($logFiles as $path) {
            foreach ($this->readRecentRadioDiagnosticsLines($path, self::RADIO_DIAGNOSTICS_LOG_TAIL_LINES) as $line) {
                $decoded = json_decode($line, true);
                if (!is_array($decoded)) {
                    continue;
                }

                $records->push($decoded);
            }
        }

        $flattenedEvents = collect();
        $droppedEvents = 0;
        $batchesSeen = 0;
        $throttledBatches = 0;
        $knownClients = [];

        foreach ($records as $record) {
            $message = (string) ($record['message'] ?? '');
            $context = is_array($record['context'] ?? null) ? $record['context'] : [];

            if ($message === 'public-radio-client-diagnostics') {
                $batchesSeen++;
                $clientId = isset($context['client_id']) ? (string) $context['client_id'] : null;
                if ($clientId !== null && $clientId !== '') {
                    $knownClients[$clientId] = true;
                }

                foreach ($this->flattenRadioDiagnosticsRecord($record) as $event) {
                    $flattenedEvents->push($event);
                }

                continue;
            }

            if ($message === 'public-radio-client-diagnostics-throttled') {
                $throttledBatches++;
                $droppedEvents += (int) ($context['dropped_count'] ?? 0);
                $clientId = isset($context['client_id']) ? (string) $context['client_id'] : null;
                if ($clientId !== null && $clientId !== '') {
                    $knownClients[$clientId] = true;
                }
            }
        }

        $sortedEvents = $flattenedEvents
            ->sortByDesc(fn (array $event): string => (string) ($event['logged_at'] ?? $event['timestamp'] ?? ''))
            ->values();

        $driftValues = $sortedEvents
            ->pluck('drift')
            ->filter(static fn ($value): bool => is_numeric($value))
            ->map(static fn ($value): float => (float) $value)
            ->values();

        $topEvents = $sortedEvents
            ->pluck('event')
            ->filter(static fn ($value): bool => is_string($value) && $value !== '')
            ->countBy()
            ->sortDesc()
            ->take(6)
            ->map(fn (int $count, string $event): array => [
                'event' => $event,
                'count' => $count,
            ])
            ->values()
            ->all();

        $uniqueTrackIds = $sortedEvents
            ->pluck('track_id')
            ->filter(static fn ($value): bool => is_int($value) && $value > 0)
            ->unique()
            ->values();

        return [
            'available' => $sortedEvents->isNotEmpty() || $droppedEvents > 0,
            'log_files' => array_map(static fn (string $path): string => basename($path), $logFiles),
            'batches_seen' => $batchesSeen,
            'throttled_batches' => $throttledBatches,
            'total_events' => $sortedEvents->count(),
            'dropped_events' => $droppedEvents,
            'unique_clients' => count($knownClients),
            'unique_tracks' => $uniqueTrackIds->count(),
            'waiting_events' => $sortedEvents->where('event', 'waiting')->count(),
            'stalled_events' => $sortedEvents->where('event', 'stalled')->count(),
            'error_events' => $sortedEvents->filter(static function (array $event): bool {
                $name = (string) ($event['event'] ?? '');
                return $name === 'audio-error' || str_starts_with($name, 'hls-error:');
            })->count(),
            'fallback_events' => $sortedEvents->where('event', 'native-hls-fallback-armed')->count(),
            'average_drift' => $driftValues->isEmpty() ? null : round($driftValues->avg(), 3),
            'max_drift' => $driftValues->isEmpty() ? null : round($driftValues->max(), 3),
            'latest_logged_at' => $sortedEvents->first()['logged_at'] ?? null,
            'top_events' => $topEvents,
            'recent_events' => $sortedEvents->take(self::RADIO_DIAGNOSTICS_RECENT_EVENT_LIMIT)->all(),
        ];
    }

    /**
     * @return array<int, string>
     */
    private function getRecentRadioDiagnosticsLogFiles(): array
    {
        $patterns = [
            AppStoragePath::log('public-radio-diagnostics-*.log'),
            AppStoragePath::log('public-radio-diagnostics.log'),
        ];

        $paths = [];
        foreach ($patterns as $pattern) {
            foreach (glob($pattern) ?: [] as $path) {
                if (is_file($path)) {
                    $paths[$path] = filemtime($path) ?: 0;
                }
            }
        }

        arsort($paths);

        return array_slice(array_keys($paths), 0, self::RADIO_DIAGNOSTICS_LOG_FILE_LIMIT);
    }

    /**
     * @return array<int, string>
     */
    private function readRecentRadioDiagnosticsLines(string $path, int $limit): array
    {
        $lines = @file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        if (!is_array($lines) || $lines === []) {
            return [];
        }

        return array_slice($lines, -$limit);
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    private function flattenRadioDiagnosticsRecord(array $record): array
    {
        $context = is_array($record['context'] ?? null) ? $record['context'] : [];
        $events = is_array($context['events'] ?? null) ? $context['events'] : [];
        $loggedAt = isset($record['datetime']) ? (string) $record['datetime'] : null;
        $stationSlug = isset($context['station_slug']) ? (string) $context['station_slug'] : null;
        $clientId = isset($context['client_id']) ? (string) $context['client_id'] : null;
        $clientIp = isset($context['client_ip']) ? (string) $context['client_ip'] : null;
        $userAgent = isset($context['user_agent']) ? (string) $context['user_agent'] : null;
        $viewerId = isset($context['viewer_id']) ? (int) $context['viewer_id'] : null;

        $flattened = [];
        foreach ($events as $event) {
            if (!is_array($event)) {
                continue;
            }

            $flattened[] = [
                'logged_at' => $loggedAt,
                'station_slug' => $stationSlug,
                'viewer_id' => $viewerId,
                'client_id' => $clientId,
                'client_ip' => $clientIp,
                'user_agent' => $userAgent,
                'event' => isset($event['event']) ? (string) $event['event'] : '',
                'timestamp' => isset($event['timestamp']) ? (string) $event['timestamp'] : null,
                'playback_mode' => isset($event['playback_mode']) ? (string) $event['playback_mode'] : null,
                'transport_preference' => isset($event['transport_preference']) ? (string) $event['transport_preference'] : null,
                'selected_bitrate' => isset($event['selected_bitrate']) ? (string) $event['selected_bitrate'] : null,
                'track_id' => isset($event['track_id']) ? (int) $event['track_id'] : null,
                'drift' => isset($event['drift']) ? (float) $event['drift'] : null,
                'playback_rate' => isset($event['playback_rate']) ? (float) $event['playback_rate'] : null,
                'media_error_code' => isset($event['media_error_code']) ? (int) $event['media_error_code'] : null,
                'waiting' => array_key_exists('waiting', $event) ? (bool) $event['waiting'] : null,
                'paused' => array_key_exists('paused', $event) ? (bool) $event['paused'] : null,
                'ready_state' => isset($event['ready_state']) ? (int) $event['ready_state'] : null,
                'current_src' => isset($event['current_src']) ? (string) $event['current_src'] : null,
            ];
        }

        return $flattened;
    }
}
