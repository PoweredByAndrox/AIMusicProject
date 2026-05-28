<?php

namespace App\Services;

use App\Models\Fragment;
use App\Models\ListeningHistory;
use App\Models\PreferenceModelStatus;
use App\Models\QueueTask;
use App\Models\TrackRating;
use App\Models\UserPatternExampleRating;
use App\Models\UserTrackLike;
use Illuminate\Support\Facades\DB;

/**
 * PreferenceTrainingService
 *
 * Automatically triggers preference model training after any feedback change.
 * Uses multiple signals in legacy global mode and user ratings in scoped mode.
 */
class PreferenceTrainingService
{
    /**
     * Minimum play count to consider a track as "frequently played"
     */
    private const MIN_PLAY_COUNT = 2;

    /**
     * Trigger model training after any feedback change.
     * Debounces rapid changes by checking if already training.
     */
    public static function triggerTraining(?int $userId = null): void
    {
        if ($userId !== null) {
            self::triggerUserScopedTraining($userId);
            return;
        }

        $likedTracks = self::normalizeTrackIds(UserTrackLike::query()->pluck('track_id')->toArray());
        $likedFragments = Fragment::liked()->get(['id', 'track_id', 'start_time', 'end_time', 'comment'])->toArray();
        $dislikedFragments = Fragment::disliked()->get(['id', 'track_id', 'start_time', 'end_time', 'comment'])->toArray();
        $ratedTracks = self::normalizeTrackIds(TrackRating::where('rating', '>=', 4)->pluck('track_id')->toArray());
        $lowRatedTracks = self::normalizeTrackIds(TrackRating::where('rating', '<=', 2)->pluck('track_id')->toArray());

        $frequentlyPlayed = self::normalizeTrackIds(ListeningHistory::select('track_id')
            ->selectRaw('COUNT(*) as play_count')
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= ?', [self::MIN_PLAY_COUNT])
            ->pluck('track_id')
            ->toArray());

        $likedTracksForPayload = self::excludeTrackIds($likedTracks, $ratedTracks);

        $playCounts = ListeningHistory::select('track_id')
            ->selectRaw('COUNT(*) as play_count')
            ->groupBy('track_id')
            ->pluck('play_count', 'track_id')
            ->toArray();

        $positiveTracks = self::mergeTrackIds($likedTracks, $ratedTracks, $frequentlyPlayed);
        $totalPositive = count($positiveTracks) + count($likedFragments);

        if ($totalPositive < 1) {
            return;
        }

        $status = PreferenceModelStatus::forUser(null)->first();
        if (!$status) {
            $status = PreferenceModelStatus::create([
                'user_id' => null,
                'status' => 'not_trained',
                'likes_count' => $totalPositive,
            ]);
        }

        if ($status->status === 'training') {
            $stuckMinutes = 10;
            $isStuck = $status->updated_at && $status->updated_at->lt(now()->subMinutes($stuckMinutes));

            if (!$isStuck) {
                return;
            }

            $lastTrainingTask = QueueTask::where('task_type', QueueTask::TYPE_TRAINING)
                ->orderBy('created_at', 'desc')
                ->first();

            if ($lastTrainingTask && in_array($lastTrainingTask->status, ['failed', 'completed'], true)) {
                \Log::warning("Preference model stuck in 'training' for {$stuckMinutes}+ min, last task status: {$lastTrainingTask->status}. Resetting.");
                $status->status = $lastTrainingTask->status === 'completed' ? 'trained' : 'not_trained';
                $status->save();

                if ($lastTrainingTask->status === 'completed') {
                    return;
                }
            } else {
                return;
            }
        }

        $locked = DB::table('preference_model_status')
            ->where('id', $status->id)
            ->whereNull('user_id')
            ->where('status', '!=', 'training')
            ->update([
                'status' => 'training',
                'likes_count' => $totalPositive,
                'updated_at' => now(),
            ]);

        if ($locked === 0) {
            return;
        }

        // Skipped tracks — strong negative signal (early skips = user dislike)
        $skipData = DB::table('track_skips')
            ->select('track_id')
            ->selectRaw('COUNT(*) as skip_count')
            ->selectRaw('AVG(position_at_skip) as avg_skip_position')
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= 2')
            ->get()
            ->keyBy('track_id')
            ->map(fn ($row) => [
                'skip_count' => (int) $row->skip_count,
                'avg_skip_position' => round((float) $row->avg_skip_position, 1),
            ])
            ->toArray();

        $skippedTrackIds = array_keys($skipData);

        // Completion rates — tracks played to end are stronger positive signals
        $completionRates = DB::table('listening_history')
            ->select('track_id')
            ->selectRaw('COUNT(*) as total_plays')
            ->selectRaw('SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed_plays')
            ->groupBy('track_id')
            ->get()
            ->keyBy('track_id')
            ->map(fn ($row) => [
                'total_plays' => (int) $row->total_plays,
                'completed_plays' => (int) $row->completed_plays,
                'completion_rate' => $row->total_plays > 0
                    ? round((int) $row->completed_plays / (int) $row->total_plays, 2)
                    : 0.0,
            ])
            ->toArray();

        $queueTask = QueueTask::create([
            'task_type' => QueueTask::TYPE_TRAINING,
            'payload' => [
                'liked_track_ids' => $likedTracksForPayload,
                'liked_fragments' => $likedFragments,
                'disliked_fragments' => $dislikedFragments,
                'high_rated_track_ids' => $ratedTracks,
                'low_rated_track_ids' => $lowRatedTracks,
                'frequently_played_track_ids' => $frequentlyPlayed,
                'play_counts' => $playCounts,
                'skipped_track_ids' => $skippedTrackIds,
                'skip_data' => $skipData,
                'completion_rates' => $completionRates,
            ],
            'status' => QueueTask::STATUS_PENDING,
        ]);

        MLQueueService::pushModelTraining($queueTask->id);
    }

    /**
     * Get current feedback stats.
     */
    public static function getStats(?int $userId = null): array
    {
        if ($userId !== null) {
            return self::getUserScopedStats($userId);
        }

        $likedTrackIds = self::normalizeTrackIds(UserTrackLike::query()->pluck('track_id')->toArray());
        $likedTracks = count($likedTrackIds);
        $likedFragments = Fragment::liked()->count();
        $dislikedFragments = Fragment::disliked()->count();
        $fragmentsWithComments = Fragment::whereNotNull('comment')->count();
        $highRatedTrackIds = self::normalizeTrackIds(TrackRating::where('rating', '>=', 4)->pluck('track_id')->toArray());
        $highRatedTracks = count($highRatedTrackIds);
        $lowRatedTracks = TrackRating::where('rating', '<=', 2)->count();
        $frequentlyPlayedTrackIds = self::normalizeTrackIds(ListeningHistory::select('track_id')
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= ?', [self::MIN_PLAY_COUNT])
            ->pluck('track_id')
            ->toArray());
        $frequentlyPlayed = count($frequentlyPlayedTrackIds);

        $totalPlays = ListeningHistory::count();

        $ratedSegments = DB::table('pattern_examples')
            ->whereNotNull('rating')
            ->count();
        $highRatedSegments = DB::table('pattern_examples')
            ->where('rating', '>=', 4)
            ->count();
        $avgSegmentRating = DB::table('pattern_examples')
            ->whereNotNull('rating')
            ->avg('rating');

        return [
            'liked_tracks' => $likedTracks,
            'liked_fragments' => $likedFragments,
            'disliked_fragments' => $dislikedFragments,
            'fragments_with_comments' => $fragmentsWithComments,
            'high_rated_tracks' => $highRatedTracks,
            'low_rated_tracks' => $lowRatedTracks,
            'frequently_played' => $frequentlyPlayed,
            'total_plays' => $totalPlays,
            'rated_segments' => $ratedSegments,
            'high_rated_segments' => $highRatedSegments,
            'avg_segment_rating' => $avgSegmentRating ? round($avgSegmentRating, 1) : null,
            'total_positive' => count(self::mergeTrackIds($likedTrackIds, $highRatedTrackIds, $frequentlyPlayedTrackIds))
                + $likedFragments
                + $highRatedSegments,
            'training_scope' => 'global_feedback',
        ];
    }

    private static function triggerUserScopedTraining(int $userId): void
    {
        $stats = self::getUserScopedStats($userId);
        if ($stats['total_positive'] < 1) {
            return;
        }

        $status = PreferenceModelStatus::forUser($userId)->first();
        if (!$status) {
            $status = PreferenceModelStatus::create([
                'user_id' => $userId,
                'status' => 'not_trained',
                'likes_count' => $stats['total_positive'],
            ]);
        }

        if ($status->status === 'training') {
            $stuckMinutes = 10;
            $isStuck = $status->updated_at && $status->updated_at->lt(now()->subMinutes($stuckMinutes));

            if (!$isStuck) {
                return;
            }

            $lastTrainingTask = QueueTask::where('task_type', QueueTask::TYPE_TRAINING)
                ->whereRaw("(payload->>'user_id')::bigint = ?", [$userId])
                ->orderBy('created_at', 'desc')
                ->first();

            if ($lastTrainingTask && in_array($lastTrainingTask->status, ['failed', 'completed'], true)) {
                \Log::warning("Preference model for user {$userId} stuck in 'training' for {$stuckMinutes}+ min, last task status: {$lastTrainingTask->status}. Resetting.");
                $status->status = $lastTrainingTask->status === 'completed' ? 'trained' : 'not_trained';
                $status->save();

                if ($lastTrainingTask->status === 'completed') {
                    return;
                }
            } else {
                return;
            }
        }

        $locked = DB::table('preference_model_status')
            ->where('id', $status->id)
            ->where('user_id', $userId)
            ->where('status', '!=', 'training')
            ->update([
                'status' => 'training',
                'likes_count' => $stats['total_positive'],
                'updated_at' => now(),
            ]);

        if ($locked === 0) {
            return;
        }

        // User-scoped skip data
        $skipData = DB::table('track_skips')
            ->where('user_id', $userId)
            ->select('track_id')
            ->selectRaw('COUNT(*) as skip_count')
            ->selectRaw('AVG(position_at_skip) as avg_skip_position')
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= 2')
            ->get()
            ->keyBy('track_id')
            ->map(fn ($row) => [
                'skip_count' => (int) $row->skip_count,
                'avg_skip_position' => round((float) $row->avg_skip_position, 1),
            ])
            ->toArray();

        // User-scoped completion rates
        $completionRates = ListeningHistory::forUser($userId)
            ->select('track_id')
            ->selectRaw('COUNT(*) as total_plays')
            ->selectRaw('SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed_plays')
            ->groupBy('track_id')
            ->get()
            ->keyBy('track_id')
            ->map(fn ($row) => [
                'total_plays' => (int) $row->total_plays,
                'completed_plays' => (int) $row->completed_plays,
                'completion_rate' => $row->total_plays > 0
                    ? round((int) $row->completed_plays / (int) $row->total_plays, 2)
                    : 0.0,
            ])
            ->toArray();

        $likedTrackIds = self::normalizeTrackIds(
            UserTrackLike::forUser($userId)
                ->pluck('track_id')
                ->all()
        );
        $highRatedTrackIds = self::normalizeTrackIds(
            TrackRating::forUser($userId)
                ->where('rating', '>=', 4)
                ->pluck('track_id')
                ->all()
        );
        $lowRatedTrackIds = self::normalizeTrackIds(
            TrackRating::forUser($userId)
                ->where('rating', '<=', 2)
                ->pluck('track_id')
                ->all()
        );
        $frequentlyPlayedTrackIds = self::normalizeTrackIds(
            ListeningHistory::forUser($userId)
                ->select('track_id')
                ->selectRaw('COUNT(*) as play_count')
                ->groupBy('track_id')
                ->havingRaw('COUNT(*) >= ?', [self::MIN_PLAY_COUNT])
                ->pluck('track_id')
                ->all()
        );
        $likedTrackIdsForPayload = self::excludeTrackIds($likedTrackIds, $highRatedTrackIds);

        $queueTask = QueueTask::create([
            'task_type' => QueueTask::TYPE_TRAINING,
            'payload' => [
                'user_id' => $userId,
                'liked_track_ids' => $likedTrackIdsForPayload,
                'liked_fragments' => Fragment::forUser($userId)->liked()
                    ->get(['id', 'track_id', 'start_time', 'end_time', 'comment'])
                    ->toArray(),
                'disliked_fragments' => Fragment::forUser($userId)->disliked()
                    ->get(['id', 'track_id', 'start_time', 'end_time', 'comment'])
                    ->toArray(),
                'high_rated_track_ids' => $highRatedTrackIds,
                'low_rated_track_ids' => $lowRatedTrackIds,
                'frequently_played_track_ids' => $frequentlyPlayedTrackIds,
                'play_counts' => ListeningHistory::forUser($userId)
                    ->select('track_id')
                    ->selectRaw('COUNT(*) as play_count')
                    ->groupBy('track_id')
                    ->pluck('play_count', 'track_id')
                    ->toArray(),
                'skipped_track_ids' => array_keys($skipData),
                'skip_data' => $skipData,
                'completion_rates' => $completionRates,
                'training_scope' => 'user_scoped_feedback',
            ],
            'status' => QueueTask::STATUS_PENDING,
        ]);

        MLQueueService::pushModelTraining($queueTask->id, $userId);
    }

    private static function getUserScopedStats(int $userId): array
    {
        $likedTrackIds = self::normalizeTrackIds(
            UserTrackLike::forUser($userId)
                ->pluck('track_id')
                ->all()
        );
        $likedTracks = count($likedTrackIds);
        $likedFragments = Fragment::forUser($userId)->liked()->count();
        $dislikedFragments = Fragment::forUser($userId)->disliked()->count();
        $fragmentsWithComments = Fragment::forUser($userId)->whereNotNull('comment')->count();
        $highRatedTrackIds = self::normalizeTrackIds(
            TrackRating::forUser($userId)
                ->where('rating', '>=', 4)
                ->pluck('track_id')
                ->all()
        );
        $highRatedTracks = count($highRatedTrackIds);

        $lowRatedTracks = TrackRating::forUser($userId)
            ->where('rating', '<=', 2)
            ->count();

        $frequentlyPlayedTrackIds = self::normalizeTrackIds(ListeningHistory::forUser($userId)
            ->select('track_id')
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= ?', [self::MIN_PLAY_COUNT])
            ->pluck('track_id')
            ->all());
        $frequentlyPlayed = count($frequentlyPlayedTrackIds);

        $totalPlays = ListeningHistory::forUser($userId)->count();

        $ratedSegments = UserPatternExampleRating::forUser($userId)->count();
        $highRatedSegments = UserPatternExampleRating::forUser($userId)
            ->where('rating', '>=', 4)
            ->count();
        $avgSegmentRating = UserPatternExampleRating::forUser($userId)->avg('rating');

        return [
            'liked_tracks' => $likedTracks,
            'liked_fragments' => $likedFragments,
            'disliked_fragments' => $dislikedFragments,
            'fragments_with_comments' => $fragmentsWithComments,
            'high_rated_tracks' => $highRatedTracks,
            'low_rated_tracks' => $lowRatedTracks,
            'frequently_played' => $frequentlyPlayed,
            'total_plays' => $totalPlays,
            'rated_segments' => $ratedSegments,
            'high_rated_segments' => $highRatedSegments,
            'avg_segment_rating' => $avgSegmentRating ? round($avgSegmentRating, 1) : null,
            'total_positive' => count(self::mergeTrackIds($likedTrackIds, $highRatedTrackIds, $frequentlyPlayedTrackIds))
                + $likedFragments
                + $highRatedSegments,
            'training_scope' => 'user_scoped_feedback',
        ];
    }

    /**
     * @param array<int, int|string> $trackIds
     * @return array<int, int>
     */
    private static function normalizeTrackIds(array $trackIds): array
    {
        return array_values(array_unique(array_map(static fn ($id): int => (int) $id, $trackIds)));
    }

    /**
     * @param array<int, int|string> $trackIds
     * @param array<int, int|string> $excludedTrackIds
     * @return array<int, int>
     */
    private static function excludeTrackIds(array $trackIds, array $excludedTrackIds): array
    {
        $excluded = array_flip(self::normalizeTrackIds($excludedTrackIds));

        return array_values(array_filter(
            self::normalizeTrackIds($trackIds),
            static fn (int $trackId): bool => !isset($excluded[$trackId])
        ));
    }

    /**
     * @param array<int, int|string> ...$trackIdSets
     * @return array<int, int>
     */
    private static function mergeTrackIds(array ...$trackIdSets): array
    {
        $merged = [];

        foreach ($trackIdSets as $trackIdSet) {
            $merged = array_merge($merged, self::normalizeTrackIds($trackIdSet));
        }

        return array_values(array_unique($merged));
    }
}
