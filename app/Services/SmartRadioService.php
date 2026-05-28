<?php

namespace App\Services;

use App\Models\RadioFeedback;
use App\Models\ListeningHistory;
use App\Models\PatternExample;
use App\Models\Track;
use App\Models\TrackRating;
use App\Models\UserPatternExampleRating;
use App\Models\UserTrackScore;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Cache;

/**
 * SmartRadioService
 * 
 * Handles Smart Radio functionality including enabling/disabling radio mode,
 * generating recommendations, and recording user feedback.
 * 
 * Requirements: 15.1, 15.2, 15.3, 17.1, 17.2, 17.4
 */
class SmartRadioService
{
    /**
     * Cache key for Smart Radio state.
     */
    private const STATE_CACHE_KEY = 'smart_radio_state';

    /**
     * Cache TTL in seconds (24 hours).
     */
    private const STATE_CACHE_TTL = 86400;

    /**
     * Minimum number of tracks to maintain in queue.
     * Requirement 15.2: WHEN Queue содержит менее 3 треков THEN THE Smart_Radio SHALL добавить новые рекомендованные треки
     */
    private const MIN_QUEUE_SIZE = 3;

    /**
     * Hours to exclude recently played tracks.
     * Requirement 16.1: THE Smart_Radio SHALL исключать треки, воспроизведённые за последние 2 часа
     */
    private const EXCLUDE_HOURS = 2;

    /**
     * Hours for reduced priority tracks.
     * Requirement 16.2: THE Smart_Radio SHALL снижать приоритет треков, воспроизведённых за последние 24 часа
     */
    private const REDUCED_PRIORITY_HOURS = 24;

    /**
     * Weight adjustment for "more" feedback.
     * Requirement 17.1: WHEN пользователь нажимает "Больше такого" THEN THE Smart_Radio SHALL увеличить вес текущего трека
     */
    private const MORE_WEIGHT_ADJUSTMENT = 0.5;

    /**
     * Weight adjustment for "less" feedback.
     * Requirement 17.2: WHEN пользователь нажимает "Меньше такого" THEN THE Smart_Radio SHALL уменьшить вес текущего трека
     */
    private const LESS_WEIGHT_ADJUSTMENT = -0.5;

    public function __construct(
        private readonly ListeningHistoryService $historyService,
        private readonly SkipTrackerService $skipService
    ) {}

    /**
     * Enable Smart Radio mode with a seed track.
     * 
     * Requirement 15.1: WHEN пользователь включает Smart_Radio THEN THE System SHALL 
     * автоматически добавлять треки в Queue
     *
     * @param int $seedTrackId The ID of the seed track to base recommendations on
     * @return array{enabled: bool, seed_track_id: int, message: string}
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function enable(int $seedTrackId): array
    {
        // Verify seed track exists
        $seedTrack = Track::findOrFail($seedTrackId);

        $state = $this->getState();
        $state['enabled'] = true;
        $state['seedTrackId'] = $seedTrackId;
        $state['enabledAt'] = now()->toIso8601String();

        $this->saveState($state);

        return [
            'enabled' => true,
            'seed_track_id' => $seedTrackId,
            'message' => 'Smart Radio enabled with seed track: ' . $seedTrack->title,
        ];
    }

    /**
     * Disable Smart Radio mode.
     * 
     * Requirement 15.6: WHEN пользователь выключает Smart_Radio THEN THE System SHALL 
     * прекратить автоматическое добавление треков
     *
     * @return array{enabled: bool, message: string}
     */
    public function disable(): array
    {
        $state = $this->getState();
        $state['enabled'] = false;
        $state['seedTrackId'] = null;
        $state['disabledAt'] = now()->toIso8601String();

        $this->saveState($state);

        return [
            'enabled' => false,
            'message' => 'Smart Radio disabled',
        ];
    }

    /**
     * Check if Smart Radio is currently enabled.
     *
     * @return bool
     */
    public function isEnabled(): bool
    {
        return $this->getState()['enabled'];
    }

    /**
     * Get the current seed track ID.
     *
     * @return int|null
     */
    public function getSeedTrackId(): ?int
    {
        return $this->getState()['seedTrackId'];
    }

    /**
     * Get recommendations based on user preferences (likes, ratings, play counts).
     * Works without needing to enable Smart Radio.
     *
     * @param int $count Number of recommendations to return
     * @return Collection<int, Track> Collection of recommended tracks
     */
    public function getPreferenceBasedRecommendations(int $count = 10, ?int $userId = null): Collection
    {
        $resolvedUserId = $userId ?? Auth::id();

        // Get tracks to exclude (played in last 2 hours)
        $excludedTrackIds = $this->historyService->getRecentlyPlayedTrackIds(self::EXCLUDE_HOURS, $resolvedUserId);

        // Get tracks with reduced priority (played in last 24 hours)
        $reducedPriorityTrackIds = $this->historyService->getRecentlyPlayedTrackIds(self::REDUCED_PRIORITY_HOURS, $resolvedUserId);

        // Get penalized tracks (frequently skipped)
        $penalizedTrackIds = $this->skipService->getPenalizedTrackIds();

        // Get liked tracks for preference learning
        $likedTracks = Track::likedByUser($resolvedUserId)->get();
        $likedTrackIds = $likedTracks->pluck('id')->all();
        $likedGenres = $likedTracks->pluck('genre')->filter()->unique()->toArray();
        $likedMoods = $likedTracks->pluck('mood')->filter()->unique()->toArray();
        
        // Get average BPM of liked tracks
        $avgBpm = $likedTracks->avg('bpm');

        // Get high-rated tracks
        $highRatedTrackIds = TrackRating::forUser($resolvedUserId)
            ->where('rating', '>=', 4)
            ->pluck('track_id')
            ->toArray();

        $ratedTrackIds = TrackRating::forUser($resolvedUserId)
            ->pluck('track_id')
            ->unique()
            ->values()
            ->all();

        $ratedSegmentTrackIds = PatternExample::query()
            ->whereIn('id', UserPatternExampleRating::forUser($resolvedUserId)
                ->pluck('pattern_example_id')
                ->unique()
                ->values()
                ->all())
            ->pluck('track_id')
            ->unique()
            ->values()
            ->all();

        // Get frequently played tracks
        $frequentlyPlayed = ListeningHistory::forUser($resolvedUserId)
            ->select('track_id')
            ->selectRaw('COUNT(*) as play_count')
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= 2')
            ->pluck('play_count', 'track_id')
            ->toArray();

        // Build query for all analyzed tracks
        $excludedCandidateTrackIds = array_values(array_unique(array_merge(
            $excludedTrackIds,
            $likedTrackIds,
            $ratedTrackIds,
            $ratedSegmentTrackIds,
        )));

        $candidates = Track::where('analysis_status', 'completed')
            ->whereNotNull('embedding_path')
            ->whereNotIn('id', $excludedCandidateTrackIds)
            ->get();
        $preferenceScores = $resolvedUserId === null
            ? []
            : UserTrackScore::forUser($resolvedUserId)
                ->whereIn('track_id', $candidates->pluck('id')->all())
                ->pluck('preference_score', 'track_id')
                ->toArray();

        // Score each candidate
        $scoredCandidates = $candidates->map(function ($track) use (
            $reducedPriorityTrackIds,
            $penalizedTrackIds,
            $likedTrackIds,
            $likedGenres,
            $likedMoods,
            $avgBpm,
            $highRatedTrackIds,
            $frequentlyPlayed,
            $preferenceScores
        ) {
            $score = array_key_exists($track->id, $preferenceScores)
                ? (float) $preferenceScores[$track->id]
                : 0.5;

            // Boost for liked track
            if (in_array($track->id, $likedTrackIds, true)) {
                $score += 0.3;
            }

            // Boost for high rating
            if (in_array($track->id, $highRatedTrackIds)) {
                $score += 0.25;
            }

            // Boost for frequently played
            if (isset($frequentlyPlayed[$track->id])) {
                $playCount = $frequentlyPlayed[$track->id];
                $score += min(0.2, $playCount * 0.05);
            }

            // Boost for matching genre
            if ($track->genre && in_array($track->genre, $likedGenres)) {
                $score += 0.15;
            }

            // Boost for matching mood
            if ($track->mood && in_array($track->mood, $likedMoods)) {
                $score += 0.1;
            }

            // Boost for similar BPM (within 15%)
            if ($avgBpm && $track->bpm) {
                $bpmDiff = abs($track->bpm - $avgBpm) / $avgBpm;
                if ($bpmDiff < 0.15) {
                    $score += 0.1 * (1 - $bpmDiff / 0.15);
                }
            }

            // Penalty for recently played
            if (in_array($track->id, $reducedPriorityTrackIds)) {
                $score *= 0.5;
            }

            // Penalty for frequently skipped
            if (in_array($track->id, $penalizedTrackIds)) {
                $score *= 0.3;
            }

            // Add some randomness to avoid always same order
            $score += (mt_rand(0, 100) / 1000);

            $track->recommendation_score = max(0, $score);
            return $track;
        });

        // Sort by score and return top N
        return $scoredCandidates
            ->sortByDesc('recommendation_score')
            ->take($count)
            ->values();
    }

    /**
     * Get next track recommendations for Smart Radio.
     * 
     * Requirement 15.2: WHEN Queue содержит менее 3 треков THEN THE Smart_Radio SHALL 
     * добавить новые рекомендованные треки
     * Requirement 15.3: THE Smart_Radio SHALL использовать эмбеддинг текущего трека для поиска похожих
     * Requirement 16.1: THE Smart_Radio SHALL исключать треки, воспроизведённые за последние 2 часа
     * Requirement 16.2: THE Smart_Radio SHALL снижать приоритет треков, воспроизведённых за последние 24 часа
     *
     * @param int $count Number of recommendations to return (default 5)
     * @return Collection<int, Track> Collection of recommended tracks
     */
    public function getNextRecommendations(int $count = 5): Collection
    {
        $state = $this->getState();

        if (!$state['enabled'] || $state['seedTrackId'] === null) {
            return new Collection();
        }

        // Get tracks to exclude (played in last 2 hours)
        $excludedTrackIds = $this->historyService->getRecentlyPlayedTrackIds(self::EXCLUDE_HOURS);

        // Get tracks with reduced priority (played in last 24 hours but not in last 2 hours)
        $reducedPriorityTrackIds = array_diff(
            $this->historyService->getRecentlyPlayedTrackIds(self::REDUCED_PRIORITY_HOURS),
            $excludedTrackIds
        );

        // Get tracks that should be penalized due to skips
        $penalizedTrackIds = $this->skipService->getPenalizedTrackIds();

        // Get feedback weights
        $feedbackWeights = $state['feedbackWeights'] ?? [];

        // Build query for recommendations
        $query = Track::query()
            ->where('id', '!=', $state['seedTrackId'])
            ->whereNotIn('id', $excludedTrackIds)
            ->where('analysis_status', 'completed');

        // Get seed track for similarity comparison
        $seedTrack = Track::find($state['seedTrackId']);

        if ($seedTrack) {
            // Try to find similar tracks based on genre, mood, BPM
            if ($seedTrack->genre) {
                $query->where(function ($q) use ($seedTrack) {
                    $q->where('genre', $seedTrack->genre)
                      ->orWhereNull('genre');
                });
            }

            if ($seedTrack->bpm) {
                // Allow BPM within ±15% range
                $bpmMin = $seedTrack->bpm * 0.85;
                $bpmMax = $seedTrack->bpm * 1.15;
                $query->where(function ($q) use ($bpmMin, $bpmMax) {
                    $q->whereBetween('bpm', [$bpmMin, $bpmMax])
                      ->orWhereNull('bpm');
                });
            }
        }

        // Get all candidate tracks
        $candidates = $query->get();

        // Score and sort candidates
        $scoredCandidates = $candidates->map(function ($track) use (
            $reducedPriorityTrackIds,
            $penalizedTrackIds,
            $feedbackWeights,
            $seedTrack
        ) {
            $score = 1.0;

            // Apply reduced priority for recently played tracks
            if (in_array($track->id, $reducedPriorityTrackIds)) {
                $score *= 0.5;
            }

            // Apply penalty for frequently skipped tracks
            if (in_array($track->id, $penalizedTrackIds)) {
                $score *= 0.3;
            }

            // Apply feedback weights
            if (isset($feedbackWeights[$track->id])) {
                $score += $feedbackWeights[$track->id];
            }

            // Boost tracks with ratings
            if ($track->rating && $track->rating->rating >= 4) {
                $score *= 1.2;
            }

            // Similarity bonus based on genre/mood match
            if ($seedTrack) {
                if ($track->genre === $seedTrack->genre) {
                    $score *= 1.1;
                }
                if ($track->mood === $seedTrack->mood) {
                    $score *= 1.1;
                }
            }

            $track->recommendation_score = max(0, $score);
            return $track;
        });

        // Sort by score and return top N
        return $scoredCandidates
            ->sortByDesc('recommendation_score')
            ->take($count)
            ->values();
    }

    /**
     * Record user feedback for a track.
     * 
     * Requirement 17.1: WHEN пользователь нажимает "Больше такого" THEN THE Smart_Radio 
     * SHALL увеличить вес текущего трека в рекомендациях
     * Requirement 17.2: WHEN пользователь нажимает "Меньше такого" THEN THE Smart_Radio 
     * SHALL уменьшить вес текущего трека в рекомендациях
     * Requirement 17.4: THE System SHALL сохранять обратную связь для обучения Preference_Model
     *
     * @param int $trackId The ID of the track
     * @param string $type The feedback type ('more' or 'less')
     * @return array{track_id: int, feedback_type: string, weight_adjustment: float, new_weight: float}
     * @throws \InvalidArgumentException If feedback type is invalid
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function recordFeedback(int $trackId, string $type): array
    {
        // Validate feedback type
        if (!in_array($type, ['more', 'less'], true)) {
            throw new \InvalidArgumentException('Feedback type must be "more" or "less"');
        }

        // Verify track exists
        Track::findOrFail($trackId);

        // Calculate weight adjustment
        $weightAdjustment = $type === 'more' 
            ? self::MORE_WEIGHT_ADJUSTMENT 
            : self::LESS_WEIGHT_ADJUSTMENT;

        // Update feedback weights in state
        $state = $this->getState();
        $feedbackWeights = $state['feedbackWeights'] ?? [];
        
        $currentWeight = $feedbackWeights[$trackId] ?? 0.0;
        $newWeight = $currentWeight + $weightAdjustment;
        $feedbackWeights[$trackId] = $newWeight;
        
        $state['feedbackWeights'] = $feedbackWeights;
        $this->saveState($state);

        // Save feedback to database for Preference_Model training
        // Requirement 17.4: THE System SHALL сохранять обратную связь для обучения Preference_Model
        RadioFeedback::create([
            'user_id' => Auth::id(),
            'track_id' => $trackId,
            'feedback_type' => $type,
            'created_at' => now(),
        ]);

        return [
            'track_id' => $trackId,
            'feedback_type' => $type,
            'weight_adjustment' => $weightAdjustment,
            'new_weight' => $newWeight,
        ];
    }

    /**
     * Get the current feedback weight for a track.
     *
     * @param int $trackId The ID of the track
     * @return float The current feedback weight
     */
    public function getFeedbackWeight(int $trackId): float
    {
        $state = $this->getState();
        $feedbackWeights = $state['feedbackWeights'] ?? [];
        
        return $feedbackWeights[$trackId] ?? 0.0;
    }

    /**
     * Get all feedback weights.
     *
     * @return array<int, float> Array of track_id => weight
     */
    public function getAllFeedbackWeights(): array
    {
        $state = $this->getState();
        return $state['feedbackWeights'] ?? [];
    }

    /**
     * Get the current Smart Radio state.
     *
     * @return array{enabled: bool, seedTrackId: int|null, feedbackWeights: array<int, float>, enabledAt: string|null, disabledAt: string|null}
     */
    public function getState(): array
    {
        return Cache::get($this->stateCacheKey(), [
            'enabled' => false,
            'seedTrackId' => null,
            'feedbackWeights' => [],
            'enabledAt' => null,
            'disabledAt' => null,
        ]);
    }

    /**
     * Save the Smart Radio state.
     *
     * @param array $state The state to save
     * @return void
     */
    private function saveState(array $state): void
    {
        Cache::put($this->stateCacheKey(), $state, self::STATE_CACHE_TTL);
    }

    /**
     * Reset the Smart Radio state (useful for testing).
     *
     * @return void
     */
    public function resetState(): void
    {
        Cache::forget($this->stateCacheKey());
    }

    private function stateCacheKey(): string
    {
        $userId = Auth::id();

        return $userId === null
            ? self::STATE_CACHE_KEY . ':guest'
            : self::STATE_CACHE_KEY . ':' . $userId;
    }

    /**
     * Get the minimum queue size constant.
     *
     * @return int
     */
    public static function getMinQueueSize(): int
    {
        return self::MIN_QUEUE_SIZE;
    }

    /**
     * Get the exclude hours constant.
     *
     * @return int
     */
    public static function getExcludeHours(): int
    {
        return self::EXCLUDE_HOURS;
    }

    /**
     * Get the reduced priority hours constant.
     *
     * @return int
     */
    public static function getReducedPriorityHours(): int
    {
        return self::REDUCED_PRIORITY_HOURS;
    }

    /**
     * Get the "more" weight adjustment constant.
     *
     * @return float
     */
    public static function getMoreWeightAdjustment(): float
    {
        return self::MORE_WEIGHT_ADJUSTMENT;
    }

    /**
     * Get the "less" weight adjustment constant.
     *
     * @return float
     */
    public static function getLessWeightAdjustment(): float
    {
        return self::LESS_WEIGHT_ADJUSTMENT;
    }
}
