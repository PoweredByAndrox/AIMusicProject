<?php

namespace App\Services;

use App\Models\RatingHistory;
use App\Models\Track;
use App\Models\TrackRating;
use App\Models\UserTrackLike;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Support\Facades\DB;
use InvalidArgumentException;

/**
 * TrackRatingService
 * 
 * Handles 5-star rating operations for tracks with automatic history recording.
 * Requirements: 11.1, 11.2, 11.3, 14.1
 */
class TrackRatingService
{
    public function __construct(
        private readonly TrackRatingPropagationService $ratingPropagation,
    ) {}

    /**
     * Minimum allowed rating value.
     */
    private const MIN_RATING = 1;

    /**
     * Maximum allowed rating value.
     */
    private const MAX_RATING = 5;

    /**
     * Set or update the rating for a track.
     * 
     * Requirement 11.1: WHEN пользователь устанавливает рейтинг трека THEN THE 
     * Track_Rating SHALL сохранить значение от 1 до 5 звёзд
     * 
     * Requirement 11.2: WHEN пользователь изменяет рейтинг THEN THE Track_Rating 
     * SHALL обновить значение и записать в Rating_History
     * 
     * Requirement 14.1: WHEN пользователь изменяет рейтинг THEN THE Rating_History 
     * SHALL записать: track_id, old_rating, new_rating, timestamp
     *
     * @param int $trackId The ID of the track to rate
     * @param int $rating The rating value (1-5)
     * @return TrackRating The created or updated rating
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     * @throws InvalidArgumentException If rating is not between 1 and 5
     */
    public function setRating(int $userId, int $trackId, int $rating): TrackRating
    {
        // Validate rating range
        if ($rating < self::MIN_RATING || $rating > self::MAX_RATING) {
            throw new InvalidArgumentException(
                sprintf('Rating must be between %d and %d', self::MIN_RATING, self::MAX_RATING)
            );
        }

        // Verify track exists
        $track = Track::findOrFail($trackId);

        $result = DB::transaction(function () use ($userId, $track, $rating): array {
            // Get existing rating if any
            $existingRating = TrackRating::forUser($userId)
                ->where('track_id', $track->id)
                ->first();
            $oldRating = $existingRating?->rating;
            $changed = $existingRating === null || $oldRating !== $rating;

            if ($existingRating) {
                // Only record history and update if rating actually changed
                if ($oldRating !== $rating) {
                    // Record history entry
                    RatingHistory::create([
                        'user_id' => $userId,
                        'track_id' => $track->id,
                        'old_rating' => $oldRating,
                        'new_rating' => $rating,
                        'changed_at' => now(),
                    ]);

                    // Update existing rating
                    $existingRating->update(['rating' => $rating]);
                }

                $freshRating = $existingRating->fresh();
                $this->syncLikeProjection($userId, $track->id, $freshRating?->rating);

                return [
                    'rating' => $freshRating,
                    'changed' => $changed,
                ];
            }

            // Create new rating (no history for first rating)
            $createdRating = TrackRating::create([
                'user_id' => $userId,
                'track_id' => $track->id,
                'rating' => $rating,
            ]);

            $this->syncLikeProjection($userId, $track->id, $createdRating->rating);

            return [
                'rating' => $createdRating,
                'changed' => $changed,
            ];
        });
        
        // NOTE: Auto-training disabled — user triggers training manually via UI button
        // PreferenceTrainingService::triggerTraining();
        
        // Keep explicit whole-track ratings independent from per-segment feedback.
        // Segment ratings must remain user-owned granular signals and must not be
        // overwritten when the user likes or rates the entire track.

        if (($result['changed'] ?? false) === true) {
            PreferenceTrainingService::triggerTraining($userId);
        }

        return $result['rating'];
    }

    /**
     * Remove the rating for a track.
     * 
     * Requirement 11.3: WHEN пользователь удаляет рейтинг THEN THE Track_Rating 
     * SHALL установить значение в NULL
     *
     * @param int $trackId The ID of the track
     * @return void
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function removeRating(int $userId, int $trackId): void
    {
        // Verify track exists
        Track::findOrFail($trackId);

        $existingRating = TrackRating::forUser($userId)
            ->where('track_id', $trackId)
            ->first();

        if ($existingRating) {
            DB::transaction(function () use ($userId, $trackId, $existingRating) {
                $oldRating = $existingRating->rating;

                // Record history entry for removal
                if ($oldRating !== null) {
                    RatingHistory::create([
                        'user_id' => $userId,
                        'track_id' => $trackId,
                        'old_rating' => $oldRating,
                        'new_rating' => null,
                        'changed_at' => now(),
                    ]);
                }

                // Delete the rating record
                $existingRating->delete();
                $this->syncLikeProjection($userId, $trackId, null);
            });

            PreferenceTrainingService::triggerTraining($userId);
        }
    }

    private function syncLikeProjection(int $userId, int $trackId, ?int $rating): void
    {
        if ($rating === self::MAX_RATING) {
            UserTrackLike::updateOrCreate([
                'user_id' => $userId,
                'track_id' => $trackId,
            ], []);
            return;
        }

        UserTrackLike::forUser($userId)
            ->where('track_id', $trackId)
            ->delete();
    }

    /**
     * Get the current rating for a track.
     *
     * @param int $trackId The ID of the track
     * @return int|null The rating value (1-5) or null if not rated
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getRating(int $userId, int $trackId): ?int
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return $this->ratingPropagation->resolveEffectiveTrackRating($userId, $trackId);
    }

    public function getExplicitRating(int $userId, int $trackId): ?int
    {
        // Verify track exists
        Track::findOrFail($trackId);

        $rating = TrackRating::forUser($userId)
            ->where('track_id', $trackId)
            ->first();

        return $rating?->rating;
    }

    /**
     * Get the rating history for a track.
     * 
     * Requirement 14.1: THE Rating_History SHALL хранить все изменения рейтингов 
     * без ограничения по времени
     *
     * @param int $trackId The ID of the track
     * @return Collection<int, RatingHistory> Collection of rating history entries
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getHistory(int $userId, int $trackId): Collection
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return RatingHistory::forUser($userId)
            ->where('track_id', $trackId)
            ->orderBy('changed_at', 'desc')
            ->get();
    }

    /**
     * Get all tracks with a rating at or above the specified minimum.
     * 
     * Requirement 11.5: THE System SHALL позволять фильтровать треки по рейтингу 
     * (например, "4 звезды и выше")
     *
     * @param int $minRating The minimum rating (1-5)
     * @return Collection<int, Track> Collection of tracks with rating >= minRating
     * @throws InvalidArgumentException If minRating is not between 1 and 5
     */
    public function getTracksWithRating(int $userId, int $minRating): Collection
    {
        // Validate minRating range
        if ($minRating < self::MIN_RATING || $minRating > self::MAX_RATING) {
            throw new InvalidArgumentException(
                sprintf('Minimum rating must be between %d and %d', self::MIN_RATING, self::MAX_RATING)
            );
        }

        return Track::whereHas('rating', function ($query) use ($minRating) {
            $query->where('rating', '>=', $minRating);
        })->with('rating')->get();
    }
}
