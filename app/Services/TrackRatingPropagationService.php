<?php

namespace App\Services;

use App\Models\PatternExample;
use App\Models\RatingHistory;
use App\Models\TrackRating;
use App\Models\UserPatternExampleRating;
use Illuminate\Support\Facades\DB;

class TrackRatingPropagationService
{
    public function syncSegmentsFromTrackRating(int $userId, int $trackId, int $rating): void
    {
        $examples = PatternExample::query()
            ->where('track_id', $trackId)
            ->pluck('id');

        if ($examples->isEmpty()) {
            return;
        }

        foreach ($examples as $exampleId) {
            UserPatternExampleRating::updateOrCreate(
                [
                    'user_id' => $userId,
                    'pattern_example_id' => (int) $exampleId,
                ],
                [
                    'rating' => (float) $rating,
                    'is_positive' => $rating >= 3,
                ],
            );
        }
    }

    public function syncTrackFromSegments(int $userId, int $trackId, ?float $computed = null): ?TrackRating
    {
        $computed ??= $this->computeWeightedSegmentRating($userId, $trackId);

        return DB::transaction(function () use ($userId, $trackId, $computed): ?TrackRating {
            $existing = TrackRating::forUser($userId)
                ->where('track_id', $trackId)
                ->first();

            if ($computed === null) {
                return $existing;
            }

            $rounded = (int) max(1, min(5, round($computed)));
            $oldRating = $existing?->rating;

            if ($existing) {
                if ($oldRating !== $rounded) {
                    RatingHistory::create([
                        'user_id' => $userId,
                        'track_id' => $trackId,
                        'old_rating' => $oldRating,
                        'new_rating' => $rounded,
                        'changed_at' => now(),
                    ]);
                    $existing->update(['rating' => $rounded]);
                }

                return $existing->fresh();
            }

            return TrackRating::create([
                'user_id' => $userId,
                'track_id' => $trackId,
                'rating' => $rounded,
            ]);
        });
    }

    public function computeWeightedSegmentRating(int $userId, int $trackId): ?float
    {
        $examples = PatternExample::query()
            ->where('track_id', $trackId)
            ->with('pattern:id,importance')
            ->get(['id', 'pattern_id']);

        if ($examples->isEmpty()) {
            return null;
        }

        $ratings = UserPatternExampleRating::forUser($userId)
            ->whereIn('pattern_example_id', $examples->pluck('id')->all())
            ->get()
            ->keyBy('pattern_example_id');

        $weightedSum = 0.0;
        $totalWeight = 0.0;

        foreach ($examples as $example) {
            $rating = $ratings->get((int) $example->id);
            if ($rating === null || $rating->rating === null) {
                continue;
            }

            $weight = max(1, (int) ($example->pattern->importance ?? 1));
            $weightedSum += (float) $rating->rating * $weight;
            $totalWeight += $weight;
        }

        if ($totalWeight <= 0) {
            return null;
        }

        return round($weightedSum / $totalWeight, 1);
    }

    public function resolveEffectiveTrackRating(int $userId, int $trackId): ?int
    {
        $explicit = TrackRating::forUser($userId)
            ->where('track_id', $trackId)
            ->value('rating');

        if ($explicit !== null) {
            return (int) $explicit;
        }

        $computed = $this->computeWeightedSegmentRating($userId, $trackId);
        if ($computed === null) {
            return null;
        }

        return (int) max(1, min(5, round($computed)));
    }

    public function sentimentForRating(?float $rating): ?string
    {
        if ($rating === null) {
            return null;
        }

        if ($rating <= 1.5) {
            return 'dislike';
        }

        if ($rating >= 4.5) {
            return 'love';
        }

        return 'rated';
    }
}
