<?php

namespace App\Services;

use App\Models\Fragment;
use App\Models\PatternExample;
use App\Models\Track;
use App\Models\TrackRating;
use App\Models\UserPatternExampleRating;
use App\Support\SegmentDisplayLabel;

class TrackFeedbackSnapshotService
{
    public function __construct(
        private readonly TrackRatingPropagationService $ratingPropagation,
        private readonly UserPreferenceScoreService $preferenceScoring,
    ) {}

    /**
     * @return array{
     *   track_id:int,
     *   fragments: array<int, array<string, mixed>>,
     *   segments: array<int, array<string, mixed>>,
     *   track_rating: float|int|null,
     *   explicit_track_rating: int|null,
     *   computed_track_rating: float|null,
     *   preference_score: float|null,
     *   sentiment: string|null
     * }
     */
    public function buildForUser(int $userId, Track $track): array
    {
        $examples = PatternExample::query()
            ->where('track_id', $track->id)
            ->with('pattern')
            ->orderBy('start_time')
            ->get();

        $ratingsByExample = UserPatternExampleRating::forUser($userId)
            ->whereIn('pattern_example_id', $examples->pluck('id')->all())
            ->get()
            ->keyBy('pattern_example_id');

        $fragments = Fragment::forUser($userId)
            ->where('track_id', $track->id)
            ->orderBy('start_time')
            ->get()
            ->map(static fn (Fragment $fragment): array => [
                'id' => (int) $fragment->id,
                'track_id' => (int) $fragment->track_id,
                'start_time' => (float) $fragment->start_time,
                'end_time' => (float) $fragment->end_time,
                'is_liked' => $fragment->is_liked,
                'name' => null,
                'comment' => $fragment->comment,
            ])
            ->values()
            ->all();

        $segments = $examples->map(function (PatternExample $example) use ($ratingsByExample): array {
            $type = SegmentDisplayLabel::extractTypeFromPatternName($example->pattern->name ?? null);
            $userRating = $ratingsByExample->get($example->id);

            return [
                'id' => $example->id,
                'start' => $example->start_time,
                'end' => $example->end_time,
                'type' => $type,
                'display_label' => SegmentDisplayLabel::fromType($type),
                'importance' => $example->pattern->importance ?? 1,
                'is_liked' => $userRating?->is_positive,
                'rating' => $userRating?->rating,
                'pattern_id' => $example->pattern_id,
                'pattern_name' => $example->pattern->name ?? null,
            ];
        })->values()->all();

        $explicitTrackRating = TrackRating::forUser($userId)
            ->where('track_id', $track->id)
            ->value('rating');

        $computedTrackRating = $this->ratingPropagation->computeWeightedSegmentRating($userId, (int) $track->id);
        $effectiveTrackRating = $explicitTrackRating !== null
            ? (int) $explicitTrackRating
            : ($computedTrackRating !== null ? (int) max(1, min(5, round($computedTrackRating))) : null);

        return [
            'track_id' => (int) $track->id,
            'fragments' => $fragments,
            'segments' => $segments,
            'track_rating' => $effectiveTrackRating,
            'explicit_track_rating' => $explicitTrackRating !== null ? (int) $explicitTrackRating : null,
            'computed_track_rating' => $computedTrackRating,
            'preference_score' => $this->resolvePreferenceScore($userId, $track),
            'sentiment' => $this->ratingPropagation->sentimentForRating(
                $effectiveTrackRating !== null ? (float) $effectiveTrackRating : $computedTrackRating
            ),
        ];
    }

    private function resolvePreferenceScore(int $userId, Track $track): ?float
    {
        return $this->preferenceScoring->scoreTrack($userId, $track);
    }
}
