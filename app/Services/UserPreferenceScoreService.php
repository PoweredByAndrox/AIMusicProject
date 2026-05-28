<?php

namespace App\Services;

use App\Models\ListeningHistory;
use App\Models\Track;
use App\Models\TrackRating;
use App\Models\UserTrackLike;
use App\Models\UserTrackScore;
use Illuminate\Support\Collection;

class UserPreferenceScoreService
{
    /**
     * @var array<int, array<string, mixed>>
     */
    private array $profileCache = [];

    /**
     * @param iterable<int, Track> $tracks
     * @param array<string, mixed> $filters
     * @return array<int, float>
     */
    public function scoreTracks(
        int $userId,
        iterable $tracks,
        array $filters = [],
        bool $allowNeutralFallback = true
    ): array {
        $trackCollection = $tracks instanceof Collection
            ? $tracks->values()
            : collect($tracks)->filter(fn ($track) => $track instanceof Track)->values();

        if ($trackCollection->isEmpty()) {
            return [];
        }

        $profile = $this->buildProfile($userId);
        $trackIds = $trackCollection
            ->pluck('id')
            ->map(static fn ($trackId): int => (int) $trackId)
            ->values()
            ->all();

        $storedScores = UserTrackScore::forUser($userId)
            ->whereIn('track_id', $trackIds)
            ->pluck('preference_score', 'track_id')
            ->map(static fn ($score): float => (float) $score)
            ->toArray();

        $scores = [];

        foreach ($trackCollection as $track) {
            $trackId = (int) $track->id;
            $storedScore = array_key_exists($trackId, $storedScores)
                ? (float) $storedScores[$trackId]
                : null;

            if (!$profile['has_explicit_signals']) {
                if ($storedScore !== null) {
                    $scores[$trackId] = round(min(0.99, max(0.05, $storedScore)), 4);
                } elseif ($allowNeutralFallback) {
                    $scores[$trackId] = 0.5;
                }

                continue;
            }

            $heuristicScore = $this->computeHeuristicScore($track, $profile, $filters);
            $score = $heuristicScore;

            if ($storedScore !== null) {
                $positiveTrackCount = (int) $profile['positive_track_count'];
                if ($positiveTrackCount >= 12) {
                    $score = ($heuristicScore * 0.7) + ($storedScore * 0.3);
                } elseif ($positiveTrackCount >= 4) {
                    $score = ($heuristicScore * 0.55) + ($storedScore * 0.45);
                } else {
                    $score = ($heuristicScore * 0.4) + ($storedScore * 0.6);
                }
            }

            $scores[$trackId] = round(min(0.99, max(0.05, $score)), 4);
        }

        return $scores;
    }

    /**
     * @param array<string, mixed> $filters
     */
    public function scoreTrack(
        int $userId,
        Track $track,
        array $filters = [],
        bool $allowNeutralFallback = false
    ): ?float {
        $scores = $this->scoreTracks($userId, [$track], $filters, $allowNeutralFallback);

        return $scores[(int) $track->id] ?? null;
    }

    /**
     * @return array<string, mixed>
     */
    private function buildProfile(int $userId): array
    {
        if (isset($this->profileCache[$userId])) {
            return $this->profileCache[$userId];
        }

        $trackWeights = [];

        foreach (UserTrackLike::forUser($userId)->pluck('track_id')->all() as $trackId) {
            $trackWeights[(int) $trackId] = max($trackWeights[(int) $trackId] ?? 0.0, 1.7);
        }

        $highRatings = TrackRating::forUser($userId)
            ->where('rating', '>=', 4)
            ->get(['track_id', 'rating']);

        foreach ($highRatings as $rating) {
            $trackId = (int) $rating->track_id;
            $weight = ((int) $rating->rating) >= 5 ? 1.5 : 1.1;
            $trackWeights[$trackId] = max($trackWeights[$trackId] ?? 0.0, $weight);
        }

        $playCounts = ListeningHistory::forUser($userId)
            ->select('track_id')
            ->selectRaw('COUNT(*) as play_count')
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= 2')
            ->pluck('play_count', 'track_id')
            ->toArray();

        foreach ($playCounts as $trackId => $playCount) {
            $existingWeight = $trackWeights[(int) $trackId] ?? 0.0;
            $bonus = min(0.6, max(0.0, (((int) $playCount) - 1) * 0.12));
            $trackWeights[(int) $trackId] = max($existingWeight, 0.8 + $bonus);
        }

        $positiveTracks = empty($trackWeights)
            ? collect()
            : Track::query()
                ->whereIn('id', array_keys($trackWeights))
                ->where('analysis_status', 'completed')
                ->get(['id', 'genre', 'mood', 'bpm', 'key']);

        $genreWeights = [];
        $moodWeights = [];
        $keyWeights = [];
        $weightedBpmSum = 0.0;
        $weightedBpmWeight = 0.0;

        foreach ($positiveTracks as $track) {
            $weight = (float) ($trackWeights[(int) $track->id] ?? 0.0);
            if ($weight <= 0) {
                continue;
            }

            $genre = $this->normalizeToken($track->genre);
            if ($genre !== '') {
                $genreWeights[$genre] = ($genreWeights[$genre] ?? 0.0) + $weight;
            }

            $mood = $this->normalizeToken($track->mood);
            if ($mood !== '') {
                $moodWeights[$mood] = ($moodWeights[$mood] ?? 0.0) + $weight;
            }

            $key = $this->normalizeToken($track->key);
            if ($key !== '') {
                $keyWeights[$key] = ($keyWeights[$key] ?? 0.0) + $weight;
            }

            if ($track->bpm !== null && $track->bpm > 0) {
                $weightedBpmSum += ((float) $track->bpm) * $weight;
                $weightedBpmWeight += $weight;
            }
        }

        $profile = [
            'has_explicit_signals' => $positiveTracks->isNotEmpty(),
            'positive_track_count' => $positiveTracks->count(),
            'genre_weights' => $genreWeights,
            'mood_weights' => $moodWeights,
            'key_weights' => $keyWeights,
            'max_genre_weight' => empty($genreWeights) ? 0.0 : max($genreWeights),
            'max_mood_weight' => empty($moodWeights) ? 0.0 : max($moodWeights),
            'max_key_weight' => empty($keyWeights) ? 0.0 : max($keyWeights),
            'avg_bpm' => $weightedBpmWeight > 0 ? $weightedBpmSum / $weightedBpmWeight : null,
        ];

        $this->profileCache[$userId] = $profile;

        return $profile;
    }

    /**
     * @param array<string, mixed> $profile
     * @param array<string, mixed> $filters
     */
    private function computeHeuristicScore(Track $track, array $profile, array $filters): float
    {
        $score = 0.22;
        $trackGenre = $this->normalizeToken($track->genre);

        if ($trackGenre !== '' && (float) $profile['max_genre_weight'] > 0.0) {
            $exactGenreWeight = (float) ($profile['genre_weights'][$trackGenre] ?? 0.0);
            if ($exactGenreWeight > 0.0) {
                $score += 0.45 * ($exactGenreWeight / (float) $profile['max_genre_weight']);
            } else {
                $bestRelatedWeight = 0.0;
                foreach ($profile['genre_weights'] as $preferredGenre => $weight) {
                    if ($this->areGenresRelated($trackGenre, (string) $preferredGenre)) {
                        $bestRelatedWeight = max($bestRelatedWeight, (float) $weight);
                    }
                }

                if ($bestRelatedWeight > 0.0) {
                    $score += 0.22 * ($bestRelatedWeight / (float) $profile['max_genre_weight']);
                }
            }
        }

        $trackMood = $this->normalizeToken($track->mood);
        if ($trackMood !== '' && (float) $profile['max_mood_weight'] > 0.0) {
            $moodWeight = (float) ($profile['mood_weights'][$trackMood] ?? 0.0);
            if ($moodWeight > 0.0) {
                $score += 0.10 * ($moodWeight / (float) $profile['max_mood_weight']);
            }
        }

        $trackKey = $this->normalizeToken($track->key);
        if ($trackKey !== '' && (float) $profile['max_key_weight'] > 0.0) {
            $keyWeight = (float) ($profile['key_weights'][$trackKey] ?? 0.0);
            if ($keyWeight > 0.0) {
                $score += 0.06 * ($keyWeight / (float) $profile['max_key_weight']);
            }
        }

        $avgBpm = $profile['avg_bpm'];
        if ($track->bpm !== null && $track->bpm > 0 && is_numeric($avgBpm)) {
            $bpmDiff = abs((float) $track->bpm - (float) $avgBpm);
            if ($bpmDiff < 8) {
                $score += 0.15;
            } elseif ($bpmDiff < 16) {
                $score += 0.11;
            } elseif ($bpmDiff < 28) {
                $score += 0.07;
            } elseif ($bpmDiff < 40) {
                $score += 0.03;
            }
        }

        $score += $this->filterContextBoost($track, $filters);
        $score += ((crc32((string) $track->id . date('Y-m-d')) % 15) / 1000);

        return min(0.99, $score);
    }

    /**
     * @param array<string, mixed> $filters
     */
    private function filterContextBoost(Track $track, array $filters): float
    {
        $boost = 0.0;
        $trackGenre = $this->normalizeToken($track->genre);
        $trackMood = $this->normalizeToken($track->mood);

        $filterGenre = $this->normalizeToken($filters['genre'] ?? null);
        if ($filterGenre !== '') {
            if ($trackGenre === $filterGenre) {
                $boost += 0.18;
            } elseif ($trackGenre !== '' && $this->areGenresRelated($trackGenre, $filterGenre)) {
                $boost += 0.08;
            }
        }

        $filterMood = $this->normalizeToken($filters['mood'] ?? null);
        if ($filterMood !== '' && $trackMood === $filterMood) {
            $boost += 0.08;
        }

        $bpmMin = isset($filters['bpm_min']) && is_numeric($filters['bpm_min'])
            ? (float) $filters['bpm_min']
            : null;
        $bpmMax = isset($filters['bpm_max']) && is_numeric($filters['bpm_max'])
            ? (float) $filters['bpm_max']
            : null;

        if ($track->bpm !== null && $track->bpm > 0 && $bpmMin !== null && $bpmMax !== null) {
            $trackBpm = (float) $track->bpm;
            if ($trackBpm >= $bpmMin && $trackBpm <= $bpmMax) {
                $boost += 0.05;
            }
        }

        return $boost;
    }

    private function normalizeToken(string|null $value): string
    {
        return mb_strtolower(trim((string) $value));
    }

    private function areGenresRelated(string $genre1, string $genre2): bool
    {
        $groups = [
            ['dubstep', 'drum & bass', 'jungle', 'grime', 'uk garage', 'halftime', 'breakbeat'],
            ['house', 'tech house', 'deep house', 'progressive house', 'electro house'],
            ['trance', 'progressive trance', 'psytrance', 'uplifting trance', 'tech trance'],
            ['hardstyle', 'hands up', 'jumpstyle', 'hardcore', 'happy hardcore'],
        ];

        foreach ($groups as $group) {
            $inGroup1 = false;
            $inGroup2 = false;

            foreach ($group as $groupGenre) {
                if (str_contains($genre1, $groupGenre) || str_contains($groupGenre, $genre1)) {
                    $inGroup1 = true;
                }
                if (str_contains($genre2, $groupGenre) || str_contains($groupGenre, $genre2)) {
                    $inGroup2 = true;
                }
            }

            if ($inGroup1 && $inGroup2) {
                return true;
            }
        }

        return false;
    }
}
