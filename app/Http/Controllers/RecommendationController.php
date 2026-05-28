<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Models\Fragment;
use App\Models\PatternExample;
use App\Models\PreferenceModelStatus;
use App\Models\QueueTask;
use App\Models\TrackRating;
use App\Models\UserPatternExampleRating;
use App\Services\UserPreferenceScoreService;
use App\Support\SegmentDisplayLabel;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Collection;

/**
 * RecommendationController
 * 
 * Handles personalized recommendations and similar track search.
 * Requirements: 9.1, 9.2, 9.3, 9.4, 10.1, 10.2, 10.3, 10.4, 10.5
 */
class RecommendationController extends Controller
{
    public function __construct(
        private readonly UserPreferenceScoreService $preferenceScoring,
    ) {}

    /**
     * Get personalized recommendations.
     * 
     * Requirements: 9.1, 9.2, 9.3, 9.4
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function getRecommendations(Request $request): JsonResponse
    {
        $userId = (int) $request->user()->id;

        // Allow serving the last trained model while a new training run is in progress.
        $status = PreferenceModelStatus::forUser($userId)->first();
        $hasUsableModel = $status
            && !empty($status->model_path)
            && !empty($status->trained_at);

        if (!$hasUsableModel) {
            return response()->json([
                'error' => 'Model not trained',
                'message' => 'Модель ещё не обучена. Пожалуйста, обучите модель или используйте поиск похожих треков.',
                'suggest_training' => true,
            ], 400);
        }

        // Build query with filters
        // IMPORTANT: Exclude already liked tracks, rated tracks, and tracks with rated segments
        $ratedTrackIds = TrackRating::forUser($userId)
            ->pluck('track_id')
            ->unique()
            ->values()
            ->all();

        $ratedSegmentTrackIds = PatternExample::query()
            ->whereIn('id', UserPatternExampleRating::forUser($userId)
                ->pluck('pattern_example_id')
                ->unique()
                ->values()
                ->all())
            ->pluck('track_id')
            ->unique()
            ->values()
            ->all();

        $excludedTrackIds = array_values(array_unique(array_merge(
            array_map(static fn ($trackId): int => (int) $trackId, $ratedTrackIds),
            array_map(static fn ($trackId): int => (int) $trackId, $ratedSegmentTrackIds),
        )));

        $query = Track::analyzed()
            ->whereNotIn('id', $excludedTrackIds);

        // Filter by genre (Requirement 9.3)
        if ($request->has('genre')) {
            $query->byGenre($request->input('genre'));
        }

        // Filter by mood (Requirement 9.3)
        if ($request->has('mood')) {
            $query->byMood($request->input('mood'));
        }

        // Filter by BPM range (Requirement 9.3)
        if ($request->has('bpm_min') && $request->has('bpm_max')) {
            $query->bpmRange(
                (float) $request->input('bpm_min'),
                (float) $request->input('bpm_max')
            );
        }

        // Get tracks with embeddings
        $tracks = $query->whereNotNull('embedding_path')->get();

        if ($tracks->isEmpty()) {
            return response()->json([
                'recommendations' => [],
                'message' => 'No tracks available for recommendations',
            ]);
        }

        // Create recommendation task
        $limit = min((int) $request->input('limit', 20), 100);
        
        $queueTask = QueueTask::create([
            'task_type' => QueueTask::TYPE_RECOMMENDATION,
            'payload' => [
                'track_ids' => $tracks->pluck('id')->toArray(),
                'limit' => $limit,
                'filters' => [
                    'genre' => $request->input('genre'),
                    'mood' => $request->input('mood'),
                    'bpm_min' => $request->input('bpm_min'),
                    'bpm_max' => $request->input('bpm_max'),
                ],
            ],
            'status' => QueueTask::STATUS_PENDING,
        ]);

        // For now, return tracks sorted by created_at as placeholder
        // In production, this would wait for ML worker response
        
        // Try to get real scores from ML worker via Redis
        $trackIds = $tracks->pluck('id')->toArray();
        $scores = $this->getPreferenceScores(
            $userId,
            $tracks,
            $request->only(['genre', 'mood', 'bpm_min', 'bpm_max'])
        );
        
        $recommendations = $tracks
            ->map(function ($track) use ($scores) {
                $score = $scores[$track->id] ?? 0.5;
                return [
                    'track' => $track,
                    'score' => round($score, 4),
                    'reason' => 'preference_model',
                ];
            })
            ->sortByDesc('score')
            ->values();

        $recommendations = $this->diversifyRecommendations(
            $recommendations,
            $limit,
            $request->filled('genre')
        );

        // Batch load ratings for all recommended tracks
        $trackIds = $recommendations->pluck('track.id')->toArray();
        $ratings = TrackRating::forUser($userId)
            ->whereIn('track_id', $trackIds)
            ->pluck('rating', 'track_id')
            ->toArray();
        
        // Batch load segments for all recommended tracks
        $segments = PatternExample::whereIn('track_id', $trackIds)
            ->with('pattern')
            ->orderBy('start_time')
            ->get()
            ->groupBy('track_id')
            ->map(function ($examples) {
                return $examples->map(function ($example) {
                    $type = SegmentDisplayLabel::extractTypeFromPatternName($example->pattern->name ?? null);
                    return [
                        'id' => $example->id,
                        'start' => $example->start_time,
                        'end' => $example->end_time,
                        'type' => $type,
                        'display_label' => SegmentDisplayLabel::fromType($type),
                        'importance' => $example->pattern->importance ?? 1,
                        'rating' => $example->rating,
                    ];
                });
            })
            ->toArray();

        return response()->json([
            'recommendations' => $recommendations,
            'ratings' => $ratings,
            'segments' => $segments,
            'total' => $recommendations->count(),
            'model_status' => $status->status,
        ]);
    }
    
    /**
     * Get preference scores for tracks based on similarity to liked tracks.
     * 
     * Uses pre-calculated preference_score from ML model if available,
     * otherwise falls back to characteristic-based similarity.
     * 
     * @param array $trackIds
     * @return array
     */
    private function getPreferenceScores(int $userId, Collection $tracks, array $filters = []): array
    {
        return $this->preferenceScoring->scoreTracks($userId, $tracks, $filters);
    }

    private function diversifyRecommendations(
        Collection $recommendations,
        int $limit,
        bool $preserveGenreOrder = false
    ): Collection {
        $pool = $recommendations
            ->sortByDesc('score')
            ->take(max($limit * 5, 40))
            ->values();

        if ($preserveGenreOrder) {
            return $pool->take($limit)->values();
        }

        $selected = collect();
        $genreCounts = [];

        while ($selected->count() < $limit && $pool->isNotEmpty()) {
            $bestIndex = 0;
            $bestAdjustedScore = null;

            foreach ($pool as $index => $item) {
                $genreKey = strtolower(trim((string) data_get($item, 'track.genre', '')));
                $penalty = $genreKey !== ''
                    ? min(0.08, (($genreCounts[$genreKey] ?? 0) * 0.025))
                    : 0.0;
                $adjustedScore = (float) data_get($item, 'score', 0.0) - $penalty;

                if ($bestAdjustedScore === null || $adjustedScore > $bestAdjustedScore) {
                    $bestAdjustedScore = $adjustedScore;
                    $bestIndex = $index;
                }
            }

            $chosen = $pool->pull($bestIndex);
            if ($chosen === null) {
                break;
            }

            $selected->push($chosen);

            $genreKey = strtolower(trim((string) data_get($chosen, 'track.genre', '')));
            if ($genreKey !== '') {
                $genreCounts[$genreKey] = ($genreCounts[$genreKey] ?? 0) + 1;
            }
        }

        return $selected->values();
    }
    
    /**
     * Check if two genres are related (e.g., Dubstep and Drum & Bass are both bass music).
     */
    private function areGenresRelated(string $genre1, string $genre2): bool
    {
        $bassGenres = ['Dubstep', 'Drum & Bass', 'Jungle', 'Grime', 'UK Garage', 'Halftime', 'Breakbeat'];
        $houseGenres = ['House', 'Tech House', 'Deep House', 'Progressive House', 'Electro House'];
        $tranceGenres = ['Trance', 'Progressive Trance', 'Psytrance', 'Uplifting Trance'];
        
        $groups = [$bassGenres, $houseGenres, $tranceGenres];
        
        foreach ($groups as $group) {
            $inGroup1 = false;
            $inGroup2 = false;
            foreach ($group as $g) {
                if (str_contains($genre1, $g) || str_contains($g, $genre1)) $inGroup1 = true;
                if (str_contains($genre2, $g) || str_contains($g, $genre2)) $inGroup2 = true;
            }
            if ($inGroup1 && $inGroup2) return true;
        }
        
        return false;
    }

    /**
     * Find similar tracks.
     * 
     * Requirements: 10.1, 10.3, 10.4, 10.5
     *
     * @param int $trackId
     * @param Request $request
     * @return JsonResponse
     */
    public function findSimilarTracks(int $trackId, Request $request): JsonResponse
    {
        $track = Track::find($trackId);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        if (!$track->embedding_path) {
            return response()->json([
                'error' => 'Not analyzed',
                'message' => 'Track has not been analyzed yet',
            ], 400);
        }

        $k = min((int) $request->input('k', 10), 50);

        // Build query for candidate tracks
        $query = Track::analyzed()
            ->whereNotNull('embedding_path')
            ->where('id', '!=', $trackId);

        // Apply filters (Requirement 10.5)
        if ($request->has('genre')) {
            $query->byGenre($request->input('genre'));
        }

        if ($request->has('mood')) {
            $query->byMood($request->input('mood'));
        }

        if ($request->has('bpm_min') && $request->has('bpm_max')) {
            $query->bpmRange(
                (float) $request->input('bpm_min'),
                (float) $request->input('bpm_max')
            );
        }

        $candidates = $query->get();

        if ($candidates->isEmpty()) {
            return response()->json([
                'similar_tracks' => [],
                'query_track' => $track,
            ]);
        }

        // Return placeholder results (in production, would poll for results)
        // Calculate similarity based on track characteristics
        $similarTracks = $candidates
            ->map(function ($candidate) use ($track) {
                $similarity = $this->calculateTrackSimilarity($track, $candidate);
                return [
                    'track_id' => $candidate->id,
                    'track' => $candidate,
                    'similarity' => round($similarity, 2),
                ];
            })
            ->sortByDesc('similarity')
            ->take($k)
            ->values();

        return response()->json([
            'similar_tracks' => $similarTracks,
            'query_track' => $track,
            'task_id' => null,
        ]);
    }
    
    /**
     * Calculate similarity between two tracks based on characteristics.
     * 
     * @param Track $track1
     * @param Track $track2
     * @return float
     */
    private function calculateTrackSimilarity(Track $track1, Track $track2): float
    {
        $score = 0.0;
        $maxScore = 0.0;
        
        // Genre similarity (weight: 30%)
        if ($track1->genre && $track2->genre) {
            $maxScore += 0.30;
            if ($track1->genre === $track2->genre) {
                $score += 0.30;
            } elseif (str_contains(strtolower($track1->genre), strtolower($track2->genre)) || 
                      str_contains(strtolower($track2->genre), strtolower($track1->genre))) {
                $score += 0.20;
            }
        }
        
        // BPM similarity (weight: 25%)
        if ($track1->bpm && $track2->bpm) {
            $maxScore += 0.25;
            $bpmDiff = abs($track1->bpm - $track2->bpm);
            if ($bpmDiff < 3) {
                $score += 0.25;
            } elseif ($bpmDiff < 8) {
                $score += 0.20;
            } elseif ($bpmDiff < 15) {
                $score += 0.15;
            } elseif ($bpmDiff < 25) {
                $score += 0.10;
            } elseif ($bpmDiff < 40) {
                $score += 0.05;
            }
        }
        
        // Key similarity (weight: 20%)
        if ($track1->key && $track2->key) {
            $maxScore += 0.20;
            if ($track1->key === $track2->key) {
                $score += 0.20;
            } elseif ($this->areKeysCompatible($track1->key, $track2->key)) {
                $score += 0.12;
            }
        }
        
        // Mood similarity (weight: 25%)
        if ($track1->mood && $track2->mood) {
            $maxScore += 0.25;
            if ($track1->mood === $track2->mood) {
                $score += 0.25;
            } elseif ($this->areMoodsSimilar($track1->mood, $track2->mood)) {
                $score += 0.15;
            }
        }
        
        // Calculate final similarity
        if ($maxScore > 0) {
            $similarity = $score / $maxScore;
        } else {
            $similarity = 0.5; // Default when no data
        }
        
        // Add some randomness to avoid all being the same
        $similarity += (mt_rand(-5, 5) / 100);
        
        return min(0.95, max(0.10, round($similarity, 2)));
    }
    
    /**
     * Check if two musical keys are compatible (Camelot wheel).
     */
    private function areKeysCompatible(string $key1, string $key2): bool
    {
        // Simple compatibility check - same letter or adjacent numbers
        $k1 = strtoupper(trim($key1));
        $k2 = strtoupper(trim($key2));
        
        // Extract number and letter from Camelot notation (e.g., "8A", "11B")
        preg_match('/(\d+)([AB])/i', $k1, $m1);
        preg_match('/(\d+)([AB])/i', $k2, $m2);
        
        if (empty($m1) || empty($m2)) return false;
        
        $num1 = (int)$m1[1];
        $num2 = (int)$m2[1];
        $letter1 = strtoupper($m1[2]);
        $letter2 = strtoupper($m2[2]);
        
        // Same key
        if ($num1 === $num2 && $letter1 === $letter2) return true;
        
        // Adjacent numbers, same letter
        if ($letter1 === $letter2 && abs($num1 - $num2) === 1) return true;
        
        // Same number, different letter (relative major/minor)
        if ($num1 === $num2 && $letter1 !== $letter2) return true;
        
        return false;
    }
    
    /**
     * Check if two moods are similar.
     */
    private function areMoodsSimilar(string $mood1, string $mood2): bool
    {
        $moodGroups = [
            'energetic' => ['energetic', 'aggressive', 'powerful', 'intense', 'driving'],
            'happy' => ['happy', 'uplifting', 'euphoric', 'joyful', 'bright'],
            'dark' => ['dark', 'melancholic', 'sad', 'moody', 'atmospheric'],
            'chill' => ['chill', 'relaxed', 'calm', 'peaceful', 'ambient'],
        ];
        
        $m1 = strtolower($mood1);
        $m2 = strtolower($mood2);
        
        foreach ($moodGroups as $group) {
            $in1 = false;
            $in2 = false;
            foreach ($group as $mood) {
                if (str_contains($m1, $mood)) $in1 = true;
                if (str_contains($m2, $mood)) $in2 = true;
            }
            if ($in1 && $in2) return true;
        }
        
        return false;
    }

    /**
     * Find similar fragments.
     * 
     * Requirements: 10.2
     *
     * @param int $fragmentId
     * @param Request $request
     * @return JsonResponse
     */
    public function findSimilarFragments(int $fragmentId, Request $request): JsonResponse
    {
        $fragment = Fragment::with('track')->find($fragmentId);

        if (!$fragment) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        }

        if (!$fragment->embedding_path) {
            return response()->json([
                'error' => 'Not analyzed',
                'message' => 'Fragment has not been analyzed yet',
            ], 400);
        }

        $k = min((int) $request->input('k', 10), 50);

        // Get candidate fragments
        $candidates = Fragment::analyzed()
            ->whereNotNull('embedding_path')
            ->where('id', '!=', $fragmentId)
            ->with('track')
            ->get();

        if ($candidates->isEmpty()) {
            return response()->json([
                'similar_fragments' => [],
                'query_fragment' => $fragment,
            ]);
        }

        // Return placeholder results
        $similarFragments = $candidates
            ->map(function ($candidate) use ($fragment) {
                // Calculate similarity based on parent track characteristics
                $similarity = 0.5;
                if ($fragment->track && $candidate->track) {
                    if ($fragment->track->genre === $candidate->track->genre) {
                        $similarity += 0.2;
                    }
                    if ($fragment->track->mood === $candidate->track->mood) {
                        $similarity += 0.1;
                    }
                }
                // Time range similarity
                $fragDuration = $fragment->end_time - $fragment->start_time;
                $candDuration = $candidate->end_time - $candidate->start_time;
                if (abs($fragDuration - $candDuration) < 5) {
                    $similarity += 0.1;
                }
                return [
                    'fragment_id' => $candidate->id,
                    'fragment' => $candidate,
                    'similarity' => round(min(0.99, $similarity), 2),
                ];
            })
            ->sortByDesc('similarity')
            ->take($k)
            ->values();

        return response()->json([
            'similar_fragments' => $similarFragments,
            'query_fragment' => $fragment,
        ]);
    }
}
