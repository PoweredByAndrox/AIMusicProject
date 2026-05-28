<?php

namespace App\Http\Controllers;

use App\Models\Pattern;
use App\Models\PatternExample;
use App\Models\PatternMatch;
use App\Models\Track;
use App\Models\TrackRating;
use App\Models\UserPatternExampleRating;
use App\Models\UserTrackScore;
use App\Support\SegmentDisplayLabel;
use App\Services\MLQueueService;
use App\Services\TrackFeedbackSnapshotService;
use App\Services\TrackRatingPropagationService;
use Illuminate\Contracts\Cache\LockTimeoutException;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\Cache;

/**
 * PatternController - Manage melodic patterns for search and personalization.
 */
class PatternController extends Controller
{
    public function __construct(
        private MLQueueService $mlQueue,
        private TrackFeedbackSnapshotService $trackFeedbackSnapshots,
        private TrackRatingPropagationService $ratingPropagation,
    ) {}

    /**
     * List all patterns.
     */
    public function index(): JsonResponse
    {
        $patterns = Pattern::withCount(['examples', 'matches'])
            ->orderBy('created_at', 'desc')
            ->get();

        return response()->json($patterns);
    }

    /**
     * Create a new pattern.
     */
    public function store(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'name' => 'required|string|max:255',
            'description' => 'nullable|string',
            'representation_type' => 'in:pitch_contour,chroma,midi',
            'transpose_invariant' => 'boolean',
            'tempo_invariant' => 'boolean',
            'match_threshold' => 'numeric|min:0|max:1',
            'importance' => 'integer|min:0|max:3',
        ]);

        $pattern = Pattern::create([
            'name' => $validated['name'],
            'description' => $validated['description'] ?? null,
            'representation_type' => $validated['representation_type'] ?? 'chroma',
            'transpose_invariant' => $validated['transpose_invariant'] ?? true,
            'tempo_invariant' => $validated['tempo_invariant'] ?? true,
            'match_threshold' => $validated['match_threshold'] ?? 0.7,
            'importance' => $validated['importance'] ?? 1,
        ]);

        return response()->json($pattern, 201);
    }

    /**
     * Get a single pattern with examples.
     */
    public function show(Pattern $pattern): JsonResponse
    {
        $pattern->load(['examples.track', 'matches.track']);
        
        return response()->json($pattern);
    }

    /**
     * Update a pattern.
     */
    public function update(Request $request, Pattern $pattern): JsonResponse
    {
        $validated = $request->validate([
            'name' => 'string|max:255',
            'description' => 'nullable|string',
            'representation_type' => 'in:pitch_contour,chroma,midi',
            'transpose_invariant' => 'boolean',
            'tempo_invariant' => 'boolean',
            'match_threshold' => 'numeric|min:0|max:1',
            'importance' => 'integer|min:0|max:3',
        ]);

        $pattern->update($validated);

        return response()->json($pattern);
    }

    /**
     * Delete a pattern.
     */
    public function destroy(Pattern $pattern): JsonResponse
    {
        $pattern->delete();

        return response()->json(['message' => 'Pattern deleted']);
    }

    /**
     * Add an example to a pattern.
     */
    public function addExample(Request $request, Pattern $pattern): JsonResponse
    {
        $validated = $request->validate([
            'track_id' => 'required|exists:tracks,id',
            'start_time' => 'required|numeric|min:0',
            'end_time' => 'required|numeric|gt:start_time',
            'is_positive' => 'boolean',
            'notes' => 'nullable|string',
        ]);

        $example = $pattern->examples()->create([
            'track_id' => $validated['track_id'],
            'start_time' => $validated['start_time'],
            'end_time' => $validated['end_time'],
            'is_positive' => $validated['is_positive'] ?? true,
            'notes' => $validated['notes'] ?? null,
        ]);

        $example->load('track');

        return response()->json($example, 201);
    }

    /**
     * Remove an example from a pattern.
     */
    public function removeExample(Pattern $pattern, PatternExample $example): JsonResponse
    {
        if ($example->pattern_id !== $pattern->id) {
            return response()->json(['error' => 'Example does not belong to this pattern'], 400);
        }

        $example->delete();

        return response()->json(['message' => 'Example removed']);
    }

    /**
     * Search for pattern matches across library.
     */
    public function search(Pattern $pattern): JsonResponse
    {
        // Check if pattern has examples
        if ($pattern->positiveExamples()->count() === 0) {
            return response()->json([
                'error' => 'Pattern needs at least one positive example'
            ], 400);
        }

        // Queue search task
        $taskId = $this->mlQueue->pushTask('search_pattern', [
            'pattern_id' => $pattern->id,
            'limit' => 50,
        ]);

        return response()->json([
            'message' => 'Search started',
            'task_id' => $taskId,
        ]);
    }

    /**
     * Get pattern matches.
     */
    public function matches(Pattern $pattern): JsonResponse
    {
        $matches = $pattern->matches()
            ->with('track')
            ->orderBy('score', 'desc')
            ->get();

        return response()->json($matches);
    }

    /**
     * Confirm or reject a match.
     */
    public function confirmMatch(Request $request, Pattern $pattern, PatternMatch $match): JsonResponse
    {
        if ($match->pattern_id !== $pattern->id) {
            return response()->json(['error' => 'Match does not belong to this pattern'], 400);
        }

        $validated = $request->validate([
            'confirmed' => 'required|boolean',
        ]);

        $match->update(['confirmed' => $validated['confirmed']]);

        // If confirmed, optionally add as example
        if ($validated['confirmed']) {
            // Check if example already exists
            $exists = $pattern->examples()
                ->where('track_id', $match->track_id)
                ->where('start_time', $match->start_time)
                ->exists();

            if (!$exists) {
                $pattern->examples()->create([
                    'track_id' => $match->track_id,
                    'start_time' => $match->start_time,
                    'end_time' => $match->end_time,
                    'is_positive' => true,
                ]);
            }
        }

        return response()->json($match);
    }

    /**
     * Extract chroma for a track (for pattern creation).
     */
    public function extractChroma(Track $track): JsonResponse
    {
        if ($track->analysis_status !== 'completed') {
            return response()->json([
                'error' => 'Track must be analyzed first'
            ], 400);
        }

        $taskId = $this->mlQueue->pushTask('extract_chroma', [
            'track_id' => $track->id,
        ]);

        return response()->json([
            'message' => 'Chroma extraction started',
            'task_id' => $taskId,
        ]);
    }

    /**
     * Extract pitch contour for a track.
     */
    public function extractPitch(Track $track): JsonResponse
    {
        if ($track->analysis_status !== 'completed') {
            return response()->json([
                'error' => 'Track must be analyzed first'
            ], 400);
        }

        $taskId = $this->mlQueue->pushTask('extract_pitch', [
            'track_id' => $track->id,
        ]);

        return response()->json([
            'message' => 'Pitch extraction started',
            'task_id' => $taskId,
        ]);
    }

    /**
     * Get segments for a track (from pattern_examples).
     */
    public function getTrackSegments(Request $request, Track $track): JsonResponse
    {
        $userId = (int) ($request->user()?->id ?? 0);
        if ($userId <= 0) {
            return response()->json([
                'error' => 'Unauthorized',
                'message' => 'Authentication is required',
            ], 401);
        }

        return response()->json($this->trackFeedbackSnapshots->buildForUser($userId, $track));
    }

    private function resolvePreferenceScore(?int $userId, Track $track): ?float
    {
        if ($userId === null) {
            return null;
        }

        $storedScore = UserTrackScore::forUser($userId)
            ->where('track_id', $track->id)
            ->value('preference_score');

        if ($storedScore !== null) {
            return round((float) $storedScore, 4);
        }

        $likedTrackIds = TrackRating::forUser($userId)
            ->where('rating', '>=', 4)
            ->pluck('track_id')
            ->unique()
            ->values()
            ->all();

        if ($likedTrackIds === []) {
            return null;
        }

        $likedTracks = Track::whereIn('id', $likedTrackIds)
            ->where('analysis_status', 'completed')
            ->get(['genre', 'mood', 'bpm', 'key']);

        if ($likedTracks->isEmpty()) {
            return null;
        }

        $preferredGenres = $likedTracks->pluck('genre')->filter()->countBy();
        $preferredMoods = $likedTracks->pluck('mood')->filter()->countBy();
        $avgBpm = $likedTracks->avg('bpm') ?: null;
        $preferredKeys = $likedTracks->pluck('key')->filter()->countBy();

        $score = 0.3;

        if ($track->genre && $preferredGenres->isNotEmpty()) {
            $genreScore = 0.0;
            $trackGenre = mb_strtolower(trim((string) $track->genre));
            foreach ($preferredGenres as $genre => $count) {
                $preferredGenre = mb_strtolower(trim((string) $genre));
                if ($trackGenre === $preferredGenre) {
                    $genreScore = max($genreScore, 0.35);
                } elseif (str_contains($trackGenre, $preferredGenre) || str_contains($preferredGenre, $trackGenre)) {
                    $genreScore = max($genreScore, 0.25);
                } elseif ($this->areGenresRelated($trackGenre, $preferredGenre)) {
                    $genreScore = max($genreScore, 0.15);
                }
            }
            $score += $genreScore;
        }

        if ($track->bpm && $avgBpm) {
            $bpmDiff = abs($track->bpm - $avgBpm);
            if ($bpmDiff < 10) {
                $score += 0.15;
            } elseif ($bpmDiff < 20) {
                $score += 0.10;
            } elseif ($bpmDiff < 40) {
                $score += 0.05;
            }
        }

        if ($track->mood && $preferredMoods->has($track->mood)) {
            $score += 0.10;
        }

        if ($track->key && $preferredKeys->has($track->key)) {
            $score += 0.05;
        }

        return round(min(0.99, $score), 4);
    }

    private function areGenresRelated(string $genre1, string $genre2): bool
    {
        $groups = [
            ['dubstep', 'drum & bass', 'jungle', 'grime', 'uk garage', 'halftime', 'breakbeat'],
            ['house', 'tech house', 'deep house', 'progressive house', 'electro house'],
            ['trance', 'progressive trance', 'psytrance', 'uplifting trance'],
            ['hardstyle', 'hands up', 'jumpstyle', 'hardcore', 'happy hardcore'],
        ];

        foreach ($groups as $group) {
            if (in_array($genre1, $group, true) && in_array($genre2, $group, true)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Analyze track to detect segments.
     */
    public function analyzeSegments(Track $track): JsonResponse
    {
        if ($track->analysis_status !== 'completed') {
            return response()->json([
                'error' => 'Track must be analyzed first'
            ], 400);
        }

        $lockKey = "segment-analysis-enqueue:{$track->id}";

        try {
            return Cache::lock($lockKey, 10)->block(3, function () use ($track) {
                $existingStatus = MLQueueService::getSegmentAnalysisStatus($track->id);

                if (($existingStatus['status'] ?? null) === 'processing') {
                    return response()->json([
                        'message' => 'Segment analysis already queued.',
                        'track_id' => $track->id,
                        'task_id' => $existingStatus['task_id'] ?? null,
                        'status' => 'processing',
                        'deduplicated' => true,
                    ], 202);
                }

                $queuedTask = MLQueueService::promoteAndDeduplicateQueuedSegmentAnalysisTask($track->id);
                if ($queuedTask !== null) {
                    $latestStatus = MLQueueService::getSegmentAnalysisStatus($track->id);

                    if (($latestStatus['status'] ?? null) === 'processing') {
                        return response()->json([
                            'message' => 'Segment analysis already queued.',
                            'track_id' => $track->id,
                            'task_id' => $latestStatus['task_id'] ?? $queuedTask['task_id'],
                            'status' => 'processing',
                            'deduplicated' => true,
                        ], 202);
                    }

                    MLQueueService::updateSegmentAnalysisStatus($track->id, [
                        'status' => 'queued',
                        'task_id' => $queuedTask['task_id'],
                        'track_id' => $track->id,
                        'progress' => 0,
                        'message' => 'Waiting in queue...',
                    ]);

                    return response()->json([
                        'message' => 'Segment analysis already queued.',
                        'track_id' => $track->id,
                        'task_id' => $queuedTask['task_id'],
                        'status' => 'queued',
                        'deduplicated' => true,
                    ], 202);
                }

                if ($this->isActiveSegmentAnalysisStatus($existingStatus)) {
                    return response()->json([
                        'message' => 'Segment analysis already queued.',
                        'track_id' => $track->id,
                        'task_id' => $existingStatus['task_id'] ?? null,
                        'status' => $existingStatus['status'] ?? 'queued',
                        'deduplicated' => true,
                    ], 202);
                }

                $taskId = MLQueueService::pushPrioritySegmentAnalysis($track->id);

                return response()->json([
                    'message' => 'Segment analysis started',
                    'track_id' => $track->id,
                    'task_id' => $taskId,
                    'deduplicated' => false,
                ]);
            });
        } catch (LockTimeoutException $e) {
            $existingStatus = MLQueueService::getSegmentAnalysisStatus($track->id);

            if ($this->isActiveSegmentAnalysisStatus($existingStatus)) {
                return response()->json([
                    'message' => 'Segment analysis already queued.',
                    'track_id' => $track->id,
                    'task_id' => $existingStatus['task_id'] ?? null,
                    'status' => $existingStatus['status'] ?? 'queued',
                    'deduplicated' => true,
                ], 202);
            }

            $queuedTask = MLQueueService::getQueuedSegmentAnalysisTask($track->id);
            if ($queuedTask !== null) {
                $latestStatus = MLQueueService::getSegmentAnalysisStatus($track->id);

                if ($this->isActiveSegmentAnalysisStatus($latestStatus)) {
                    return response()->json([
                        'message' => 'Segment analysis already queued.',
                        'track_id' => $track->id,
                        'task_id' => $latestStatus['task_id'] ?? $queuedTask['task_id'],
                        'status' => $latestStatus['status'] ?? 'queued',
                        'deduplicated' => true,
                    ], 202);
                }

                MLQueueService::updateSegmentAnalysisStatus($track->id, [
                    'status' => 'queued',
                    'task_id' => $queuedTask['task_id'],
                    'track_id' => $track->id,
                    'progress' => 0,
                    'message' => 'Waiting in queue...',
                ]);

                return response()->json([
                    'message' => 'Segment analysis already queued.',
                    'track_id' => $track->id,
                    'task_id' => $queuedTask['task_id'],
                    'status' => 'queued',
                    'deduplicated' => true,
                ], 202);
            }

            return response()->json([
                'error' => 'Segment analysis enqueue is busy',
                'message' => 'Could not reserve the segment analysis slot. Try again.',
            ], 409);
        }
    }
    
    /**
     * Get segment analysis status for a track.
     */
    public function getAnalysisStatus(Track $track): JsonResponse
    {
        $status = MLQueueService::getSegmentAnalysisStatus($track->id);
        
        if (!$status) {
            return response()->json([
                'status' => 'idle',
                'track_id' => $track->id,
            ]);
        }
        
        return response()->json($status);
    }

    /**
     * @param array<string, mixed>|null $status
     */
    private function isActiveSegmentAnalysisStatus(?array $status): bool
    {
        if (!is_array($status)) {
            return false;
        }

        return in_array($status['status'] ?? null, ['queued', 'processing'], true);
    }

    /**
     * Rate a segment (pattern example) with stars (0-5).
     */
    public function rateSegment(Request $request, int $segment): JsonResponse
    {
        $example = PatternExample::find($segment);
        
        if (!$example) {
            return response()->json(['error' => 'Segment not found'], 404);
        }

        $validated = $request->validate([
            'rating' => 'required|numeric|min:0|max:5',
        ]);

        $userRating = UserPatternExampleRating::updateOrCreate([
            'user_id' => $request->user()?->id,
            'pattern_example_id' => $example->id,
        ], [
            'rating' => $validated['rating'],
            'is_positive' => $validated['rating'] >= 3, // 3+ stars = positive
        ]);

        $userId = (int) ($request->user()?->id ?? 0);
        $computedTrackRating = $userId > 0
            ? $this->ratingPropagation->computeWeightedSegmentRating($userId, (int) $example->track_id)
            : null;
        $syncedTrackRating = $userId > 0
            ? $this->ratingPropagation->syncTrackFromSegments($userId, (int) $example->track_id, $computedTrackRating)
            : null;
        $effectiveTrackRating = $syncedTrackRating?->rating;
        $sentimentSource = $effectiveTrackRating !== null ? (float) $effectiveTrackRating : $computedTrackRating;

        return response()->json([
            'message' => 'Segment rated',
            'segment' => [
                'id' => $example->id,
                'pattern_example_id' => $example->id,
                'track_id' => $example->track_id,
                'start_time' => $example->start_time,
                'end_time' => $example->end_time,
                'rating' => $userRating->rating,
                'is_positive' => $userRating->is_positive,
            ],
            'track_rating' => $effectiveTrackRating,
            'computed_track_rating' => $computedTrackRating,
            'sentiment' => $this->ratingPropagation->sentimentForRating($sentimentSource),
            'feedback' => null,
        ]);
    }
}
