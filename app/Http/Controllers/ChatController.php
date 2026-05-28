<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Models\ListeningHistory;
use App\Models\PreferenceModelStatus;
use App\Models\TrackRating;
use App\Models\UserTrackScore;
use App\Services\ChatSearchService;
use App\Services\LLMChatService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Redis;

/**
 * ChatController - API for AI Music Chat segment search.
 * 
 * Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
 */
class ChatController extends Controller
{
    public function __construct(
        private readonly ChatSearchService $chatSearchService,
        private readonly LLMChatService $llmChatService
    ) {}

    /**
     * Search for similar segments.
     * 
     * POST /api/chat/search
     * 
     * Request body:
     * {
     *   "track_id": 123,
     *   "start_time": 29.0,
     *   "end_time": 35.0,
     *   "limit": 10,
     *   "min_similarity": 0.3
     * }
     * 
     * Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function search(Request $request): JsonResponse
    {
        // Validate required fields
        $trackId = $request->input('track_id');
        $startTime = $request->input('start_time');
        $endTime = $request->input('end_time');
        
        // Requirement 6.6: Validate track_id
        if ($trackId === null || !is_numeric($trackId)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'track_id is required and must be a number',
                'code' => 'INVALID_TRACK_ID'
            ], 422);
        }
        
        // Requirement 6.6: Validate start_time
        if ($startTime === null || !is_numeric($startTime)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'start_time is required and must be a number',
                'code' => 'INVALID_START_TIME'
            ], 422);
        }
        
        // Requirement 6.6: Validate end_time
        if ($endTime === null || !is_numeric($endTime)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'end_time is required and must be a number',
                'code' => 'INVALID_END_TIME'
            ], 422);
        }
        
        $startTime = (float) $startTime;
        $endTime = (float) $endTime;
        $trackId = (int) $trackId;
        
        // Requirement 6.6: Validate time range
        if ($startTime < 0 || $endTime < 0) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Time values cannot be negative',
                'code' => 'NEGATIVE_TIME'
            ], 422);
        }
        
        if ($startTime >= $endTime) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'start_time must be less than end_time',
                'code' => 'INVALID_RANGE'
            ], 422);
        }
        
        // Requirement 6.5: Check if track exists
        $track = Track::find($trackId);
        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => "Track with ID {$trackId} not found",
                'code' => 'TRACK_NOT_FOUND'
            ], 404);
        }
        
        // Optional parameters with defaults
        $limit = min((int) $request->input('limit', 10), 50); // Max 50 results
        $minSimilarity = max(0.0, min(1.0, (float) $request->input('min_similarity', 0.3)));
        
        try {
            $results = $this->chatSearchService->searchSimilarSegments(
                $trackId,
                $startTime,
                $endTime,
                $limit,
                $minSimilarity
            );
            
            return response()->json($results);
            
        } catch (\InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
                'code' => 'VALIDATION_ERROR'
            ], 422);
            
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Search failed',
                'message' => $e->getMessage(),
                'code' => 'SEARCH_ERROR'
            ], 500);
        }
    }
    
    /**
     * Get track info for chat context.
     * 
     * GET /api/chat/track/{id}
     *
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function getTrack(int $id): JsonResponse
    {
        $trackInfo = $this->chatSearchService->getTrackInfo($id);
        
        if (!$trackInfo) {
            return response()->json([
                'error' => 'Not found',
                'message' => "Track with ID {$id} not found"
            ], 404);
        }
        
        return response()->json($trackInfo);
    }
    
    /**
     * Get personalized recommendations with explanations.
     * 
     * GET /api/chat/recommend
     * 
     * Returns tracks the user hasn't listened to, with explanations
     * of why each track might match their taste.
     * Uses ML worker for embedding-based similarity.
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function recommend(Request $request): JsonResponse
    {
        $limit = min((int) $request->input('limit', 5), 20);
        
        // Generate unique task ID for result correlation
        $taskId = uniqid('chat_recommend_', true);
        
        // Push task to ML worker queue using MLQueueService
        \App\Services\MLQueueService::push([
            'type' => 'embedding_recommendations',
            'task_id' => $taskId,
            'user_id' => (int) $request->user()->id,
            'limit' => $limit,
            'created_at' => now()->toIso8601String(),
        ]);
        
        // Wait for result (with timeout)
        $resultKey = "chat_recommend_result:{$taskId}";
        $startWait = time();
        $timeout = 30; // 30 second timeout
        
        while (time() - $startWait < $timeout) {
            $result = Redis::connection()->get($resultKey);
            
            if ($result) {
                // Clean up result key
                Redis::connection()->del($resultKey);
                return response()->json(json_decode($result, true));
            }
            
            // Small delay before next check
            usleep(100000); // 100ms
        }
        
        // Timeout - fall back to simple recommendations
        return $this->getSimpleRecommendations($limit);
    }
    
    /**
     * Fallback simple recommendations without ML worker.
     */
    private function getSimpleRecommendations(int $limit): JsonResponse
    {
        $userId = (int) request()->user()->id;

        // Get user's liked tracks for preference analysis
        $likedTracks = Track::likedByUser($userId)
            ->where('analysis_status', 'completed')
            ->get();
        
        // Get highly rated tracks (4-5 stars)
        $highRatedTrackIds = TrackRating::forUser($userId)
            ->where('rating', '>=', 4)
            ->pluck('track_id')
            ->toArray();
        
        $highRatedTracks = Track::whereIn('id', $highRatedTrackIds)
            ->where('analysis_status', 'completed')
            ->get();
        
        // Combine liked and highly rated for preference profile
        $preferredTracks = $likedTracks->merge($highRatedTracks)->unique('id');

        if ($preferredTracks->isEmpty()) {
            $fallbackCandidates = Track::where('analysis_status', 'completed')
                ->whereDoesntHave('trackLikes', function ($query) use ($userId) {
                    $query->where('user_id', $userId);
                })
                ->whereNotNull('genre')
                ->get();
            $fallbackScores = UserTrackScore::forUser($userId)
                ->whereIn('track_id', $fallbackCandidates->pluck('id')->all())
                ->pluck('preference_score', 'track_id')
                ->toArray();

            if (!empty($fallbackScores)) {
                $fallbackRecommendations = $fallbackCandidates->map(function ($track) use ($fallbackScores) {
                    $score = (float) (($fallbackScores[$track->id] ?? 0.0) * 100.0);

                    return [
                        'track' => $track,
                        'score' => $score,
                        'reasons' => $score > 0 ? ['Похож по звуку на ваши лайки и сегменты'] : [],
                        'explanation' => $score > 0 ? 'Похож по звуку на ваши лайки и сегменты' : 'Может вам понравиться',
                    ];
                })
                ->filter(fn($r) => $r['score'] > 0)
                ->sortByDesc('score')
                ->take($limit)
                ->values();

                return response()->json([
                    'recommendations' => $fallbackRecommendations,
                    'total' => $fallbackRecommendations->count(),
                    'preference_summary' => [
                        'top_genres' => [],
                        'top_moods' => [],
                        'avg_bpm' => null,
                        'tracks_analyzed' => count($fallbackScores),
                    ],
                    'has_preferences' => true,
                    'message' => $fallbackRecommendations->isEmpty() ? 'РќРµ РЅР°С€С‘Р» РїРѕРґС…РѕРґСЏС‰РёС… С‚СЂРµРєРѕРІ.' : null,
                ]);
            }
        }
        
        if ($preferredTracks->isEmpty()) {
            return response()->json([
                'recommendations' => [],
                'message' => 'Недостаточно данных о ваших предпочтениях. Поставьте лайки или оценки трекам.',
                'has_preferences' => false,
            ]);
        }
        
        // Build preference profile
        $preferredGenres = $preferredTracks->pluck('genre')->filter()->countBy()->sortDesc();
        $preferredMoods = $preferredTracks->pluck('mood')->filter()->countBy()->sortDesc();
        $avgBpm = $preferredTracks->avg('bpm') ?: 140;
        
        // Get listening history to exclude
        $listenedTrackIds = ListeningHistory::forUser($userId)->pluck('track_id')->unique()->toArray();
        
        // Get candidate tracks
        $candidates = Track::where('analysis_status', 'completed')
            ->whereNotIn('id', $listenedTrackIds)
            ->whereDoesntHave('trackLikes', function ($query) use ($userId) {
                $query->where('user_id', $userId);
            })
            ->whereNotNull('genre')
            ->get();
        $preferenceScores = UserTrackScore::forUser($userId)
            ->whereIn('track_id', $candidates->pluck('id')->all())
            ->pluck('preference_score', 'track_id')
            ->toArray();
        
        if ($candidates->isEmpty()) {
            return response()->json([
                'recommendations' => [],
                'message' => 'Вы уже прослушали все доступные треки!',
                'has_preferences' => true,
            ]);
        }
        
        // Score candidates
        $recommendations = $candidates->map(function ($track) use ($preferredGenres, $preferredMoods, $avgBpm, $preferenceScores) {
            $score = array_key_exists($track->id, $preferenceScores)
                ? ((float) $preferenceScores[$track->id] * 100.0)
                : 0;
            $reasons = [];

            if (array_key_exists($track->id, $preferenceScores)) {
                $reasons[] = 'Похож по звуку на ваши лайки и сегменты';
            }
            
            if ($track->genre && $preferredGenres->isNotEmpty()) {
                $topGenre = $preferredGenres->keys()->first();
                if ($track->genre === $topGenre) {
                    $score += 40;
                    $reasons[] = "Жанр {$track->genre} — ваш любимый";
                } elseif ($preferredGenres->has($track->genre)) {
                    $score += 30;
                    $reasons[] = "Жанр {$track->genre} вам нравится";
                }
            }
            
            if ($track->bpm) {
                $bpmDiff = abs($track->bpm - $avgBpm);
                if ($bpmDiff < 10) {
                    $score += 20;
                    $reasons[] = "Темп {$track->bpm} BPM близок к вашему среднему";
                }
            }
            
            if ($track->mood && $preferredMoods->isNotEmpty() && $preferredMoods->has($track->mood)) {
                $score += 20;
                $reasons[] = "Настроение \"{$track->mood}\" вам подходит";
            }
            
            return [
                'track' => $track,
                'score' => $score,
                'reasons' => $reasons,
                'explanation' => implode('. ', $reasons) ?: 'Может вам понравиться',
            ];
        })
        ->filter(fn($r) => $r['score'] > 20)
        ->sortByDesc('score')
        ->take($limit)
        ->values();
        
        return response()->json([
            'recommendations' => $recommendations,
            'total' => $recommendations->count(),
            'preference_summary' => [
                'top_genres' => $preferredGenres->take(3)->keys()->toArray(),
                'top_moods' => $preferredMoods->take(2)->keys()->toArray(),
                'avg_bpm' => round($avgBpm),
                'tracks_analyzed' => $preferredTracks->count(),
            ],
            'has_preferences' => true,
            'message' => $recommendations->isEmpty() ? 'Не нашёл подходящих треков.' : null,
        ]);
    }
    
    /**
     * AI Chat with natural language understanding.
     * 
     * POST /api/chat/ai
     * 
     * Request body:
     * {
     *   "message": "Найди мне что-нибудь похожее на Drum & Bass но помедленнее",
     *   "context": { "current_track": {...} }
     * }
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function aiChat(Request $request): JsonResponse
    {
        $message = $request->input('message');
        
        if (empty($message)) {
            return response()->json([
                'error' => 'Message is required',
            ], 422);
        }
        
        $context = $request->input('context');
        
        try {
            $result = $this->llmChatService->chat($message, $context);
            return response()->json($result);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'AI chat failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }
    
    /**
     * Check LLM service health.
     * 
     * GET /api/chat/ai/health
     *
     * @return JsonResponse
     */
    public function aiHealth(): JsonResponse
    {
        $health = $this->llmChatService->checkHealth();
        $status = $health['status'] === 'ok' ? 200 : 503;
        return response()->json($health, $status);
    }
    
}
