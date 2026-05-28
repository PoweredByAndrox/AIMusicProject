<?php

namespace App\Http\Controllers;

use App\Services\SmartRadioService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Database\Eloquent\ModelNotFoundException;

/**
 * SmartRadioController
 * 
 * Handles REST API endpoints for Smart Radio functionality.
 * Requirements: 15.1, 17.1
 */
class SmartRadioController extends Controller
{
    public function __construct(
        private readonly SmartRadioService $radioService
    ) {}

    /**
     * Enable Smart Radio mode.
     * 
     * POST /api/radio/enable
     * Requirement 15.1: WHEN пользователь включает Smart_Radio THEN THE System SHALL 
     * автоматически добавлять треки в Queue
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function enable(Request $request): JsonResponse
    {
        $seedTrackId = $request->input('seed_track_id');

        if ($seedTrackId === null || !is_numeric($seedTrackId)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'seed_track_id is required and must be a number',
            ], 400);
        }

        try {
            $result = $this->radioService->enable((int) $seedTrackId);

            return response()->json([
                'message' => $result['message'],
                'enabled' => $result['enabled'],
                'seed_track_id' => $result['seed_track_id'],
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Seed track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to enable Smart Radio: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Disable Smart Radio mode.
     * 
     * POST /api/radio/disable
     * Requirement 15.6: WHEN пользователь выключает Smart_Radio THEN THE System SHALL 
     * прекратить автоматическое добавление треков
     *
     * @return JsonResponse
     */
    public function disable(): JsonResponse
    {
        try {
            $result = $this->radioService->disable();

            return response()->json([
                'message' => $result['message'],
                'enabled' => $result['enabled'],
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to disable Smart Radio: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get Smart Radio status.
     * 
     * GET /api/radio/status
     *
     * @return JsonResponse
     */
    public function status(): JsonResponse
    {
        try {
            $state = $this->radioService->getState();

            return response()->json([
                'enabled' => $state['enabled'],
                'seed_track_id' => $state['seedTrackId'],
                'enabled_at' => $state['enabledAt'],
                'disabled_at' => $state['disabledAt'],
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to get Smart Radio status: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get next track recommendations.
     * 
     * GET /api/radio/recommendations
     * Requirement 15.2: WHEN Queue содержит менее 3 треков THEN THE Smart_Radio SHALL 
     * добавить новые рекомендованные треки
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function recommendations(Request $request): JsonResponse
    {
        $count = $request->input('count', 10);

        if (!is_numeric($count) || (int) $count < 1 || (int) $count > 50) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'count must be a number between 1 and 50',
            ], 400);
        }

        try {
            // Get recommendations based on user preferences (works without enabling radio)
            $recommendations = $this->radioService->getPreferenceBasedRecommendations((int) $count);

            return response()->json([
                'count' => $recommendations->count(),
                'tracks' => $recommendations->map(function ($track) {
                    return [
                        'id' => $track->id,
                        'title' => $track->title,
                        'artist' => $track->artist,
                        'album' => $track->album,
                        'duration' => $track->duration,
                        'genre' => $track->genre,
                        'mood' => $track->mood,
                        'bpm' => $track->bpm,
                        'key' => $track->key,
                        'analysis_status' => $track->analysis_status,
                        'recommendation_score' => $track->recommendation_score ?? null,
                    ];
                }),
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to get recommendations: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Record user feedback for a track.
     * 
     * POST /api/radio/feedback
     * Requirement 17.1: WHEN пользователь нажимает "Больше такого" THEN THE Smart_Radio 
     * SHALL увеличить вес текущего трека в рекомендациях
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function feedback(Request $request): JsonResponse
    {
        $trackId = $request->input('track_id');
        $feedbackType = $request->input('feedback_type');

        if ($trackId === null || !is_numeric($trackId)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'track_id is required and must be a number',
            ], 400);
        }

        if (!in_array($feedbackType, ['more', 'less'], true)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'feedback_type must be "more" or "less"',
            ], 400);
        }

        try {
            $result = $this->radioService->recordFeedback((int) $trackId, $feedbackType);

            return response()->json([
                'message' => 'Feedback recorded successfully',
                'track_id' => $result['track_id'],
                'feedback_type' => $result['feedback_type'],
                'weight_adjustment' => $result['weight_adjustment'],
                'new_weight' => $result['new_weight'],
            ], 201);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to record feedback: ' . $e->getMessage(),
            ], 500);
        }
    }
}
