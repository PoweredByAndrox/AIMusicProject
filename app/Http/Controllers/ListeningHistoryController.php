<?php

namespace App\Http\Controllers;

use App\Services\ListeningHistoryService;
use App\Services\SkipTrackerService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Database\Eloquent\ModelNotFoundException;

/**
 * ListeningHistoryController
 * 
 * Handles REST API endpoints for listening history and skip tracking.
 * Requirements: 16.3, 18.1
 */
class ListeningHistoryController extends Controller
{
    public function __construct(
        private readonly ListeningHistoryService $historyService,
        private readonly SkipTrackerService $skipService
    ) {}

    /**
     * Record the start of a track playback.
     * 
     * POST /api/tracks/{id}/play
     * Requirement 16.3: THE Listening_History SHALL записывать: track_id, timestamp, 
     * duration_played, completed
     *
     * @param Request $request
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function recordPlay(Request $request, int $id): JsonResponse
    {
        $source = $request->input('source');

        // Validate source if provided
        $validSources = ['queue', 'playlist', 'radio', 'ai_dj'];
        if ($source !== null && !in_array($source, $validSources, true)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Source must be one of: ' . implode(', ', $validSources),
            ], 400);
        }

        try {
            $history = $this->historyService->recordPlay($id, $source);

            return response()->json([
                'message' => 'Play recorded successfully',
                'history' => [
                    'id' => $history->id,
                    'track_id' => $history->track_id,
                    'started_at' => $history->started_at,
                    'source' => $history->source,
                ],
            ], 201);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to record play: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Record the completion or stop of a track playback.
     * 
     * PUT /api/listening-history/{id}/complete
     *
     * @param Request $request
     * @param int $id Listening history entry ID
     * @return JsonResponse
     */
    public function recordComplete(Request $request, int $id): JsonResponse
    {
        $durationPlayed = $request->input('duration_played');
        $completed = $request->boolean('completed', false);

        if ($durationPlayed === null || !is_numeric($durationPlayed)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'duration_played is required and must be a number',
            ], 400);
        }

        if ((float) $durationPlayed < 0) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'duration_played cannot be negative',
            ], 400);
        }

        try {
            $history = $this->historyService->recordComplete($id, (float) $durationPlayed, $completed);

            return response()->json([
                'message' => 'Playback completion recorded successfully',
                'history' => [
                    'id' => $history->id,
                    'track_id' => $history->track_id,
                    'started_at' => $history->started_at,
                    'duration_played' => $history->duration_played,
                    'completed' => $history->completed,
                    'source' => $history->source,
                ],
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Listening history entry not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to record completion: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get recent listening history.
     * 
     * GET /api/listening-history
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function getRecentHistory(Request $request): JsonResponse
    {
        $hours = $request->input('hours', 24);

        if (!is_numeric($hours) || (int) $hours < 1) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'hours must be a positive integer',
            ], 400);
        }

        try {
            $history = $this->historyService->getRecentHistory((int) $hours);

            return response()->json([
                'hours' => (int) $hours,
                'count' => $history->count(),
                'history' => $history,
            ]);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to get history: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Record a skip event for a track.
     * 
     * POST /api/tracks/{id}/skip
     * Requirement 18.1: WHEN пользователь пропускает трек в первые 30 секунд 
     * THEN THE Skip_Tracker SHALL записать событие пропуска
     *
     * @param Request $request
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function recordSkip(Request $request, int $id): JsonResponse
    {
        $positionAtSkip = $request->input('position_at_skip');
        $source = $request->input('source');

        if ($positionAtSkip === null || !is_numeric($positionAtSkip)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'position_at_skip is required and must be a number',
            ], 400);
        }

        if ((float) $positionAtSkip < 0) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'position_at_skip cannot be negative',
            ], 400);
        }

        // Validate source if provided
        $validSources = ['queue', 'playlist', 'radio', 'ai_dj'];
        if ($source !== null && !in_array($source, $validSources, true)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Source must be one of: ' . implode(', ', $validSources),
            ], 400);
        }

        try {
            $skip = $this->skipService->recordSkip($id, (float) $positionAtSkip, $source);

            // Get updated skip stats
            $stats = $this->skipService->getSkipStats($id);

            return response()->json([
                'message' => 'Skip recorded successfully',
                'skip' => [
                    'id' => $skip->id,
                    'track_id' => $skip->track_id,
                    'position_at_skip' => $skip->position_at_skip,
                    'source' => $skip->source,
                    'skipped_at' => $skip->skipped_at,
                ],
                'stats' => $stats,
            ], 201);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to record skip: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get skip statistics for a track.
     * 
     * GET /api/tracks/{id}/skip-stats
     * Requirement 18.5: THE System SHALL отображать статистику пропусков 
     * в информации о треке
     *
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function getSkipStats(int $id): JsonResponse
    {
        try {
            $stats = $this->skipService->getSkipStats($id);

            return response()->json([
                'track_id' => $id,
                'stats' => $stats,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to get skip stats: ' . $e->getMessage(),
            ], 500);
        }
    }
}
