<?php

namespace App\Http\Controllers;

use App\Services\UserTrackPreferenceService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use InvalidArgumentException;

/**
 * UserTrackPreferenceController
 * 
 * Handles REST API endpoints for user track preferences (volume, position).
 * Requirements: 4.1, 4.2
 */
class UserTrackPreferenceController extends Controller
{
    public function __construct(
        private readonly UserTrackPreferenceService $preferenceService
    ) {}

    /**
     * Get the preferences for a track.
     * 
     * GET /api/tracks/{id}/preferences
     *
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function show(Request $request, int $id): JsonResponse
    {
        try {
            $preferences = $this->preferenceService->getPreferences((int) $request->user()->id, $id);

            if ($preferences === null) {
                return response()->json([
                    'track_id' => $id,
                    'volume' => 1.0,
                    'last_position' => 0,
                    'has_preferences' => false,
                ]);
            }

            return response()->json([
                'track_id' => $id,
                'volume' => $preferences->volume,
                'last_position' => $preferences->last_position,
                'updated_at' => $preferences->updated_at,
                'has_preferences' => true,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to get preferences: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Update the preferences for a track.
     * 
     * PUT /api/tracks/{id}/preferences
     * 
     * Requirement 4.1: WHEN пользователь изменяет громкость для трека 
     * THEN THE User_Preference SHALL сохранить настройку громкости
     * 
     * Requirement 4.2: WHEN пользователь останавливает воспроизведение 
     * THEN THE User_Preference SHALL сохранить текущую позицию
     *
     * @param Request $request
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function update(Request $request, int $id): JsonResponse
    {
        $volume = $request->input('volume');
        $position = $request->input('last_position');

        // At least one preference must be provided
        if ($volume === null && $position === null) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'At least one of volume or last_position must be provided',
            ], 400);
        }

        // Validate types
        if ($volume !== null && !is_numeric($volume)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Volume must be a number',
            ], 400);
        }

        if ($position !== null && !is_numeric($position)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Last position must be a number',
            ], 400);
        }

        try {
            $preferences = $this->preferenceService->savePreferences(
                (int) $request->user()->id,
                $id,
                $volume !== null ? (float) $volume : null,
                $position !== null ? (float) $position : null
            );

            return response()->json([
                'message' => 'Preferences updated successfully',
                'preferences' => [
                    'track_id' => $preferences->track_id,
                    'volume' => $preferences->volume,
                    'last_position' => $preferences->last_position,
                    'updated_at' => $preferences->updated_at,
                ],
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to update preferences: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Delete the preferences for a track.
     * 
     * DELETE /api/tracks/{id}/preferences
     *
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function destroy(Request $request, int $id): JsonResponse
    {
        try {
            $this->preferenceService->deletePreferences((int) $request->user()->id, $id);

            return response()->json([
                'message' => 'Preferences deleted successfully',
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to delete preferences: ' . $e->getMessage(),
            ], 500);
        }
    }
}
