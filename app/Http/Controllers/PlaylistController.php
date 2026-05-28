<?php

namespace App\Http\Controllers;

use App\Models\Playlist;
use App\Services\PlaylistService;
use App\Services\AIPlaylistGeneratorService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use InvalidArgumentException;

/**
 * PlaylistController
 * 
 * Handles REST API endpoints for playlist management.
 * Requirements: 5.1, 5.2, 5.3, 5.4
 */
class PlaylistController extends Controller
{
    public function __construct(
        private readonly PlaylistService $playlistService,
        private readonly AIPlaylistGeneratorService $aiPlaylistGenerator
    ) {}

    /**
     * Display a listing of playlists.
     * 
     * GET /api/playlists
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function index(Request $request): JsonResponse
    {
        $query = Playlist::ownedBy((int) $request->user()->id);

        // Filter by AI-generated status
        if ($request->has('ai_generated')) {
            $isAiGenerated = filter_var($request->input('ai_generated'), FILTER_VALIDATE_BOOLEAN);
            $query->where('is_ai_generated', $isAiGenerated);
        }

        // Search by name
        if ($request->has('search')) {
            $search = $request->input('search');
            $query->where('name', 'ilike', "%{$search}%");
        }

        // Sorting
        $sortBy = $request->input('sort_by', 'created_at');
        $sortOrder = $request->input('sort_order', 'desc');
        $allowedSorts = ['name', 'created_at', 'updated_at', 'track_count', 'total_duration'];
        
        if (in_array($sortBy, $allowedSorts)) {
            $query->orderBy($sortBy, $sortOrder === 'asc' ? 'asc' : 'desc');
        }

        // Pagination
        $perPage = min((int) $request->input('per_page', 20), 100);
        $playlists = $query->paginate($perPage);

        return response()->json($playlists);
    }

    /**
     * Store a newly created playlist.
     * 
     * POST /api/playlists
     * Requirement 5.1: WHEN пользователь создаёт плейлист THEN THE Playlist_Manager 
     * SHALL создать запись с названием и описанием
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function store(Request $request): JsonResponse
    {
        $name = $request->input('name');
        $description = $request->input('description');

        if (empty($name) || !is_string($name)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Playlist name is required',
            ], 400);
        }

        try {
            $playlist = $this->playlistService->create((int) $request->user()->id, $name, $description);

            return response()->json([
                'message' => 'Playlist created successfully',
                'playlist' => $playlist,
            ], 201);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Creation failed',
                'message' => 'Failed to create playlist: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Display the specified playlist with its tracks.
     * 
     * GET /api/playlists/{id}
     *
     * @param int $id
     * @return JsonResponse
     */
    public function show(Request $request, int $id): JsonResponse
    {
        $playlist = Playlist::ownedBy((int) $request->user()->id)
            ->with('tracks')
            ->find($id);

        if (!$playlist) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Playlist not found',
            ], 404);
        }

        return response()->json($playlist);
    }

    /**
     * Update the specified playlist.
     * 
     * PUT /api/playlists/{id}
     * Requirement 5.5: THE Playlist_Manager SHALL позволять переименовывать 
     * плейлисты и изменять описание
     *
     * @param Request $request
     * @param int $id
     * @return JsonResponse
     */
    public function update(Request $request, int $id): JsonResponse
    {
        $data = [];

        if ($request->has('name')) {
            $data['name'] = $request->input('name');
        }

        if ($request->has('description')) {
            $data['description'] = $request->input('description');
        }

        if (empty($data)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'No fields to update',
            ], 400);
        }

        try {
            $playlist = $this->playlistService->update((int) $request->user()->id, $id, $data);

            return response()->json([
                'message' => 'Playlist updated successfully',
                'playlist' => $playlist,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Playlist not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Update failed',
                'message' => 'Failed to update playlist: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Remove the specified playlist.
     * 
     * DELETE /api/playlists/{id}
     * Requirement 5.4: WHEN пользователь удаляет плейлист THEN THE Playlist_Manager 
     * SHALL удалить плейлист и все связи с треками
     *
     * @param int $id
     * @return JsonResponse
     */
    public function destroy(Request $request, int $id): JsonResponse
    {
        try {
            $this->playlistService->delete((int) $request->user()->id, $id);

            return response()->json([
                'message' => 'Playlist deleted successfully',
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Playlist not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Deletion failed',
                'message' => 'Failed to delete playlist: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Add a track to the playlist.
     * 
     * POST /api/playlists/{id}/tracks
     * Requirement 5.2: WHEN пользователь добавляет трек в плейлист THEN THE 
     * Playlist_Manager SHALL добавить связь трека с плейлистом
     *
     * @param Request $request
     * @param int $id
     * @return JsonResponse
     */
    public function addTrack(Request $request, int $id): JsonResponse
    {
        $trackId = $request->input('track_id');
        $position = $request->input('position');

        if (empty($trackId) || !is_numeric($trackId)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Track ID is required',
            ], 400);
        }

        try {
            $this->playlistService->addTrack(
                (int) $request->user()->id,
                $id, 
                (int) $trackId, 
                $position !== null ? (int) $position : null
            );

            $playlist = Playlist::ownedBy((int) $request->user()->id)
                ->with('tracks')
                ->find($id);

            return response()->json([
                'message' => 'Track added to playlist successfully',
                'playlist' => $playlist,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Playlist or track not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to add track to playlist: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Remove a track from the playlist.
     * 
     * DELETE /api/playlists/{id}/tracks/{trackId}
     * Requirement 5.3: WHEN пользователь удаляет трек из плейлиста THEN THE 
     * Playlist_Manager SHALL удалить связь без удаления трека
     *
     * @param int $id
     * @param int $trackId
     * @return JsonResponse
     */
    public function removeTrack(int $id, int $trackId): JsonResponse
    {
        try {
            $this->playlistService->removeTrack((int) $request->user()->id, $id, $trackId);

            $playlist = Playlist::ownedBy((int) $request->user()->id)
                ->with('tracks')
                ->find($id);

            return response()->json([
                'message' => 'Track removed from playlist successfully',
                'playlist' => $playlist,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Playlist not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to remove track from playlist: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Reorder tracks in the playlist.
     * 
     * PUT /api/playlists/{id}/tracks/reorder
     * Requirement 5.6: THE Playlist_Manager SHALL позволять изменять порядок 
     * треков в плейлисте перетаскиванием
     *
     * @param Request $request
     * @param int $id
     * @return JsonResponse
     */
    public function reorderTracks(Request $request, int $id): JsonResponse
    {
        $trackOrder = $request->input('track_order');

        if (!is_array($trackOrder) || empty($trackOrder)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Track order array is required',
            ], 400);
        }

        // Validate that all elements are integers
        foreach ($trackOrder as $trackId) {
            if (!is_numeric($trackId)) {
                return response()->json([
                    'error' => 'Validation failed',
                    'message' => 'Track order must contain only track IDs',
                ], 400);
            }
        }

        try {
            $this->playlistService->reorderTracks((int) $request->user()->id, $id, array_map('intval', $trackOrder));

            $playlist = Playlist::ownedBy((int) $request->user()->id)
                ->with('tracks')
                ->find($id);

            return response()->json([
                'message' => 'Tracks reordered successfully',
                'playlist' => $playlist,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Playlist not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to reorder tracks: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Generate an AI playlist based on parameters.
     * 
     * POST /api/playlists/generate
     * Requirement 6.1: WHEN пользователь запрашивает AI-плейлист THEN THE 
     * AI_Playlist_Generator SHALL создать плейлист на основе параметров
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function generate(Request $request): JsonResponse
    {
        $params = [
            'name' => $request->input('name', 'AI Generated Playlist'),
            'duration' => $request->input('duration', 60),
            'genre' => $request->input('genre'),
            'mood' => $request->input('mood'),
            'bpm_min' => $request->input('bpm_min'),
            'bpm_max' => $request->input('bpm_max'),
            'use_preferences' => $request->boolean('use_preferences', true),
        ];

        try {
            $playlist = $this->aiPlaylistGenerator->generate((int) $request->user()->id, $params);

            return response()->json([
                'message' => 'AI playlist generated successfully',
                'playlist' => $playlist,
            ], 201);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Generation failed',
                'message' => 'Failed to generate AI playlist: ' . $e->getMessage(),
            ], 500);
        }
    }
}
