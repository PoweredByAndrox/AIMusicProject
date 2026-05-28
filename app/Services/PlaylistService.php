<?php

namespace App\Services;

use App\Models\Playlist;
use App\Models\PlaylistTrack;
use App\Models\Track;
use Illuminate\Support\Facades\DB;
use InvalidArgumentException;

/**
 * PlaylistService
 * 
 * Handles CRUD operations for playlists and track management within playlists.
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
 */
class PlaylistService
{
    /**
     * Create a new playlist.
     * 
     * Requirement 5.1: WHEN пользователь создаёт плейлист THEN THE Playlist_Manager 
     * SHALL создать запись с названием и описанием
     *
     * @param string $name The name of the playlist
     * @param string|null $description Optional description for the playlist
     * @return Playlist The created playlist
     * @throws InvalidArgumentException If name is empty
     */
    public function create(int $userId, string $name, ?string $description = null): Playlist
    {
        $name = trim($name);
        
        if (empty($name)) {
            throw new InvalidArgumentException('Playlist name cannot be empty');
        }

        return Playlist::create([
            'user_id' => $userId,
            'name' => $name,
            'description' => $description,
            'is_ai_generated' => false,
            'track_count' => 0,
            'total_duration' => 0,
        ]);
    }

    /**
     * Update an existing playlist.
     * 
     * Requirement 5.5: THE Playlist_Manager SHALL позволять переименовывать 
     * плейлисты и изменять описание
     *
     * @param int $playlistId The ID of the playlist to update
     * @param array{name?: string, description?: string|null} $data The data to update
     * @return Playlist The updated playlist
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If playlist not found
     * @throws InvalidArgumentException If name is provided but empty
     */
    public function update(int $userId, int $playlistId, array $data): Playlist
    {
        $playlist = Playlist::ownedBy($userId)->findOrFail($playlistId);

        if (array_key_exists('name', $data)) {
            $name = trim($data['name']);
            if (empty($name)) {
                throw new InvalidArgumentException('Playlist name cannot be empty');
            }
            $data['name'] = $name;
        }

        $playlist->update($data);
        
        return $playlist->fresh();
    }

    /**
     * Delete a playlist and all its track associations.
     * 
     * Requirement 5.4: WHEN пользователь удаляет плейлист THEN THE Playlist_Manager 
     * SHALL удалить плейлист и все связи с треками
     * Note: Tracks themselves are NOT deleted from the library
     *
     * @param int $playlistId The ID of the playlist to delete
     * @return void
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If playlist not found
     */
    public function delete(int $userId, int $playlistId): void
    {
        $playlist = Playlist::ownedBy($userId)->findOrFail($playlistId);
        
        // The cascade delete in the database will handle playlist_tracks
        $playlist->delete();
    }

    /**
     * Add a track to a playlist at a specific position.
     * 
     * Requirement 5.2: WHEN пользователь добавляет трек в плейлист THEN THE 
     * Playlist_Manager SHALL добавить связь трека с плейлистом
     *
     * @param int $playlistId The ID of the playlist
     * @param int $trackId The ID of the track to add
     * @param int|null $position Optional position (1-based). If null, adds to end
     * @return void
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If playlist or track not found
     * @throws InvalidArgumentException If track is already in the playlist
     */
    public function addTrack(int $userId, int $playlistId, int $trackId, ?int $position = null): void
    {
        $playlist = Playlist::ownedBy($userId)->findOrFail($playlistId);
        $track = Track::findOrFail($trackId);

        // Check if track is already in the playlist
        $existingTrack = PlaylistTrack::where('playlist_id', $playlistId)
            ->where('track_id', $trackId)
            ->first();

        if ($existingTrack) {
            throw new InvalidArgumentException('Track is already in the playlist');
        }

        DB::transaction(function () use ($playlist, $track, $position) {
            // Get the current max position
            $maxPosition = PlaylistTrack::where('playlist_id', $playlist->id)
                ->max('position') ?? 0;

            // Determine the target position
            if ($position === null || $position > $maxPosition + 1) {
                $targetPosition = $maxPosition + 1;
            } else {
                $targetPosition = max(1, $position);
                
                // Shift existing tracks at and after the target position
                PlaylistTrack::where('playlist_id', $playlist->id)
                    ->where('position', '>=', $targetPosition)
                    ->increment('position');
            }

            // Create the playlist track entry
            PlaylistTrack::create([
                'playlist_id' => $playlist->id,
                'track_id' => $track->id,
                'position' => $targetPosition,
                'added_at' => now(),
            ]);

            // Update playlist statistics
            $playlist->increment('track_count');
            $playlist->increment('total_duration', (int) $track->duration);
        });
    }

    /**
     * Remove a track from a playlist.
     * 
     * Requirement 5.3: WHEN пользователь удаляет трек из плейлиста THEN THE 
     * Playlist_Manager SHALL удалить связь без удаления трека
     *
     * @param int $playlistId The ID of the playlist
     * @param int $trackId The ID of the track to remove
     * @return void
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If playlist not found
     * @throws InvalidArgumentException If track is not in the playlist
     */
    public function removeTrack(int $userId, int $playlistId, int $trackId): void
    {
        $playlist = Playlist::ownedBy($userId)->findOrFail($playlistId);

        $playlistTrack = PlaylistTrack::where('playlist_id', $playlistId)
            ->where('track_id', $trackId)
            ->first();

        if (!$playlistTrack) {
            throw new InvalidArgumentException('Track is not in the playlist');
        }

        DB::transaction(function () use ($playlist, $playlistTrack, $trackId) {
            $removedPosition = $playlistTrack->position;
            
            // Get track duration before deletion
            $track = Track::find($trackId);
            $trackDuration = $track ? (int) $track->duration : 0;

            // Delete the playlist track entry
            $playlistTrack->delete();

            // Shift positions of tracks after the removed one
            PlaylistTrack::where('playlist_id', $playlist->id)
                ->where('position', '>', $removedPosition)
                ->decrement('position');

            // Update playlist statistics
            $playlist->decrement('track_count');
            $playlist->decrement('total_duration', $trackDuration);
            
            // Ensure total_duration doesn't go negative
            if ($playlist->total_duration < 0) {
                $playlist->update(['total_duration' => 0]);
            }
        });
    }

    /**
     * Reorder tracks in a playlist.
     * 
     * Requirement 5.6: THE Playlist_Manager SHALL позволять изменять порядок 
     * треков в плейлисте перетаскиванием
     *
     * @param int $playlistId The ID of the playlist
     * @param array<int> $trackOrder Array of track IDs in the desired order
     * @return void
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If playlist not found
     * @throws InvalidArgumentException If track order doesn't match playlist tracks
     */
    public function reorderTracks(int $userId, int $playlistId, array $trackOrder): void
    {
        $playlist = Playlist::ownedBy($userId)->findOrFail($playlistId);

        // Get current track IDs in the playlist
        $currentTrackIds = PlaylistTrack::where('playlist_id', $playlistId)
            ->pluck('track_id')
            ->toArray();

        // Validate that the provided track order contains exactly the same tracks
        $providedTrackIds = array_values($trackOrder);
        
        sort($currentTrackIds);
        $sortedProvidedIds = $providedTrackIds;
        sort($sortedProvidedIds);

        if ($currentTrackIds !== $sortedProvidedIds) {
            throw new InvalidArgumentException(
                'Track order must contain exactly the same tracks as currently in the playlist'
            );
        }

        DB::transaction(function () use ($playlistId, $trackOrder) {
            // Update positions based on the new order
            foreach ($trackOrder as $position => $trackId) {
                PlaylistTrack::where('playlist_id', $playlistId)
                    ->where('track_id', $trackId)
                    ->update(['position' => $position + 1]); // 1-based positions
            }
        });
    }
}
