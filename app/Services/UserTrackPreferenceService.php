<?php

namespace App\Services;

use App\Models\Track;
use App\Models\UserTrackPreference;
use Illuminate\Support\Facades\DB;
use InvalidArgumentException;

/**
 * UserTrackPreferenceService
 * 
 * Handles user-specific track preferences (volume, last position).
 * Requirements: 4.1, 4.2, 4.4
 */
class UserTrackPreferenceService
{
    /**
     * Minimum allowed volume value.
     */
    private const MIN_VOLUME = 0.0;

    /**
     * Maximum allowed volume value.
     */
    private const MAX_VOLUME = 1.0;

    /**
     * Save the volume preference for a track.
     * 
     * Requirement 4.1: WHEN пользователь изменяет громкость для трека 
     * THEN THE User_Preference SHALL сохранить настройку громкости
     *
     * @param int $trackId The ID of the track
     * @param float $volume The volume value (0.0 to 1.0)
     * @return UserTrackPreference The created or updated preference
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     * @throws InvalidArgumentException If volume is not between 0.0 and 1.0
     */
    public function saveVolume(int $userId, int $trackId, float $volume): UserTrackPreference
    {
        // Validate volume range
        if ($volume < self::MIN_VOLUME || $volume > self::MAX_VOLUME) {
            throw new InvalidArgumentException(
                sprintf('Volume must be between %.1f and %.1f', self::MIN_VOLUME, self::MAX_VOLUME)
            );
        }

        // Verify track exists
        $track = Track::findOrFail($trackId);

        return DB::transaction(function () use ($userId, $track, $volume) {
            $preference = UserTrackPreference::updateOrCreate(
                ['user_id' => $userId, 'track_id' => $track->id],
                [
                    'volume' => $volume,
                    'updated_at' => now(),
                ]
            );

            return $preference->fresh();
        });
    }

    /**
     * Save the last playback position for a track.
     * 
     * Requirement 4.2: WHEN пользователь останавливает воспроизведение 
     * THEN THE User_Preference SHALL сохранить текущую позицию
     *
     * @param int $trackId The ID of the track
     * @param float $position The position in seconds
     * @return UserTrackPreference The created or updated preference
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     * @throws InvalidArgumentException If position is negative or exceeds track duration
     */
    public function savePosition(int $userId, int $trackId, float $position): UserTrackPreference
    {
        // Verify track exists
        $track = Track::findOrFail($trackId);

        // Validate position
        if ($position < 0) {
            throw new InvalidArgumentException('Position cannot be negative');
        }

        if ($track->duration !== null && $position > $track->duration) {
            throw new InvalidArgumentException(
                sprintf('Position (%.2f) cannot exceed track duration (%.2f)', $position, $track->duration)
            );
        }

        return DB::transaction(function () use ($userId, $track, $position) {
            $preference = UserTrackPreference::updateOrCreate(
                ['user_id' => $userId, 'track_id' => $track->id],
                [
                    'last_position' => $position,
                    'updated_at' => now(),
                ]
            );

            return $preference->fresh();
        });
    }

    /**
     * Get the preferences for a track.
     * 
     * Requirement 4.4: THE User_Preference SHALL хранить настройки в базе данных 
     * с привязкой к track_id
     *
     * @param int $trackId The ID of the track
     * @return UserTrackPreference|null The preferences or null if not set
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getPreferences(int $userId, int $trackId): ?UserTrackPreference
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return UserTrackPreference::forUser($userId)
            ->where('track_id', $trackId)
            ->first();
    }

    /**
     * Save both volume and position preferences for a track.
     * 
     * Convenience method to update multiple preferences at once.
     *
     * @param int $trackId The ID of the track
     * @param float|null $volume The volume value (0.0 to 1.0), null to skip
     * @param float|null $position The position in seconds, null to skip
     * @return UserTrackPreference The created or updated preference
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     * @throws InvalidArgumentException If volume or position is invalid
     */
    public function savePreferences(int $userId, int $trackId, ?float $volume = null, ?float $position = null): UserTrackPreference
    {
        // Verify track exists
        $track = Track::findOrFail($trackId);

        // Validate volume if provided
        if ($volume !== null && ($volume < self::MIN_VOLUME || $volume > self::MAX_VOLUME)) {
            throw new InvalidArgumentException(
                sprintf('Volume must be between %.1f and %.1f', self::MIN_VOLUME, self::MAX_VOLUME)
            );
        }

        // Validate position if provided
        if ($position !== null) {
            if ($position < 0) {
                throw new InvalidArgumentException('Position cannot be negative');
            }

            if ($track->duration !== null && $position > $track->duration) {
                throw new InvalidArgumentException(
                    sprintf('Position (%.2f) cannot exceed track duration (%.2f)', $position, $track->duration)
                );
            }
        }

        return DB::transaction(function () use ($userId, $track, $volume, $position) {
            $updateData = ['updated_at' => now()];

            if ($volume !== null) {
                $updateData['volume'] = $volume;
            }

            if ($position !== null) {
                $updateData['last_position'] = $position;
            }

            $preference = UserTrackPreference::updateOrCreate(
                ['user_id' => $userId, 'track_id' => $track->id],
                $updateData
            );

            return $preference->fresh();
        });
    }

    /**
     * Delete preferences for a track.
     * 
     * Note: This is typically handled by cascade delete when track is deleted.
     * Requirement 4.5: WHEN трек удаляется THEN THE System SHALL удалить связанные User_Preference
     *
     * @param int $trackId The ID of the track
     * @return void
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function deletePreferences(int $userId, int $trackId): void
    {
        // Verify track exists
        Track::findOrFail($trackId);

        UserTrackPreference::forUser($userId)
            ->where('track_id', $trackId)
            ->delete();
    }
}
