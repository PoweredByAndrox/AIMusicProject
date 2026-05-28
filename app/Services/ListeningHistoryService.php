<?php

namespace App\Services;

use App\Models\ListeningHistory;
use App\Models\Track;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Auth;

/**
 * ListeningHistoryService
 * 
 * Handles listening history recording and retrieval.
 * Requirements: 16.3, 16.4
 */
class ListeningHistoryService
{
    /**
     * Number of days to retain listening history.
     * Requirement 16.4: THE System SHALL хранить Listening_History за последние 30 дней
     */
    private const RETENTION_DAYS = 30;

    /**
     * Record the start of a track playback.
     * 
     * Requirement 16.3: THE Listening_History SHALL записывать: track_id, timestamp, 
     * duration_played, completed
     *
     * @param int $trackId The ID of the track being played
     * @param string|null $source The source of playback ('queue', 'playlist', 'radio', 'ai_dj')
     * @return ListeningHistory The created listening history entry
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function recordPlay(int $trackId, ?string $source = null): ListeningHistory
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return ListeningHistory::create([
            'user_id' => Auth::id(),
            'track_id' => $trackId,
            'started_at' => now(),
            'duration_played' => 0,
            'completed' => false,
            'source' => $source,
        ]);
    }

    /**
     * Record the completion or stop of a track playback.
     * 
     * Requirement 16.3: THE Listening_History SHALL записывать: track_id, timestamp, 
     * duration_played, completed
     *
     * @param int $historyId The ID of the listening history entry
     * @param float $durationPlayed The duration played in seconds
     * @param bool $completed Whether the track was played to completion
     * @return ListeningHistory The updated listening history entry
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If history entry not found
     */
    public function recordComplete(int $historyId, float $durationPlayed, bool $completed = false): ListeningHistory
    {
        $history = ListeningHistory::forUser(Auth::id())->findOrFail($historyId);

        $history->update([
            'duration_played' => $durationPlayed,
            'completed' => $completed,
        ]);
        
        // Trigger auto training after completed play (counts as positive feedback)
        // NOTE: Auto-training disabled — user triggers training manually via UI button
        // if ($completed) {
        //     PreferenceTrainingService::triggerTraining();
        // }

        return $history->fresh();
    }

    /**
     * Get recent listening history within specified hours.
     * 
     * Requirement 16.1: THE Smart_Radio SHALL исключать треки, воспроизведённые за последние 2 часа
     * Requirement 16.2: THE Smart_Radio SHALL снижать приоритет треков, воспроизведённых за последние 24 часа
     *
     * @param int $hours Number of hours to look back (default 24)
     * @return Collection<int, ListeningHistory> Collection of listening history entries
     */
    public function getRecentHistory(int $hours = 24): Collection
    {
        return ListeningHistory::forUser(Auth::id())
            ->where('started_at', '>=', now()->subHours($hours))
            ->with('track')
            ->orderBy('started_at', 'desc')
            ->get();
    }

    /**
     * Get listening history for a specific track.
     *
     * @param int $trackId The ID of the track
     * @return Collection<int, ListeningHistory> Collection of listening history entries
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getHistoryForTrack(int $trackId): Collection
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return ListeningHistory::forUser(Auth::id())
            ->where('track_id', $trackId)
            ->orderBy('started_at', 'desc')
            ->get();
    }

    /**
     * Get track IDs played within specified hours.
     * Useful for Smart Radio exclusion logic.
     *
     * @param int $hours Number of hours to look back
     * @return array<int> Array of track IDs
     */
    public function getRecentlyPlayedTrackIds(int $hours, ?int $userId = null): array
    {
        return ListeningHistory::forUser($userId ?? Auth::id())
            ->where('started_at', '>=', now()->subHours($hours))
            ->distinct()
            ->pluck('track_id')
            ->toArray();
    }

    /**
     * Clean up old listening history entries.
     * 
     * Requirement 16.4: THE System SHALL хранить Listening_History за последние 30 дней
     *
     * @return int Number of deleted entries
     */
    public function cleanupOldHistory(): int
    {
        return ListeningHistory::forUser(Auth::id())
            ->where('started_at', '<', now()->subDays(self::RETENTION_DAYS))
            ->delete();
    }

    /**
     * Check if a track was played recently.
     *
     * @param int $trackId The ID of the track
     * @param int $hours Number of hours to check
     * @return bool True if track was played within the specified hours
     */
    public function wasPlayedRecently(int $trackId, int $hours): bool
    {
        return ListeningHistory::forUser(Auth::id())
            ->where('track_id', $trackId)
            ->where('started_at', '>=', now()->subHours($hours))
            ->exists();
    }
}
