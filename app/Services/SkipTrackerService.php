<?php

namespace App\Services;

use App\Models\Track;
use App\Models\TrackSkip;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Auth;

/**
 * SkipTrackerService
 * 
 * Handles skip event recording and statistics.
 * Requirements: 18.1, 18.2, 18.3
 */
class SkipTrackerService
{
    /**
     * Threshold in seconds for considering a skip as "early".
     * Requirement 18.1: WHEN пользователь пропускает трек в первые 30 секунд
     */
    private const EARLY_SKIP_THRESHOLD = 30;

    /**
     * Number of skips after which a track should be significantly deprioritized.
     * Requirement 18.3: WHEN трек пропущен 3+ раза THEN THE Smart_Radio SHALL 
     * значительно снизить его приоритет
     */
    private const SKIP_PENALTY_THRESHOLD = 3;

    /**
     * Record a skip event for a track.
     * 
     * Requirement 18.1: WHEN пользователь пропускает трек в первые 30 секунд 
     * THEN THE Skip_Tracker SHALL записать событие пропуска
     * 
     * Requirement 18.2: THE Skip_Tracker SHALL хранить: track_id, timestamp, 
     * position_at_skip, source
     *
     * @param int $trackId The ID of the track being skipped
     * @param float $positionAtSkip The playback position when skip occurred (in seconds)
     * @param string|null $source The source of playback ('queue', 'playlist', 'radio', 'ai_dj')
     * @return TrackSkip The created skip record
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function recordSkip(int $trackId, float $positionAtSkip, ?string $source = null): TrackSkip
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return TrackSkip::create([
            'user_id' => Auth::id(),
            'track_id' => $trackId,
            'position_at_skip' => $positionAtSkip,
            'source' => $source,
            'skipped_at' => now(),
        ]);
    }

    /**
     * Get the total skip count for a track.
     *
     * @param int $trackId The ID of the track
     * @return int Total number of skips
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getSkipCount(int $trackId): int
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return TrackSkip::where('track_id', $trackId)->count();
    }

    /**
     * Get the early skip count for a track (skips within first 30 seconds).
     * 
     * Requirement 18.1: Skips in first 30 seconds are tracked
     * Requirement 18.3: 3+ skips triggers penalty
     *
     * @param int $trackId The ID of the track
     * @return int Number of early skips
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getEarlySkipCount(int $trackId): int
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return TrackSkip::where('track_id', $trackId)
            ->earlySkips()
            ->count();
    }

    /**
     * Get skip statistics for a track.
     * 
     * Requirement 18.5: THE System SHALL отображать статистику пропусков 
     * в информации о треке
     *
     * @param int $trackId The ID of the track
     * @return array{total_skips: int, early_skips: int, should_penalize: bool, average_skip_position: float|null, sources: array}
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getSkipStats(int $trackId): array
    {
        // Verify track exists
        Track::findOrFail($trackId);

        $allSkips = TrackSkip::where('track_id', $trackId)->get();
        $earlySkips = $allSkips->filter(fn($skip) => $skip->position_at_skip < self::EARLY_SKIP_THRESHOLD);

        $totalSkips = $allSkips->count();
        $earlySkipCount = $earlySkips->count();

        // Calculate average skip position
        $averagePosition = $totalSkips > 0 
            ? $allSkips->avg('position_at_skip') 
            : null;

        // Count skips by source
        $sources = $allSkips->groupBy('source')
            ->map(fn($group) => $group->count())
            ->toArray();

        return [
            'total_skips' => $totalSkips,
            'early_skips' => $earlySkipCount,
            'should_penalize' => $earlySkipCount >= self::SKIP_PENALTY_THRESHOLD,
            'average_skip_position' => $averagePosition !== null ? round($averagePosition, 2) : null,
            'sources' => $sources,
        ];
    }

    /**
     * Check if a track should be penalized in recommendations.
     * 
     * Requirement 18.3: WHEN трек пропущен 3+ раза THEN THE Smart_Radio SHALL 
     * значительно снизить его приоритет
     *
     * @param int $trackId The ID of the track
     * @return bool True if track should be penalized
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function shouldPenalize(int $trackId): bool
    {
        return $this->getEarlySkipCount($trackId) >= self::SKIP_PENALTY_THRESHOLD;
    }

    /**
     * Get all tracks that should be penalized (3+ early skips).
     * 
     * Requirement 18.3: Tracks with 3+ skips should be deprioritized
     *
     * @return array<int> Array of track IDs that should be penalized
     */
    public function getPenalizedTrackIds(): array
    {
        return TrackSkip::select('track_id')
            ->where('position_at_skip', '<', self::EARLY_SKIP_THRESHOLD)
            ->groupBy('track_id')
            ->havingRaw('COUNT(*) >= ?', [self::SKIP_PENALTY_THRESHOLD])
            ->pluck('track_id')
            ->toArray();
    }

    /**
     * Get all skip records for a track.
     *
     * @param int $trackId The ID of the track
     * @return Collection<int, TrackSkip> Collection of skip records
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getSkipsForTrack(int $trackId): Collection
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return TrackSkip::where('track_id', $trackId)
            ->orderBy('skipped_at', 'desc')
            ->get();
    }

    /**
     * Get the skip penalty threshold constant.
     *
     * @return int The number of skips required for penalty
     */
    public static function getSkipPenaltyThreshold(): int
    {
        return self::SKIP_PENALTY_THRESHOLD;
    }

    /**
     * Get the early skip threshold constant.
     *
     * @return int The number of seconds for early skip
     */
    public static function getEarlySkipThreshold(): int
    {
        return self::EARLY_SKIP_THRESHOLD;
    }
}
