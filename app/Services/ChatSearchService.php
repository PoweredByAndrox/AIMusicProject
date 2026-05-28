<?php

namespace App\Services;

use App\Models\Track;
use Illuminate\Support\Facades\Redis;

/**
 * ChatSearchService - Service for searching similar audio segments.
 * 
 * Sends segment search tasks to ML Worker and retrieves results.
 * Requirements: 3.1, 3.2, 3.3
 */
class ChatSearchService
{
    /**
     * Queue name for ML worker tasks.
     */
    private const ML_QUEUE = 'ml_worker:tasks';
    
    /**
     * Result queue for receiving search results.
     */
    private const RESULT_QUEUE = 'ml_worker:results';
    
    /**
     * Timeout for waiting for results (seconds).
     */
    private const RESULT_TIMEOUT = 60;

    /**
     * Search for similar segments across the library.
     *
     * @param int $trackId Source track ID
     * @param float $startTime Start time in seconds
     * @param float $endTime End time in seconds
     * @param int $limit Maximum number of results
     * @param float $minSimilarity Minimum similarity threshold (0.0-1.0)
     * @return array Search results
     * @throws \Exception If track not found or search fails
     */
    public function searchSimilarSegments(
        int $trackId,
        float $startTime,
        float $endTime,
        int $limit = 10,
        float $minSimilarity = 0.3
    ): array {
        // Verify track exists
        $track = Track::find($trackId);
        if (!$track) {
            throw new \Exception("Track not found: {$trackId}");
        }
        
        // Validate time range
        if ($startTime < 0 || $endTime < 0) {
            throw new \InvalidArgumentException('Time values cannot be negative');
        }
        
        if ($startTime >= $endTime) {
            throw new \InvalidArgumentException('Start time must be less than end time');
        }
        
        // Clamp to track duration
        if ($track->duration && $endTime > $track->duration) {
            $endTime = $track->duration;
        }
        
        // Generate unique task ID for result correlation
        $taskId = uniqid('chat_search_', true);
        
        // Push task to ML worker queue
        $task = [
            'type' => 'segment_search',
            'task_id' => $taskId,
            'track_id' => $trackId,
            'start_time' => $startTime,
            'end_time' => $endTime,
            'limit' => $limit,
            'min_similarity' => $minSimilarity,
            'created_at' => now()->toIso8601String(),
        ];
        
        Redis::connection()->rpush(self::ML_QUEUE, json_encode($task));
        
        // Wait for result (with timeout)
        $resultKey = "chat_search_result:{$taskId}";
        $startWait = time();
        
        while (time() - $startWait < self::RESULT_TIMEOUT) {
            $result = Redis::connection()->get($resultKey);
            
            if ($result) {
                // Clean up result key
                Redis::connection()->del($resultKey);
                return json_decode($result, true);
            }
            
            // Small delay before next check
            usleep(100000); // 100ms
        }
        
        // Timeout - return empty results
        return [
            'query' => [
                'track_id' => $trackId,
                'start_time' => $startTime,
                'end_time' => $endTime,
            ],
            'results' => [],
            'total' => 0,
            'error' => 'Search timeout - ML worker may be busy'
        ];
    }
    
    /**
     * Get track info for display in results.
     *
     * @param int $trackId Track ID
     * @return array|null Track info or null if not found
     */
    public function getTrackInfo(int $trackId): ?array
    {
        $track = Track::find($trackId);
        
        if (!$track) {
            return null;
        }
        
        return [
            'id' => $track->id,
            'title' => $track->title,
            'artist' => $track->artist,
            'duration' => $track->duration,
            'file_path' => $track->file_path,
        ];
    }
}
