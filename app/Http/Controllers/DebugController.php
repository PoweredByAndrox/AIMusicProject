<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Models\Fragment;
use App\Models\TrackRating;
use App\Models\ListeningHistory;
use App\Models\PreferenceModelStatus;
use App\Models\QueueTask;
use App\Services\PreferenceTrainingService;
use App\Support\AppStoragePath;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Redis;

/**
 * DebugController
 * 
 * Debug endpoints for development - shows training data and system stats.
 */
class DebugController extends Controller
{
    /**
     * Persist frontend console/error logs to a dedicated file for debugging.
     */
    public function frontendLogs(Request $request): JsonResponse
    {
        $request->validate([
            'logs' => ['required', 'array', 'min:1', 'max:50'],
        ]);

        $rawLogs = $request->input('logs', []);
        $logPath = AppStoragePath::log('frontend.log');
        $lines = '';

        foreach ($rawLogs as $rawLog) {
            if (!is_array($rawLog)) {
                continue;
            }

            $context = $rawLog['context'] ?? [];
            if (!is_array($context)) {
                $context = ['value' => $context];
            }

            $record = [
                'server_timestamp' => now()->toIso8601String(),
                'client_timestamp' => $this->truncateString((string) ($rawLog['timestamp'] ?? ''), 128),
                'user_id' => $request->user()?->id,
                'user_role' => $request->user()?->role,
                'level' => $this->truncateString((string) ($rawLog['level'] ?? 'log'), 16),
                'message' => $this->truncateString((string) ($rawLog['message'] ?? ''), 4000),
                'url' => $this->truncateString((string) ($rawLog['url'] ?? ''), 2048),
                'route' => $this->truncateString((string) ($rawLog['route'] ?? ''), 1024),
                'ip' => $request->ip(),
                'user_agent' => $this->truncateString((string) ($request->userAgent() ?? ''), 1024),
                'context' => $context,
            ];

            $lines .= json_encode($record, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) . PHP_EOL;
        }

        if ($lines !== '') {
            $this->appendFrontendLogLines($logPath, $lines);
        }

        return response()->json([
            'written' => count($rawLogs),
            'path' => $logPath,
        ]);
    }

    /**
     * Get detailed stats for all tracks with training signals.
     */
    public function trackStats(): JsonResponse
    {
        // Get all tracks with their signals
        $tracks = Track::where('analysis_status', 'completed')
            ->orderBy('id', 'desc')
            ->get();
        
        // Get play counts
        $playCounts = ListeningHistory::select('track_id')
            ->selectRaw('COUNT(*) as play_count')
            ->selectRaw('SUM(CASE WHEN completed THEN 1 ELSE 0 END) as completed_plays')
            ->selectRaw('MAX(started_at) as last_played')
            ->groupBy('track_id')
            ->get()
            ->keyBy('track_id');
        
        // Get ratings
        $ratings = TrackRating::all()->keyBy('track_id');
        
        // Build stats for each track
        $trackStats = $tracks->map(function ($track) use ($playCounts, $ratings) {
            $playData = $playCounts->get($track->id);
            $rating = $ratings->get($track->id);
            
            // Calculate preference score (same logic as AIPlaylistGeneratorService)
            $score = $this->calculatePreferenceScore($track, $playData, $rating);
            
            return [
                'id' => $track->id,
                'title' => $track->title,
                'artist' => $track->artist,
                'genre' => $track->genre,
                'bpm' => $track->bpm,
                'key' => $track->key,
                'mood' => $track->mood,
                'signals' => [
                    'is_liked' => $track->is_liked,
                    'rating' => $rating?->rating,
                    'play_count' => $playData?->play_count ?? 0,
                    'completed_plays' => $playData?->completed_plays ?? 0,
                    'last_played' => $playData?->last_played,
                ],
                'preference_score' => round($score, 4),
                'is_positive' => $track->is_liked || ($rating?->rating >= 4) || ($playData?->play_count >= 2),
            ];
        });
        
        return response()->json([
            'tracks' => $trackStats,
            'total' => $trackStats->count(),
            'positive_count' => $trackStats->where('is_positive', true)->count(),
        ]);
    }
    
    /**
     * Get system-wide training stats.
     */
    public function systemStats(): JsonResponse
    {
        $stats = PreferenceTrainingService::getStats();
        $modelStatus = PreferenceModelStatus::first();
        
        // Queue stats
        $queueStats = [
            'pending_tasks' => QueueTask::where('status', 'pending')->count(),
            'processing_tasks' => QueueTask::where('status', 'processing')->count(),
            'completed_tasks' => QueueTask::where('status', 'completed')->count(),
            'failed_tasks' => QueueTask::where('status', 'failed')->count(),
            'training_tasks' => QueueTask::where('task_type', 'model_training')->count(),
        ];
        
        // Redis queue length
        $redisQueueLength = 0;
        try {
            $redisQueueLength = Redis::connection()->llen('ml_worker:tasks');
        } catch (\Exception $e) {
            // Ignore
        }
        
        // Track analysis stats
        $analysisStats = [
            'total_tracks' => Track::count(),
            'analyzed' => Track::where('analysis_status', 'completed')->count(),
            'pending' => Track::where('analysis_status', 'pending')->count(),
            'processing' => Track::where('analysis_status', 'processing')->count(),
            'failed' => Track::where('analysis_status', 'failed')->count(),
        ];
        
        // Recent activity
        $recentPlays = ListeningHistory::orderBy('started_at', 'desc')
            ->take(10)
            ->with('track:id,title,artist')
            ->get()
            ->map(fn($h) => [
                'track' => $h->track?->title ?? 'Unknown',
                'artist' => $h->track?->artist ?? '',
                'started_at' => $h->started_at,
                'completed' => $h->completed,
                'duration_played' => $h->duration_played,
            ]);
        
        return response()->json([
            'preference_stats' => $stats,
            'model_status' => [
                'status' => $modelStatus?->status ?? 'not_trained',
                'trained_at' => $modelStatus?->trained_at,
                'likes_count' => $modelStatus?->likes_count ?? 0,
            ],
            'queue_stats' => $queueStats,
            'redis_queue_length' => $redisQueueLength,
            'analysis_stats' => $analysisStats,
            'recent_plays' => $recentPlays,
        ]);
    }
    
    /**
     * Get detailed info for a single track.
     */
    public function trackDetail(int $trackId): JsonResponse
    {
        $track = Track::find($trackId);
        if (!$track) {
            return response()->json(['error' => 'Track not found'], 404);
        }
        
        $rating = TrackRating::where('track_id', $trackId)->first();
        $playHistory = ListeningHistory::where('track_id', $trackId)
            ->orderBy('started_at', 'desc')
            ->take(20)
            ->get();
        
        $fragments = Fragment::where('track_id', $trackId)->get();
        
        return response()->json([
            'track' => $track,
            'rating' => $rating,
            'play_history' => $playHistory,
            'fragments' => $fragments,
            'stats' => [
                'total_plays' => $playHistory->count(),
                'completed_plays' => $playHistory->where('completed', true)->count(),
                'total_duration_played' => $playHistory->sum('duration_played'),
                'liked_fragments' => $fragments->where('is_liked', true)->count(),
            ],
        ]);
    }
    
    /**
     * Calculate preference score for a track.
     */
    private function calculatePreferenceScore(Track $track, $playData, $rating): float
    {
        $score = 0.3; // Base score
        
        // Like bonus (+0.3)
        if ($track->is_liked) {
            $score += 0.3;
        }
        
        // Rating bonus (4+ stars = +0.2, 5 stars = +0.3)
        if ($rating) {
            if ($rating->rating >= 5) {
                $score += 0.3;
            } elseif ($rating->rating >= 4) {
                $score += 0.2;
            } elseif ($rating->rating >= 3) {
                $score += 0.1;
            }
        }
        
        // Play count bonus
        if ($playData) {
            $plays = $playData->play_count;
            if ($plays >= 10) {
                $score += 0.2;
            } elseif ($plays >= 5) {
                $score += 0.15;
            } elseif ($plays >= 2) {
                $score += 0.1;
            }
            
            // Completed plays bonus
            $completed = $playData->completed_plays;
            if ($completed >= 5) {
                $score += 0.1;
            } elseif ($completed >= 2) {
                $score += 0.05;
            }
        }
        
        return min(1.0, $score);
    }

    private function truncateString(string $value, int $limit): string
    {
        if (mb_strlen($value) <= $limit) {
            return $value;
        }

        return mb_substr($value, 0, $limit) . '...';
    }

    private function appendFrontendLogLines(string $logPath, string $lines): void
    {
        $logDirectory = dirname($logPath);

        if (!is_dir($logDirectory) && !mkdir($logDirectory, 0775, true) && !is_dir($logDirectory)) {
            throw new \RuntimeException("Unable to create frontend log directory: {$logDirectory}");
        }

        if (file_exists($logPath) && !is_writable($logPath)) {
            $orphanedPath = sprintf('%s.orphaned-%s', $logPath, now()->format('YmdHis'));

            if (!rename($logPath, $orphanedPath)) {
                throw new \RuntimeException("Unable to rotate unwritable frontend log file: {$logPath}");
            }
        }

        if (file_put_contents($logPath, $lines, FILE_APPEND | LOCK_EX) === false) {
            throw new \RuntimeException("Unable to write frontend logs to {$logPath}");
        }
    }
}
