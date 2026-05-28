<?php

namespace App\Http\Controllers;

use App\Models\Track;
use Illuminate\Http\JsonResponse;

/**
 * AIDJController
 * 
 * Handles AI DJ features: track compatibility based on BPM, key, genre.
 * Uses regular track analysis data - no separate DJ analysis needed!
 */
class AIDJController extends Controller
{
    /**
     * Get compatible tracks for DJ mixing.
     * Uses regular track analysis data (BPM, key, genre, mood).
     *
     * @param int $trackId
     * @return JsonResponse
     */
    public function compatible(int $trackId): JsonResponse
    {
        $track = Track::find($trackId);

        if (!$track) {
            return response()->json(['error' => 'Track not found'], 404);
        }

        if ($track->analysis_status !== 'completed') {
            return response()->json(['error' => 'Track not analyzed yet'], 400);
        }

        $sourceBpm = $track->bpm ?? 140;
        $sourceKey = $track->key;
        $sourceGenre = $track->genre;
        $sourceMood = $track->mood;
        $sourceCamelot = $this->keyToCamelot($sourceKey);

        // Find all analyzed tracks except current
        $candidates = Track::where('analysis_status', 'completed')
            ->where('id', '!=', $trackId)
            ->get();

        // Calculate compatibility for each track
        $compatibleTracks = $candidates
            ->map(function ($candidate) use ($sourceBpm, $sourceGenre, $sourceMood, $sourceCamelot) {
                $candBpm = $candidate->bpm ?? 140;
                $candCamelot = $this->keyToCamelot($candidate->key);
                
                $bpmScore = $this->calcBpmScore($sourceBpm, $candBpm);
                $keyScore = $this->calcKeyScore($sourceCamelot, $candCamelot);
                $genreScore = $this->calcGenreScore($sourceGenre, $candidate->genre);
                $moodScore = $sourceMood && $candidate->mood === $sourceMood ? 1.0 : 0.5;
                
                // BPM most important for mixing, then key, then genre
                $totalScore = 0.4 * $bpmScore + 0.35 * $keyScore + 0.15 * $genreScore + 0.1 * $moodScore;

                return [
                    'track' => $candidate,
                    'dj_analysis' => [
                        'camelot_key' => $candCamelot,
                        'bpm' => $candBpm,
                    ],
                    'compatibility' => [
                        'total_score' => round($totalScore, 3),
                        'key_score' => round($keyScore, 3),
                        'bpm_score' => round($bpmScore, 3),
                        'bpm_diff_percent' => $sourceBpm > 0 ? round(abs($sourceBpm - $candBpm) / $sourceBpm * 100, 1) : 0,
                    ],
                ];
            })
            ->filter(fn($item) => $item['compatibility']['total_score'] >= 0.4)
            ->sortByDesc('compatibility.total_score')
            ->take(20)
            ->values();

        return response()->json([
            'source_track' => [
                'id' => $track->id,
                'title' => $track->title,
                'artist' => $track->artist,
                'camelot_key' => $sourceCamelot,
                'bpm' => $sourceBpm,
            ],
            'compatible_tracks' => $compatibleTracks,
            'count' => $compatibleTracks->count(),
        ]);
    }
    
    private function calcBpmScore(float $src, float $cand): float
    {
        if ($src <= 0 || $cand <= 0) return 0.5;
        
        $diff = abs($src - $cand) / $src;
        $halfDiff = abs($src - $cand * 2) / $src;
        $doubleDiff = abs($src - $cand / 2) / $src;
        $best = min($diff, $halfDiff, $doubleDiff);
        
        if ($best < 0.02) return 1.0;
        if ($best < 0.05) return 0.9;
        if ($best < 0.08) return 0.75;
        if ($best < 0.12) return 0.5;
        return 0.2;
    }
    
    private function calcKeyScore(string $src, string $cand): float
    {
        if ($src === $cand) return 1.0;
        
        if (!preg_match('/^(\d+)([AB])$/', $src, $s)) return 0.5;
        if (!preg_match('/^(\d+)([AB])$/', $cand, $c)) return 0.5;
        
        $sNum = (int)$s[1]; $sLet = $s[2];
        $cNum = (int)$c[1]; $cLet = $c[2];
        
        // Relative major/minor
        if ($sNum === $cNum && $sLet !== $cLet) return 0.9;
        
        // Adjacent on wheel
        if ($sLet === $cLet) {
            $d = abs($sNum - $cNum);
            if ($d === 1 || $d === 11) return 0.85;
        }
        
        $d = min(abs($sNum - $cNum), 12 - abs($sNum - $cNum));
        if ($d <= 2) return 0.6;
        
        return 0.3;
    }
    
    private function calcGenreScore(?string $src, ?string $cand): float
    {
        if (!$src || !$cand) return 0.5;
        if ($src === $cand) return 1.0;
        if (str_contains($src, $cand) || str_contains($cand, $src)) return 0.8;
        
        $groups = [
            ['Dubstep', 'Drum & Bass', 'Jungle', 'Grime', 'UK Garage', 'Breakbeat'],
            ['House', 'Tech House', 'Deep House', 'Progressive House', 'Electro'],
            ['Trance', 'Progressive Trance', 'Psytrance'],
        ];
        
        foreach ($groups as $g) {
            $sIn = $cIn = false;
            foreach ($g as $genre) {
                if (str_contains($src, $genre)) $sIn = true;
                if (str_contains($cand, $genre)) $cIn = true;
            }
            if ($sIn && $cIn) return 0.7;
        }
        
        return 0.4;
    }
    
    private function keyToCamelot(?string $key): string
    {
        $map = [
            'C' => '8B', 'Am' => '8A', 'G' => '9B', 'Em' => '9A',
            'D' => '10B', 'Bm' => '10A', 'A' => '11B', 'F#m' => '11A',
            'E' => '12B', 'C#m' => '12A', 'B' => '1B', 'G#m' => '1A',
            'F#' => '2B', 'D#m' => '2A', 'Db' => '3B', 'Bbm' => '3A',
            'Ab' => '4B', 'Fm' => '4A', 'Eb' => '5B', 'Cm' => '5A',
            'Bb' => '6B', 'Gm' => '6A', 'F' => '7B', 'Dm' => '7A',
        ];
        return $map[$key] ?? '8B';
    }
}
