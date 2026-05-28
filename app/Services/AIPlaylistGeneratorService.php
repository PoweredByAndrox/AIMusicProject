<?php

namespace App\Services;

use App\Models\Track;
use App\Models\Playlist;
use App\Models\TrackRating;
use App\Models\UserTrackScore;
use Illuminate\Support\Collection;

/**
 * AIPlaylistGeneratorService
 * 
 * Generates AI-powered playlists based on user preferences and filters.
 * Requirements: 6.1, 6.2, 6.3, 6.4
 */
class AIPlaylistGeneratorService
{
    /**
     * Generate an AI playlist based on parameters.
     *
     * @param array $params
     * @return Playlist
     */
    public function generate(int $userId, array $params): Playlist
    {
        $targetDuration = $params['duration'] ?? 60; // minutes
        $genre = $params['genre'] ?? null;
        $mood = $params['mood'] ?? null;
        $bpmMin = $params['bpm_min'] ?? null;
        $bpmMax = $params['bpm_max'] ?? null;
        $name = $params['name'] ?? 'AI Generated Playlist';
        $usePreferences = $params['use_preferences'] ?? true;

        // Build query with filters
        $query = Track::where('analysis_status', 'completed');

        if ($genre) {
            $query->where('genre', 'ilike', "%{$genre}%");
        }

        if ($mood) {
            $query->where('mood', $mood);
        }

        if ($bpmMin && $bpmMax) {
            $query->whereBetween('bpm', [$bpmMin, $bpmMax]);
        }

        // Get candidate tracks
        $candidates = $query->get();

        // Select tracks to fill target duration, using preferences if enabled
        $selectedTracks = $this->selectTracksForDuration($userId, $candidates, $targetDuration * 60, $usePreferences);

        // Create playlist
        $playlist = Playlist::create([
            'user_id' => $userId,
            'name' => $name,
            'description' => $this->generateDescription($params),
            'is_ai_generated' => true,
        ]);

        // Add tracks to playlist
        foreach ($selectedTracks as $index => $track) {
            $playlist->tracks()->attach($track->id, ['position' => $index]);
        }

        return $playlist->load('tracks');
    }

    /**
     * Select tracks to fill target duration.
     *
     * @param Collection $candidates
     * @param int $targetSeconds
     * @param bool $usePreferences
     * @return Collection
     */
    private function selectTracksForDuration(int $userId, Collection $candidates, int $targetSeconds, bool $usePreferences = true): Collection
    {
        if ($candidates->isEmpty()) {
            return collect();
        }

        // Calculate preference scores for each track
        if ($usePreferences) {
            $scores = $this->calculatePreferenceScores($userId, $candidates->pluck('id')->toArray());
            
            // Sort by preference score (highest first)
            $sorted = $candidates->sortByDesc(function ($track) use ($scores) {
                return $scores[$track->id] ?? 0.5;
            });
        } else {
            $sorted = $candidates->shuffle();
        }

        $selected = collect();
        $totalDuration = 0;

        foreach ($sorted as $track) {
            if ($totalDuration >= $targetSeconds) {
                break;
            }

            $selected->push($track);
            $totalDuration += $track->duration;
        }

        return $selected;
    }
    
    /**
     * Calculate preference scores for tracks based on user's liked tracks.
     *
     * @param array $trackIds
     * @return array
     */
    private function calculatePreferenceScores(int $userId, array $trackIds): array
    {
        $scores = [];
        $directScores = UserTrackScore::forUser($userId)
            ->whereIn('track_id', $trackIds)
            ->pluck('preference_score', 'track_id')
            ->toArray();

        $highRatedTrackIds = TrackRating::forUser($userId)
            ->where('rating', '>=', 4)
            ->pluck('track_id')
            ->toArray();
        
        // Get user's liked tracks to build preference profile
        $likedTracks = Track::likedByUser($userId)
            ->where('analysis_status', 'completed')
            ->get();
        
        if ($likedTracks->isEmpty()) {
            if (!empty($directScores)) {
                foreach ($trackIds as $trackId) {
                    if (array_key_exists($trackId, $directScores)) {
                        $scores[$trackId] = (float) $directScores[$trackId];
                        continue;
                    }

                    $scores[$trackId] = 0.5 + (crc32($trackId . date('Y-m-d')) % 100) / 1000;
                }

                return $scores;
            }

            // No likes yet - return neutral scores with slight randomization
            foreach ($trackIds as $trackId) {
                $scores[$trackId] = 0.5 + (crc32($trackId . date('Y-m-d')) % 100) / 1000;
            }
            return $scores;
        }
        
        // Build preference profile from liked tracks
        $preferredGenres = $likedTracks->pluck('genre')->filter()->countBy();
        $preferredMoods = $likedTracks->pluck('mood')->filter()->countBy();
        $avgBpm = $likedTracks->avg('bpm') ?: 140;
        $preferredKeys = $likedTracks->pluck('key')->filter()->countBy();
        
        // Score each candidate track based on similarity to preferences
        foreach ($trackIds as $trackId) {
            if (array_key_exists($trackId, $directScores)) {
                $scores[$trackId] = (float) $directScores[$trackId];
                continue;
            }

            $track = Track::find($trackId);
            if (!$track) {
                $scores[$trackId] = 0.3;
                continue;
            }
            
            $score = 0.3; // Base score for analyzed tracks
            
            // Genre match (most important - up to 0.35)
            if ($track->genre && $preferredGenres->isNotEmpty()) {
                $genreScore = 0;
                foreach ($preferredGenres as $genre => $count) {
                    if ($track->genre === $genre) {
                        $genreScore = max($genreScore, 0.35);
                    } elseif (str_contains($track->genre, $genre) || str_contains($genre, $track->genre)) {
                        $genreScore = max($genreScore, 0.25);
                    } elseif ($this->areGenresRelated($track->genre, $genre)) {
                        $genreScore = max($genreScore, 0.15);
                    }
                }
                $score += $genreScore;
            }
            
            // BPM similarity (up to 0.15)
            if ($track->bpm && $avgBpm) {
                $bpmDiff = abs($track->bpm - $avgBpm);
                if ($bpmDiff < 10) {
                    $score += 0.15;
                } elseif ($bpmDiff < 20) {
                    $score += 0.10;
                } elseif ($bpmDiff < 40) {
                    $score += 0.05;
                }
            }
            
            // Mood match (up to 0.10)
            if ($track->mood && $preferredMoods->has($track->mood)) {
                $score += 0.10;
            }

            if (in_array($track->id, $highRatedTrackIds, true)) {
                $score += 0.20;
            }
            
            // Key match (up to 0.05)
            if ($track->key && $preferredKeys->has($track->key)) {
                $score += 0.05;
            }
            
            // Small random factor for variety (0-0.05)
            $score += (crc32($trackId . date('Y-m-d')) % 50) / 1000;
            
            $scores[$trackId] = min(0.99, $score);
        }
        
        return $scores;
    }
    
    /**
     * Check if two genres are related.
     */
    private function areGenresRelated(string $genre1, string $genre2): bool
    {
        $bassGenres = ['Dubstep', 'Drum & Bass', 'Jungle', 'Grime', 'UK Garage', 'Halftime', 'Breakbeat'];
        $houseGenres = ['House', 'Tech House', 'Deep House', 'Progressive House', 'Electro House'];
        $tranceGenres = ['Trance', 'Progressive Trance', 'Psytrance', 'Uplifting Trance'];
        
        $groups = [$bassGenres, $houseGenres, $tranceGenres];
        
        foreach ($groups as $group) {
            $inGroup1 = false;
            $inGroup2 = false;
            foreach ($group as $g) {
                if (str_contains($genre1, $g) || str_contains($g, $genre1)) $inGroup1 = true;
                if (str_contains($genre2, $g) || str_contains($g, $genre2)) $inGroup2 = true;
            }
            if ($inGroup1 && $inGroup2) return true;
        }
        
        return false;
    }

    /**
     * Generate playlist description.
     *
     * @param array $params
     * @return string
     */
    private function generateDescription(array $params): string
    {
        $parts = ['AI-generated playlist'];

        if (!empty($params['genre'])) {
            $parts[] = "Genre: {$params['genre']}";
        }

        if (!empty($params['mood'])) {
            $parts[] = "Mood: {$params['mood']}";
        }

        if (!empty($params['bpm_min']) && !empty($params['bpm_max'])) {
            $parts[] = "BPM: {$params['bpm_min']}-{$params['bpm_max']}";
        }

        if (!empty($params['duration'])) {
            $parts[] = "Target: {$params['duration']} min";
        }

        return implode(' | ', $parts);
    }
}
