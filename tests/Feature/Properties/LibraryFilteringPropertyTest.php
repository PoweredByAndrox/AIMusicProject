<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: Library Filtering Correctness
 * 
 * @tag Feature: ai-music-recommendations, Property 11: Library Filtering Correctness
 * **Validates: Requirements 11.2, 11.3, 11.4**
 * 
 * For any set of filters (genre, mood, BPM range, like status),
 * all returned tracks should match all specified filters.
 */
class LibraryFilteringPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private const GENRES = ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic'];
    private const MOODS = ['Happy', 'Sad', 'Energetic', 'Calm'];

    /**
     * @test
     */
    public function genre_filter_returns_only_matching_tracks(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\elements(self::GENRES))
            ->then(function ($targetGenre) {
                // Create tracks with different genres
                foreach (self::GENRES as $genre) {
                    Track::create([
                        'title' => "Track {$genre}",
                        'duration' => 180.0,
                        'file_path' => "/storage/{$genre}.mp3",
                        'source' => 'upload',
                        'genre' => $genre,
                    ]);
                }

                // Filter by genre
                $filtered = Track::byGenre($targetGenre)->get();

                // Verify all returned tracks have the target genre
                foreach ($filtered as $track) {
                    $this->assertEquals($targetGenre, $track->genre);
                }
            });
    }

    /**
     * @test
     */
    public function mood_filter_returns_only_matching_tracks(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\elements(self::MOODS))
            ->then(function ($targetMood) {
                // Create tracks with different moods
                foreach (self::MOODS as $mood) {
                    Track::create([
                        'title' => "Track {$mood}",
                        'duration' => 180.0,
                        'file_path' => "/storage/{$mood}.mp3",
                        'source' => 'upload',
                        'mood' => $mood,
                    ]);
                }

                // Filter by mood
                $filtered = Track::byMood($targetMood)->get();

                // Verify all returned tracks have the target mood
                foreach ($filtered as $track) {
                    $this->assertEquals($targetMood, $track->mood);
                }
            });
    }

    /**
     * @test
     */
    public function bpm_range_filter_returns_tracks_within_range(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(60, 120),  // min BPM
                Generator\choose(20, 60)    // range width
            )
            ->then(function ($minBpm, $rangeWidth) {
                $maxBpm = $minBpm + $rangeWidth;

                // Create tracks with various BPMs
                $bpms = [50, 80, 100, 120, 140, 160, 180];
                foreach ($bpms as $bpm) {
                    Track::create([
                        'title' => "Track BPM {$bpm}",
                        'duration' => 180.0,
                        'file_path' => "/storage/bpm{$bpm}.mp3",
                        'source' => 'upload',
                        'bpm' => (float) $bpm,
                    ]);
                }

                // Filter by BPM range
                $filtered = Track::bpmRange($minBpm, $maxBpm)->get();

                // Verify all returned tracks are within range
                foreach ($filtered as $track) {
                    $this->assertGreaterThanOrEqual($minBpm, $track->bpm);
                    $this->assertLessThanOrEqual($maxBpm, $track->bpm);
                }
            });
    }

    /**
     * @test
     */
    public function liked_filter_returns_only_liked_tracks(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(1, 10))
            ->then(function ($numTracks) {
                // Create mix of liked and unliked tracks
                for ($i = 0; $i < $numTracks; $i++) {
                    Track::create([
                        'title' => "Track {$i}",
                        'duration' => 180.0,
                        'file_path' => "/storage/track{$i}.mp3",
                        'source' => 'upload',
                        'is_liked' => $i % 2 === 0,
                    ]);
                }

                // Filter by liked
                $filtered = Track::liked()->get();

                // Verify all returned tracks are liked
                foreach ($filtered as $track) {
                    $this->assertTrue($track->is_liked);
                }
            });
    }

    /**
     * @test
     */
    public function combined_filters_return_matching_tracks(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\elements(self::GENRES),
                Generator\elements(self::MOODS)
            )
            ->then(function ($targetGenre, $targetMood) {
                // Create tracks with various combinations
                foreach (self::GENRES as $genre) {
                    foreach (self::MOODS as $mood) {
                        Track::create([
                            'title' => "Track {$genre} {$mood}",
                            'duration' => 180.0,
                            'file_path' => "/storage/{$genre}_{$mood}.mp3",
                            'source' => 'upload',
                            'genre' => $genre,
                            'mood' => $mood,
                        ]);
                    }
                }

                // Apply combined filters
                $filtered = Track::byGenre($targetGenre)
                    ->byMood($targetMood)
                    ->get();

                // Verify all returned tracks match both filters
                foreach ($filtered as $track) {
                    $this->assertEquals($targetGenre, $track->genre);
                    $this->assertEquals($targetMood, $track->mood);
                }
            });
    }

    /**
     * @test
     */
    public function empty_filter_returns_all_tracks(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(1, 10))
            ->then(function ($numTracks) {
                // Create tracks
                for ($i = 0; $i < $numTracks; $i++) {
                    Track::create([
                        'title' => "Track {$i}",
                        'duration' => 180.0,
                        'file_path' => "/storage/track{$i}.mp3",
                        'source' => 'upload',
                    ]);
                }

                // No filters
                $all = Track::all();

                $this->assertEquals($numTracks, $all->count());
            });
    }
}
