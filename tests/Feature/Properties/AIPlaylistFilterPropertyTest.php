<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use App\Models\Playlist;
use App\Services\AIPlaylistGeneratorService;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property 8: AI-плейлист соответствует фильтрам
 * 
 * **Validates: Requirements 6.2, 6.4, 6.5**
 * 
 * Property: For any set of filter parameters (genre, mood, BPM range),
 * all tracks in the generated playlist MUST match those filters.
 */
class AIPlaylistFilterPropertyTest extends TestCase
{
    use DatabaseTransactions;

    private AIPlaylistGeneratorService $service;

    protected function setUp(): void
    {
        parent::setUp();
        $this->service = new AIPlaylistGeneratorService();
    }

    /**
     * @test
     * Property: Generated playlist tracks match genre filter
     */
    public function generated_playlist_tracks_match_genre_filter(): void
    {
        // Create tracks with different genres
        Track::factory()->create(['genre' => 'rock', 'analysis_status' => 'completed', 'duration' => 180]);
        Track::factory()->create(['genre' => 'rock', 'analysis_status' => 'completed', 'duration' => 200]);
        Track::factory()->create(['genre' => 'jazz', 'analysis_status' => 'completed', 'duration' => 220]);
        Track::factory()->create(['genre' => 'electronic', 'analysis_status' => 'completed', 'duration' => 190]);

        $playlist = $this->service->generate([
            'genre' => 'rock',
            'duration' => 10, // 10 minutes
        ]);

        $this->assertInstanceOf(Playlist::class, $playlist);
        
        foreach ($playlist->tracks as $track) {
            $this->assertEquals('rock', $track->genre, 
                "Track {$track->id} has genre '{$track->genre}' but expected 'rock'");
        }
    }

    /**
     * @test
     * Property: Generated playlist tracks match mood filter
     */
    public function generated_playlist_tracks_match_mood_filter(): void
    {
        Track::factory()->create(['mood' => 'energetic', 'analysis_status' => 'completed', 'duration' => 180]);
        Track::factory()->create(['mood' => 'energetic', 'analysis_status' => 'completed', 'duration' => 200]);
        Track::factory()->create(['mood' => 'calm', 'analysis_status' => 'completed', 'duration' => 220]);
        Track::factory()->create(['mood' => 'sad', 'analysis_status' => 'completed', 'duration' => 190]);

        $playlist = $this->service->generate([
            'mood' => 'energetic',
            'duration' => 10,
        ]);

        foreach ($playlist->tracks as $track) {
            $this->assertEquals('energetic', $track->mood,
                "Track {$track->id} has mood '{$track->mood}' but expected 'energetic'");
        }
    }

    /**
     * @test
     * Property: Generated playlist tracks match BPM range filter
     */
    public function generated_playlist_tracks_match_bpm_range_filter(): void
    {
        Track::factory()->create(['bpm' => 120, 'analysis_status' => 'completed', 'duration' => 180]);
        Track::factory()->create(['bpm' => 125, 'analysis_status' => 'completed', 'duration' => 200]);
        Track::factory()->create(['bpm' => 140, 'analysis_status' => 'completed', 'duration' => 220]);
        Track::factory()->create(['bpm' => 90, 'analysis_status' => 'completed', 'duration' => 190]);

        $playlist = $this->service->generate([
            'bpm_min' => 115,
            'bpm_max' => 130,
            'duration' => 10,
        ]);

        foreach ($playlist->tracks as $track) {
            $this->assertGreaterThanOrEqual(115, $track->bpm,
                "Track {$track->id} has BPM {$track->bpm} which is below minimum 115");
            $this->assertLessThanOrEqual(130, $track->bpm,
                "Track {$track->id} has BPM {$track->bpm} which is above maximum 130");
        }
    }

    /**
     * @test
     * Property: Generated playlist tracks match combined filters
     */
    public function generated_playlist_tracks_match_combined_filters(): void
    {
        // Matching track
        Track::factory()->create([
            'genre' => 'electronic',
            'mood' => 'energetic',
            'bpm' => 128,
            'analysis_status' => 'completed',
            'duration' => 180,
        ]);
        
        // Non-matching tracks
        Track::factory()->create([
            'genre' => 'rock',
            'mood' => 'energetic',
            'bpm' => 128,
            'analysis_status' => 'completed',
            'duration' => 200,
        ]);
        Track::factory()->create([
            'genre' => 'electronic',
            'mood' => 'calm',
            'bpm' => 128,
            'analysis_status' => 'completed',
            'duration' => 220,
        ]);
        Track::factory()->create([
            'genre' => 'electronic',
            'mood' => 'energetic',
            'bpm' => 90,
            'analysis_status' => 'completed',
            'duration' => 190,
        ]);

        $playlist = $this->service->generate([
            'genre' => 'electronic',
            'mood' => 'energetic',
            'bpm_min' => 120,
            'bpm_max' => 135,
            'duration' => 10,
        ]);

        foreach ($playlist->tracks as $track) {
            $this->assertEquals('electronic', $track->genre);
            $this->assertEquals('energetic', $track->mood);
            $this->assertGreaterThanOrEqual(120, $track->bpm);
            $this->assertLessThanOrEqual(135, $track->bpm);
        }
    }

    /**
     * @test
     * Property: Generated playlist only includes completed tracks
     */
    public function generated_playlist_only_includes_completed_tracks(): void
    {
        Track::factory()->create(['analysis_status' => 'completed', 'duration' => 180]);
        Track::factory()->create(['analysis_status' => 'pending', 'duration' => 200]);
        Track::factory()->create(['analysis_status' => 'processing', 'duration' => 220]);
        Track::factory()->create(['analysis_status' => 'failed', 'duration' => 190]);

        $playlist = $this->service->generate(['duration' => 30]);

        foreach ($playlist->tracks as $track) {
            $this->assertEquals('completed', $track->analysis_status,
                "Track {$track->id} has status '{$track->analysis_status}' but expected 'completed'");
        }
    }

    /**
     * @test
     * Property: Generated playlist is marked as AI-generated
     */
    public function generated_playlist_is_marked_as_ai_generated(): void
    {
        Track::factory()->create(['analysis_status' => 'completed', 'duration' => 180]);

        $playlist = $this->service->generate(['duration' => 10]);

        $this->assertTrue($playlist->is_ai_generated,
            "Generated playlist should be marked as AI-generated");
    }

    /**
     * @test
     * Property: Empty result when no tracks match filters
     */
    public function empty_playlist_when_no_tracks_match_filters(): void
    {
        Track::factory()->create(['genre' => 'rock', 'analysis_status' => 'completed', 'duration' => 180]);
        Track::factory()->create(['genre' => 'jazz', 'analysis_status' => 'completed', 'duration' => 200]);

        $playlist = $this->service->generate([
            'genre' => 'nonexistent_genre',
            'duration' => 10,
        ]);

        $this->assertCount(0, $playlist->tracks,
            "Playlist should be empty when no tracks match the filter");
    }
}
