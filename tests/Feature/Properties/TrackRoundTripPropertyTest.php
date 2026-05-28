<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: Track Round-Trip Consistency
 * 
 * @tag Feature: ai-music-recommendations, Property 1: Track Round-Trip Consistency
 * **Validates: Requirements 4.1, 4.2**
 * 
 * For any valid track with metadata and ML characteristics,
 * saving to database and subsequent reading should return an equivalent object.
 */
class TrackRoundTripPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    /**
     * @test
     */
    public function track_round_trip_preserves_all_fields(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\string(),           // title
                Generator\string(),           // artist
                Generator\string(),           // album
                Generator\choose(1900, 2030), // year
                Generator\float(),            // duration
                Generator\string(),           // file_path
                Generator\elements(['upload', 'youtube']), // source
            )
            ->then(function ($title, $artist, $album, $year, $duration, $filePath, $source) {
                // Ensure non-empty title and file_path (required fields)
                $title = empty($title) ? 'Test Track' : substr($title, 0, 255);
                $filePath = empty($filePath) ? '/storage/test.mp3' : substr($filePath, 0, 500);
                $duration = abs($duration) + 0.1; // Ensure positive duration
                
                $originalData = [
                    'title' => $title,
                    'artist' => substr($artist, 0, 255) ?: null,
                    'album' => substr($album, 0, 255) ?: null,
                    'year' => $year,
                    'duration' => $duration,
                    'file_path' => $filePath,
                    'source' => $source,
                ];

                // Save to database
                $track = Track::create($originalData);
                
                // Retrieve from database
                $retrieved = Track::find($track->id);
                
                // Assert round-trip consistency
                $this->assertNotNull($retrieved);
                $this->assertEquals($originalData['title'], $retrieved->title);
                $this->assertEquals($originalData['artist'], $retrieved->artist);
                $this->assertEquals($originalData['album'], $retrieved->album);
                $this->assertEquals($originalData['year'], $retrieved->year);
                $this->assertEqualsWithDelta($originalData['duration'], $retrieved->duration, 0.0001);
                $this->assertEquals($originalData['file_path'], $retrieved->file_path);
                $this->assertEquals($originalData['source'], $retrieved->source);
            });
    }

    /**
     * @test
     */
    public function track_ml_characteristics_round_trip(): void
    {
        $genres = ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic', 'Hip-Hop', 'Metal', 'Folk'];
        $moods = ['Happy', 'Sad', 'Aggressive', 'Calm', 'Energetic', 'Melancholic'];
        $keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
        $modes = ['major', 'minor'];

        $this
            ->limitTo(100)
            ->forAll(
                Generator\elements($genres),
                Generator\choose(20, 300),    // bpm
                Generator\elements($keys),
                Generator\elements($modes),
                Generator\elements($moods),
                Generator\bool(),             // has_vocals
                Generator\bool()              // is_acoustic
            )
            ->then(function ($genre, $bpm, $key, $mode, $mood, $hasVocals, $isAcoustic) {
                $originalData = [
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'genre' => $genre,
                    'bpm' => (float) $bpm,
                    'key' => $key,
                    'mode' => $mode,
                    'mood' => $mood,
                    'has_vocals' => $hasVocals,
                    'is_acoustic' => $isAcoustic,
                ];

                $track = Track::create($originalData);
                $retrieved = Track::find($track->id);

                $this->assertNotNull($retrieved);
                $this->assertEquals($genre, $retrieved->genre);
                $this->assertEqualsWithDelta((float) $bpm, $retrieved->bpm, 0.0001);
                $this->assertEquals($key, $retrieved->key);
                $this->assertEquals($mode, $retrieved->mode);
                $this->assertEquals($mood, $retrieved->mood);
                $this->assertEquals($hasVocals, $retrieved->has_vocals);
                $this->assertEquals($isAcoustic, $retrieved->is_acoustic);
            });
    }
}
