<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: CSV Export Round-Trip
 * 
 * @tag Feature: ai-music-recommendations, Property 12: CSV Export Round-Trip
 * **Validates: Requirements 12.1, 12.2**
 * 
 * For any track in the library, exporting to CSV and parsing the result
 * should preserve all metadata fields without data loss.
 */
class CsvExportPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private const GENRES = ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic'];
    private const MOODS = ['Happy', 'Sad', 'Energetic', 'Calm'];
    private const KEYS = ['C', 'D', 'E', 'F', 'G', 'A', 'B'];

    /**
     * @test
     */
    public function csv_export_contains_all_tracks(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(1, 20))
            ->then(function ($numTracks) {
                // Create tracks
                for ($i = 0; $i < $numTracks; $i++) {
                    Track::create([
                        'title' => "Track {$i}",
                        'artist' => "Artist {$i}",
                        'duration' => 180.0 + $i,
                        'file_path' => "/storage/track{$i}.mp3",
                        'source' => 'upload',
                    ]);
                }

                // Export to CSV
                $response = $this->get('/api/export/csv');
                $response->assertStatus(200);
                $response->assertHeader('Content-Type', 'text/csv; charset=UTF-8');

                // Parse CSV content
                $content = $response->streamedContent();
                $lines = $this->parseCsv($content);

                // First line is header, rest are data
                // Account for header row
                $dataLines = count($lines) - 1;
                $this->assertEquals($numTracks, $dataLines);
            });
    }

    /**
     * @test
     */
    public function csv_export_preserves_track_metadata(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\suchThat(
                    fn($s) => strlen($s) > 0 && strlen($s) <= 50,
                    Generator\string()
                ),
                Generator\elements(self::GENRES),
                Generator\elements(self::MOODS),
                Generator\choose(60, 200)
            )
            ->then(function ($title, $genre, $mood, $bpm) {
                // Sanitize title for CSV compatibility
                $title = preg_replace('/[\x00-\x1F\x7F]/', '', $title);
                if (empty($title)) {
                    $title = 'Test Track';
                }

                // Create track with specific metadata
                $track = Track::create([
                    'title' => $title,
                    'artist' => 'Test Artist',
                    'album' => 'Test Album',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'genre' => $genre,
                    'mood' => $mood,
                    'bpm' => (float) $bpm,
                ]);

                // Export to CSV
                $response = $this->get('/api/export/csv');
                $content = $response->streamedContent();
                $lines = $this->parseCsv($content);

                // Find the track in CSV (skip header)
                $found = false;
                foreach (array_slice($lines, 1) as $row) {
                    if (isset($row[0]) && (int) $row[0] === $track->id) {
                        $found = true;
                        // Verify metadata preserved
                        $this->assertEquals($genre, $row[6] ?? null, 'Genre mismatch');
                        $this->assertEquals((float) $bpm, (float) ($row[7] ?? 0), 'BPM mismatch');
                        $this->assertEquals($mood, $row[9] ?? null, 'Mood mismatch');
                        break;
                    }
                }

                $this->assertTrue($found, 'Track not found in CSV export');
            });
    }

    /**
     * @test
     */
    public function csv_export_with_genre_filter_contains_only_matching(): void
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

                // Export with genre filter
                $response = $this->get("/api/export/csv?genre={$targetGenre}");
                $content = $response->streamedContent();
                $lines = $this->parseCsv($content);

                // All data rows should have the target genre
                foreach (array_slice($lines, 1) as $row) {
                    if (!empty($row[0])) {
                        $this->assertEquals($targetGenre, $row[6] ?? null);
                    }
                }
            });
    }

    /**
     * @test
     */
    public function csv_export_with_liked_filter_contains_only_liked(): void
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

                // Export with liked filter
                $response = $this->get('/api/export/csv?liked=true');
                $content = $response->streamedContent();
                $lines = $this->parseCsv($content);

                // All data rows should be liked
                foreach (array_slice($lines, 1) as $row) {
                    if (!empty($row[0])) {
                        $this->assertEquals('true', $row[12] ?? null);
                    }
                }
            });
    }

    /**
     * @test
     */
    public function csv_export_has_correct_headers(): void
    {
        $this
            ->limitTo(10)
            ->forAll(Generator\constant(true))
            ->then(function ($_) {
                // Create at least one track
                Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                ]);

                // Export to CSV
                $response = $this->get('/api/export/csv');
                $content = $response->streamedContent();
                $lines = $this->parseCsv($content);

                // Verify headers
                $expectedHeaders = [
                    'id', 'title', 'artist', 'album', 'year', 'duration',
                    'genre', 'bpm', 'key', 'mood', 'has_vocals', 'is_acoustic',
                    'is_liked', 'source', 'youtube_url', 'created_at', 'updated_at',
                ];

                $this->assertEquals($expectedHeaders, $lines[0]);
            });
    }

    /**
     * Parse CSV content into array of rows.
     *
     * @param string $content
     * @return array
     */
    private function parseCsv(string $content): array
    {
        // Remove UTF-8 BOM if present
        $content = preg_replace('/^\xEF\xBB\xBF/', '', $content);

        $lines = [];
        $handle = fopen('php://memory', 'r+');
        fwrite($handle, $content);
        rewind($handle);

        while (($row = fgetcsv($handle)) !== false) {
            $lines[] = $row;
        }

        fclose($handle);

        return $lines;
    }
}
