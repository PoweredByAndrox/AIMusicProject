<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use App\Models\Fragment;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Database\QueryException;
use Tests\TestCase;

/**
 * Property Test: Fragment Time Range Validity
 * 
 * @tag Feature: ai-music-recommendations, Property 2: Fragment Time Range Validity
 * **Validates: Requirements 4.3, 6.1, 6.2**
 * 
 * For any fragment created in the system, the condition must hold:
 * start_time < end_time, and both values must be within track duration [0, duration].
 */
class FragmentTimeRangePropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private Track $track;

    protected function setUp(): void
    {
        parent::setUp();
        
        // Create a track with known duration for testing
        $this->track = Track::create([
            'title' => 'Test Track',
            'duration' => 300.0, // 5 minutes
            'file_path' => '/storage/test.mp3',
            'source' => 'upload',
        ]);
    }

    /**
     * @test
     */
    public function valid_fragments_have_start_time_less_than_end_time(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 290),  // start_time (0 to 290)
                Generator\choose(1, 10)    // duration offset (1 to 10 seconds)
            )
            ->then(function ($startTime, $durationOffset) {
                $endTime = min($startTime + $durationOffset, 300.0);
                
                // Skip if invalid range
                if ($startTime >= $endTime) {
                    return;
                }

                $fragment = Fragment::create([
                    'track_id' => $this->track->id,
                    'start_time' => (float) $startTime,
                    'end_time' => (float) $endTime,
                ]);

                $retrieved = Fragment::find($fragment->id);

                $this->assertNotNull($retrieved);
                $this->assertLessThan($retrieved->end_time, $retrieved->start_time);
                $this->assertGreaterThanOrEqual(0, $retrieved->start_time);
                $this->assertLessThanOrEqual($this->track->duration, $retrieved->end_time);
            });
    }

    /**
     * @test
     */
    public function fragment_duration_is_positive(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 250),  // start_time
                Generator\choose(10, 50)   // fragment duration
            )
            ->then(function ($startTime, $fragmentDuration) {
                $endTime = min($startTime + $fragmentDuration, 300.0);
                
                if ($startTime >= $endTime) {
                    return;
                }

                $fragment = Fragment::create([
                    'track_id' => $this->track->id,
                    'start_time' => (float) $startTime,
                    'end_time' => (float) $endTime,
                ]);

                $this->assertGreaterThan(0, $fragment->duration);
                $this->assertEqualsWithDelta(
                    $endTime - $startTime,
                    $fragment->duration,
                    0.0001
                );
            });
    }

    /**
     * @test
     */
    public function database_rejects_invalid_time_range(): void
    {
        // Test that database CHECK constraint rejects end_time <= start_time
        $this->expectException(QueryException::class);

        Fragment::create([
            'track_id' => $this->track->id,
            'start_time' => 100.0,
            'end_time' => 50.0, // Invalid: end_time < start_time
        ]);
    }

    /**
     * @test
     */
    public function database_rejects_equal_start_and_end_time(): void
    {
        $this->expectException(QueryException::class);

        Fragment::create([
            'track_id' => $this->track->id,
            'start_time' => 100.0,
            'end_time' => 100.0, // Invalid: end_time == start_time
        ]);
    }
}
