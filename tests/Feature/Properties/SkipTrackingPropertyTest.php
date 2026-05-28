<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use App\Models\TrackSkip;
use App\Services\SkipTrackerService;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: Skip Tracking
 * 
 * @tag Feature: advanced-music-player, Property 18: Отслеживание пропусков
 * **Validates: Requirements 18.1, 18.3**
 * 
 * For any track skip in the first 30 seconds:
 * - A record is created in track_skips with position_at_skip < 30
 * - After 3+ skips, the track is significantly deprioritized in recommendations
 */
class SkipTrackingPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private SkipTrackerService $skipService;

    protected function setUp(): void
    {
        parent::setUp();
        $this->skipService = new SkipTrackerService();
    }

    /**
     * Helper to create a track with random data.
     */
    private function createTrack(string $title = 'Test Track'): Track
    {
        return Track::create([
            'title' => $title,
            'duration' => rand(60, 600),
            'file_path' => '/storage/test_' . uniqid() . '.mp3',
            'source' => 'upload',
        ]);
    }

    /**
     * @test
     * 
     * Property: Early skip (< 30 seconds) creates a record with position_at_skip < 30
     * Validates: Requirement 18.1
     * 
     * WHEN пользователь пропускает трек в первые 30 секунд 
     * THEN THE Skip_Tracker SHALL записать событие пропуска
     */
    public function early_skip_creates_record_with_position_less_than_30(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 2999), // 0 to 29.99 seconds (in centiseconds)
                Generator\elements(['queue', 'playlist', 'radio', 'ai_dj'])
            )
            ->then(function ($positionCentiseconds, $source) {
                // Convert to seconds with decimal precision
                $positionAtSkip = $positionCentiseconds / 100.0;
                
                // Create a fresh track for each iteration
                $track = $this->createTrack("Track for early skip at {$positionAtSkip}s");
                
                // Record the skip
                $skip = $this->skipService->recordSkip($track->id, $positionAtSkip, $source);
                
                // Verify the skip was recorded correctly
                $this->assertNotNull($skip);
                $this->assertEquals($track->id, $skip->track_id);
                $this->assertEquals($positionAtSkip, $skip->position_at_skip);
                $this->assertEquals($source, $skip->source);
                $this->assertNotNull($skip->skipped_at);
                
                // Verify position_at_skip < 30 (Property 18 requirement)
                $this->assertLessThan(30, $skip->position_at_skip);
                
                // Verify the record exists in database
                $this->assertDatabaseHas('track_skips', [
                    'track_id' => $track->id,
                    'source' => $source,
                ]);
                
                // Verify it counts as an early skip
                $earlySkipCount = $this->skipService->getEarlySkipCount($track->id);
                $this->assertEquals(1, $earlySkipCount);
            });
    }

    /**
     * @test
     * 
     * Property: After 3+ early skips, track should be penalized in recommendations
     * Validates: Requirement 18.3
     * 
     * WHEN трек пропущен 3+ раза THEN THE Smart_Radio SHALL 
     * значительно снизить его приоритет
     */
    public function three_or_more_early_skips_triggers_penalty(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(3, 10) // 3 to 10 skips
            )
            ->then(function ($numSkips) {
                // Create a fresh track
                $track = $this->createTrack("Track for penalty test with {$numSkips} skips");
                
                // Record multiple early skips (all < 30 seconds)
                for ($i = 0; $i < $numSkips; $i++) {
                    $positionAtSkip = rand(0, 2999) / 100.0; // 0 to 29.99 seconds
                    $this->skipService->recordSkip($track->id, $positionAtSkip, 'queue');
                }
                
                // Verify early skip count matches
                $earlySkipCount = $this->skipService->getEarlySkipCount($track->id);
                $this->assertEquals($numSkips, $earlySkipCount);
                
                // Verify track should be penalized (Property 18 requirement)
                $shouldPenalize = $this->skipService->shouldPenalize($track->id);
                $this->assertTrue(
                    $shouldPenalize,
                    "Track with {$numSkips} early skips should be penalized"
                );
                
                // Verify stats reflect penalty status
                $stats = $this->skipService->getSkipStats($track->id);
                $this->assertTrue($stats['should_penalize']);
                $this->assertEquals($numSkips, $stats['early_skips']);
            });
    }

    /**
     * @test
     * 
     * Property: Less than 3 early skips does NOT trigger penalty
     * Validates: Requirement 18.3 (inverse case)
     */
    public function less_than_three_early_skips_does_not_trigger_penalty(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 2) // 0, 1, or 2 skips
            )
            ->then(function ($numSkips) {
                // Create a fresh track
                $track = $this->createTrack("Track for no-penalty test with {$numSkips} skips");
                
                // Record early skips (all < 30 seconds)
                for ($i = 0; $i < $numSkips; $i++) {
                    $positionAtSkip = rand(0, 2999) / 100.0; // 0 to 29.99 seconds
                    $this->skipService->recordSkip($track->id, $positionAtSkip, 'queue');
                }
                
                // Verify early skip count matches
                $earlySkipCount = $this->skipService->getEarlySkipCount($track->id);
                $this->assertEquals($numSkips, $earlySkipCount);
                
                // Verify track should NOT be penalized
                $shouldPenalize = $this->skipService->shouldPenalize($track->id);
                $this->assertFalse(
                    $shouldPenalize,
                    "Track with only {$numSkips} early skips should NOT be penalized"
                );
                
                // Verify stats reflect no penalty
                $stats = $this->skipService->getSkipStats($track->id);
                $this->assertFalse($stats['should_penalize']);
            });
    }

    /**
     * @test
     * 
     * Property: Skips at position >= 30 seconds do NOT count toward penalty threshold
     * Validates: Requirements 18.1, 18.3 (boundary condition)
     */
    public function late_skips_do_not_count_toward_penalty(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(3000, 30000), // 30.00 to 300.00 seconds (in centiseconds)
                Generator\choose(3, 10) // number of late skips
            )
            ->then(function ($positionCentiseconds, $numSkips) {
                // Convert to seconds
                $positionAtSkip = $positionCentiseconds / 100.0;
                
                // Create a fresh track
                $track = $this->createTrack("Track for late skip test at {$positionAtSkip}s");
                
                // Record multiple late skips (all >= 30 seconds)
                for ($i = 0; $i < $numSkips; $i++) {
                    $latePosition = 30.0 + ($i * 10); // 30, 40, 50, etc.
                    $this->skipService->recordSkip($track->id, $latePosition, 'queue');
                }
                
                // Verify total skip count
                $totalSkips = $this->skipService->getSkipCount($track->id);
                $this->assertEquals($numSkips, $totalSkips);
                
                // Verify early skip count is 0 (none were < 30 seconds)
                $earlySkipCount = $this->skipService->getEarlySkipCount($track->id);
                $this->assertEquals(0, $earlySkipCount);
                
                // Verify track should NOT be penalized (no early skips)
                $shouldPenalize = $this->skipService->shouldPenalize($track->id);
                $this->assertFalse(
                    $shouldPenalize,
                    "Track with {$numSkips} late skips (>= 30s) should NOT be penalized"
                );
            });
    }

    /**
     * @test
     * 
     * Property: Mixed early and late skips - only early skips count toward penalty
     * Validates: Requirements 18.1, 18.3
     */
    public function only_early_skips_count_toward_penalty_threshold(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 5), // number of early skips
                Generator\choose(0, 5)  // number of late skips
            )
            ->then(function ($numEarlySkips, $numLateSkips) {
                // Create a fresh track
                $track = $this->createTrack("Track for mixed skip test");
                
                // Record early skips (< 30 seconds)
                for ($i = 0; $i < $numEarlySkips; $i++) {
                    $earlyPosition = rand(0, 2999) / 100.0; // 0 to 29.99 seconds
                    $this->skipService->recordSkip($track->id, $earlyPosition, 'queue');
                }
                
                // Record late skips (>= 30 seconds)
                for ($i = 0; $i < $numLateSkips; $i++) {
                    $latePosition = 30.0 + rand(0, 27000) / 100.0; // 30 to 300 seconds
                    $this->skipService->recordSkip($track->id, $latePosition, 'playlist');
                }
                
                // Verify total skip count
                $totalSkips = $this->skipService->getSkipCount($track->id);
                $this->assertEquals($numEarlySkips + $numLateSkips, $totalSkips);
                
                // Verify early skip count
                $earlySkipCount = $this->skipService->getEarlySkipCount($track->id);
                $this->assertEquals($numEarlySkips, $earlySkipCount);
                
                // Verify penalty is based ONLY on early skips
                $shouldPenalize = $this->skipService->shouldPenalize($track->id);
                $expectedPenalty = $numEarlySkips >= 3;
                
                $this->assertEquals(
                    $expectedPenalty,
                    $shouldPenalize,
                    "Penalty should be based on {$numEarlySkips} early skips, not total {$totalSkips}"
                );
            });
    }

    /**
     * @test
     * 
     * Property: Skip at exactly 30 seconds boundary does NOT count as early skip
     * Validates: Requirement 18.1 (boundary condition - "first 30 seconds" means < 30)
     */
    public function skip_at_exactly_30_seconds_is_not_early(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\choose(3, 10) // number of skips at exactly 30 seconds
            )
            ->then(function ($numSkips) {
                // Create a fresh track
                $track = $this->createTrack("Track for boundary test");
                
                // Record skips at exactly 30.0 seconds
                for ($i = 0; $i < $numSkips; $i++) {
                    $this->skipService->recordSkip($track->id, 30.0, 'queue');
                }
                
                // Verify total skip count
                $totalSkips = $this->skipService->getSkipCount($track->id);
                $this->assertEquals($numSkips, $totalSkips);
                
                // Verify early skip count is 0 (30.0 is NOT < 30)
                $earlySkipCount = $this->skipService->getEarlySkipCount($track->id);
                $this->assertEquals(0, $earlySkipCount);
                
                // Verify track should NOT be penalized
                $shouldPenalize = $this->skipService->shouldPenalize($track->id);
                $this->assertFalse(
                    $shouldPenalize,
                    "Track with skips at exactly 30s should NOT be penalized"
                );
            });
    }
}
