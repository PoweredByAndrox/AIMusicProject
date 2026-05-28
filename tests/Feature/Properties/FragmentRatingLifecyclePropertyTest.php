<?php

namespace Tests\Feature\Properties;

use App\Jobs\FragmentAnalysisJob;
use App\Models\Fragment;
use App\Models\Track;
use App\Services\FragmentService;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Support\Facades\Queue;
use Tests\TestCase;

/**
 * Property Test: Fragment Rating Lifecycle
 * 
 * @tag Feature: advanced-music-player, Property 9: Fragment Rating Lifecycle
 * **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.7**
 * 
 * For any track fragment:
 * - Creating with like sets is_liked=true
 * - Creating with dislike sets is_liked=false
 * - Changing rating updates is_liked
 * - Removing rating deletes the fragment from database
 * - After rating, ML analysis task is created
 */
class FragmentRatingLifecyclePropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private FragmentService $fragmentService;

    protected function setUp(): void
    {
        parent::setUp();
        $this->fragmentService = new FragmentService();
    }

    /**
     * Helper to create a track with random data.
     */
    private function createTrack(string $title = 'Test Track', float $duration = 300.0): Track
    {
        return Track::create([
            'title' => $title,
            'duration' => $duration,
            'file_path' => '/storage/test_' . uniqid() . '.mp3',
            'source' => 'upload',
        ]);
    }

    /**
     * @test
     * 
     * Property: Creating fragment with is_liked=true sets is_liked=true
     * Validates: Requirement 7.1
     * 
     * WHEN пользователь выделяет область на waveform и ставит лайк 
     * THEN THE Fragment_Rating SHALL создать фрагмент с is_liked=true
     */
    public function creating_fragment_with_like_sets_is_liked_true(): void
    {
        Queue::fake();

        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 100),    // start_time (0-100 seconds)
                Generator\choose(10, 50)     // duration (10-50 seconds)
            )
            ->then(function ($startTime, $fragmentDuration) {
                // Create a track with sufficient duration
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for like test", $trackDuration);
                
                $endTime = $startTime + $fragmentDuration;
                
                // Create fragment with like (is_liked=true)
                $fragment = $this->fragmentService->rateFragment(
                    $track->id,
                    (float) $startTime,
                    (float) $endTime,
                    true // is_liked = true (like)
                );
                
                // Verify is_liked is true
                $this->assertTrue($fragment->is_liked);
                $this->assertEquals($track->id, $fragment->track_id);
                $this->assertEquals((float) $startTime, $fragment->start_time);
                $this->assertEquals((float) $endTime, $fragment->end_time);
                
                // Verify in database
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                    'track_id' => $track->id,
                    'is_liked' => true,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Creating fragment with is_liked=false sets is_liked=false
     * Validates: Requirement 7.2
     * 
     * WHEN пользователь выделяет область на waveform и ставит дизлайк 
     * THEN THE Fragment_Rating SHALL создать фрагмент с is_liked=false
     */
    public function creating_fragment_with_dislike_sets_is_liked_false(): void
    {
        Queue::fake();

        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 100),    // start_time (0-100 seconds)
                Generator\choose(10, 50)     // duration (10-50 seconds)
            )
            ->then(function ($startTime, $fragmentDuration) {
                // Create a track with sufficient duration
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for dislike test", $trackDuration);
                
                $endTime = $startTime + $fragmentDuration;
                
                // Create fragment with dislike (is_liked=false)
                $fragment = $this->fragmentService->rateFragment(
                    $track->id,
                    (float) $startTime,
                    (float) $endTime,
                    false // is_liked = false (dislike)
                );
                
                // Verify is_liked is false
                $this->assertFalse($fragment->is_liked);
                $this->assertEquals($track->id, $fragment->track_id);
                $this->assertEquals((float) $startTime, $fragment->start_time);
                $this->assertEquals((float) $endTime, $fragment->end_time);
                
                // Verify in database
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                    'track_id' => $track->id,
                    'is_liked' => false,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Updating rating changes is_liked value
     * Validates: Requirement 7.3
     * 
     * WHEN пользователь изменяет оценку фрагмента 
     * THEN THE Fragment_Rating SHALL обновить значение is_liked
     */
    public function updating_rating_changes_is_liked_value(): void
    {
        Queue::fake();

        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 100),    // start_time
                Generator\choose(10, 50),    // duration
                Generator\bool()             // initial is_liked
            )
            ->then(function ($startTime, $fragmentDuration, $initialIsLiked) {
                // Create a track with sufficient duration
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for update test", $trackDuration);
                
                $endTime = $startTime + $fragmentDuration;
                
                // Create fragment with initial rating
                $fragment = $this->fragmentService->rateFragment(
                    $track->id,
                    (float) $startTime,
                    (float) $endTime,
                    $initialIsLiked
                );
                
                // Verify initial state
                $this->assertEquals($initialIsLiked, $fragment->is_liked);
                
                // Update to opposite rating
                $newIsLiked = !$initialIsLiked;
                $updatedFragment = $this->fragmentService->updateRating($fragment->id, $newIsLiked);
                
                // Verify is_liked was updated
                $this->assertEquals($newIsLiked, $updatedFragment->is_liked);
                $this->assertNotEquals($initialIsLiked, $updatedFragment->is_liked);
                
                // Verify in database
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                    'is_liked' => $newIsLiked,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Removing rating deletes the fragment from database
     * Validates: Requirement 7.4
     * 
     * WHEN пользователь удаляет оценку фрагмента 
     * THEN THE Fragment_Rating SHALL удалить запись из базы данных
     */
    public function removing_rating_deletes_fragment_from_database(): void
    {
        Queue::fake();

        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 100),    // start_time
                Generator\choose(10, 50),    // duration
                Generator\bool()             // is_liked
            )
            ->then(function ($startTime, $fragmentDuration, $isLiked) {
                // Create a track with sufficient duration
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for removal test", $trackDuration);
                
                $endTime = $startTime + $fragmentDuration;
                
                // Create fragment with rating
                $fragment = $this->fragmentService->rateFragment(
                    $track->id,
                    (float) $startTime,
                    (float) $endTime,
                    $isLiked
                );
                
                $fragmentId = $fragment->id;
                
                // Verify fragment exists
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragmentId,
                ]);
                
                // Remove the rating (deletes the fragment)
                $this->fragmentService->removeRating($fragmentId);
                
                // Verify fragment was deleted from database
                $this->assertDatabaseMissing('fragments', [
                    'id' => $fragmentId,
                ]);
                
                // Verify fragment cannot be found
                $this->assertNull(Fragment::find($fragmentId));
            });
    }

    /**
     * @test
     * 
     * Property: rated_at timestamp is set when rating is applied
     * Validates: Requirements 7.1, 7.2 (implicit - rated_at should be set)
     */
    public function rated_at_timestamp_is_set_when_rating_is_applied(): void
    {
        Queue::fake();

        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 100),    // start_time
                Generator\choose(10, 50),    // duration
                Generator\bool()             // is_liked
            )
            ->then(function ($startTime, $fragmentDuration, $isLiked) {
                // Create a track with sufficient duration
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for timestamp test", $trackDuration);
                
                $endTime = $startTime + $fragmentDuration;
                
                // Use subSecond to account for timing precision issues
                $beforeCreation = now()->subSecond();
                
                // Create fragment with rating
                $fragment = $this->fragmentService->rateFragment(
                    $track->id,
                    (float) $startTime,
                    (float) $endTime,
                    $isLiked
                );
                
                $afterCreation = now()->addSecond();
                
                // Verify rated_at is set
                $this->assertNotNull($fragment->rated_at);
                
                // Verify rated_at is within expected time range (with 1 second tolerance)
                $this->assertTrue(
                    $fragment->rated_at->greaterThanOrEqualTo($beforeCreation),
                    'rated_at should be >= time before creation (with tolerance)'
                );
                $this->assertTrue(
                    $fragment->rated_at->lessThanOrEqualTo($afterCreation),
                    'rated_at should be <= time after creation (with tolerance)'
                );
            });
    }

    /**
     * @test
     * 
     * Property: After rating, ML analysis task is created
     * Validates: Requirement 7.7
     * 
     * WHEN фрагмент оценён THEN THE System SHALL поставить его 
     * в очередь на ML-анализ эмбеддинга
     */
    public function after_rating_ml_analysis_task_is_created(): void
    {
        Queue::fake();

        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 100),    // start_time
                Generator\choose(10, 50),    // duration
                Generator\bool()             // is_liked
            )
            ->then(function ($startTime, $fragmentDuration, $isLiked) {
                // Clear any previously queued jobs
                Queue::fake();
                
                // Create a track with sufficient duration
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for ML analysis test", $trackDuration);
                
                $endTime = $startTime + $fragmentDuration;
                
                // Create fragment with rating
                $fragment = $this->fragmentService->rateFragment(
                    $track->id,
                    (float) $startTime,
                    (float) $endTime,
                    $isLiked
                );
                
                // Verify FragmentAnalysisJob was dispatched
                Queue::assertPushed(FragmentAnalysisJob::class, function ($job) use ($fragment) {
                    return $job->fragment->id === $fragment->id;
                });
                
                // Verify analysis_status is set to pending
                $this->assertEquals('pending', $fragment->analysis_status);
            });
    }

    /**
     * @test
     * 
     * Property: Fragment rating lifecycle round-trip (create -> update -> remove)
     * Validates: Requirements 7.1, 7.2, 7.3, 7.4
     */
    public function fragment_rating_lifecycle_round_trip(): void
    {
        Queue::fake();

        $this
            ->limitTo(50)
            ->forAll(
                Generator\choose(0, 100),    // start_time
                Generator\choose(10, 50),    // duration
                Generator\bool()             // initial is_liked
            )
            ->then(function ($startTime, $fragmentDuration, $initialIsLiked) {
                // Create a track with sufficient duration
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for lifecycle test", $trackDuration);
                
                $endTime = $startTime + $fragmentDuration;
                
                // Step 1: Create fragment with initial rating
                $fragment = $this->fragmentService->rateFragment(
                    $track->id,
                    (float) $startTime,
                    (float) $endTime,
                    $initialIsLiked
                );
                
                $fragmentId = $fragment->id;
                
                // Verify initial state
                $this->assertEquals($initialIsLiked, $fragment->is_liked);
                $this->assertNotNull($fragment->rated_at);
                
                // Step 2: Update rating to opposite
                $newIsLiked = !$initialIsLiked;
                $updatedFragment = $this->fragmentService->updateRating($fragmentId, $newIsLiked);
                
                // Verify updated state
                $this->assertEquals($newIsLiked, $updatedFragment->is_liked);
                $this->assertNotNull($updatedFragment->rated_at);
                
                // Step 3: Remove rating (deletes fragment)
                $this->fragmentService->removeRating($fragmentId);
                
                // Verify fragment is deleted
                $this->assertDatabaseMissing('fragments', [
                    'id' => $fragmentId,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Multiple fragments can be created for the same track
     * Validates: Requirements 7.1, 7.2 (implicit - multiple fragments allowed)
     */
    public function multiple_fragments_can_be_created_for_same_track(): void
    {
        Queue::fake();

        $this
            ->limitTo(50)
            ->forAll(Generator\choose(2, 5)) // number of fragments
            ->then(function ($numFragments) {
                // Create a track with sufficient duration for multiple non-overlapping fragments
                $fragmentDuration = 20;
                $trackDuration = ($numFragments * $fragmentDuration) + 100;
                $track = $this->createTrack("Track for multiple fragments", $trackDuration);
                
                $createdFragments = [];
                
                // Create multiple non-overlapping fragments
                for ($i = 0; $i < $numFragments; $i++) {
                    $startTime = $i * ($fragmentDuration + 5); // 5 second gap between fragments
                    $endTime = $startTime + $fragmentDuration;
                    $isLiked = $i % 2 === 0; // Alternate between like and dislike
                    
                    $fragment = $this->fragmentService->rateFragment(
                        $track->id,
                        (float) $startTime,
                        (float) $endTime,
                        $isLiked
                    );
                    
                    $createdFragments[] = $fragment;
                }
                
                // Verify all fragments were created
                $this->assertCount($numFragments, $createdFragments);
                
                // Verify all fragments exist in database
                foreach ($createdFragments as $fragment) {
                    $this->assertDatabaseHas('fragments', [
                        'id' => $fragment->id,
                        'track_id' => $track->id,
                    ]);
                }
                
                // Verify getFragmentsForTrack returns all fragments
                $retrievedFragments = $this->fragmentService->getFragmentsForTrack($track->id);
                $this->assertCount($numFragments, $retrievedFragments);
            });
    }
}
