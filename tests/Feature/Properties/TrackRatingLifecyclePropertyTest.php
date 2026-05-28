<?php

namespace Tests\Feature\Properties;

use App\Models\RatingHistory;
use App\Models\Track;
use App\Models\TrackRating;
use App\Services\TrackRatingService;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use InvalidArgumentException;
use Tests\TestCase;

/**
 * Property Test: 5-Star Rating Lifecycle
 * 
 * @tag Feature: advanced-music-player, Property 13: 5-Star Rating Lifecycle
 * **Validates: Requirements 11.1, 11.2, 11.3, 14.1**
 * 
 * For any track and rating (1-5):
 * - Setting a rating saves the value
 * - Changing a rating creates a record in Rating_History with old_rating and new_rating
 * - Removing a rating sets rating=NULL (deletes the TrackRating record)
 * - Ratings outside range 1-5 are rejected
 */
class TrackRatingLifecyclePropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private TrackRatingService $ratingService;

    protected function setUp(): void
    {
        parent::setUp();
        $this->ratingService = new TrackRatingService();
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
     * Property: Setting a rating saves the value correctly
     * Validates: Requirement 11.1
     * 
     * WHEN пользователь устанавливает рейтинг трека 
     * THEN THE Track_Rating SHALL сохранить значение от 1 до 5 звёзд
     */
    public function setting_rating_saves_the_value_correctly(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($rating) {
                // Create a fresh track for each iteration
                $track = $this->createTrack("Track for rating {$rating}");
                
                // Set the rating
                $trackRating = $this->ratingService->setRating($track->id, $rating);
                
                // Verify the rating was saved correctly
                $this->assertNotNull($trackRating);
                $this->assertEquals($rating, $trackRating->rating);
                $this->assertEquals($track->id, $trackRating->track_id);
                
                // Verify we can retrieve the rating
                $retrievedRating = $this->ratingService->getRating($track->id);
                $this->assertEquals($rating, $retrievedRating);
                
                // Verify the rating exists in the database
                $this->assertDatabaseHas('track_ratings', [
                    'track_id' => $track->id,
                    'rating' => $rating,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Changing a rating creates a history entry with old and new values
     * Validates: Requirements 11.2, 14.1
     * 
     * WHEN пользователь изменяет рейтинг 
     * THEN THE Track_Rating SHALL обновить значение и записать в Rating_History
     * THEN THE Rating_History SHALL записать: track_id, old_rating, new_rating, timestamp
     */
    public function changing_rating_creates_history_entry_with_old_and_new_values(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(1, 5),
                Generator\choose(1, 5)
            )
            ->when(function ($oldRating, $newRating) {
                // Only test when ratings are different
                return $oldRating !== $newRating;
            })
            ->then(function ($oldRating, $newRating) {
                // Create a fresh track
                $track = $this->createTrack("Track for rating change {$oldRating} -> {$newRating}");
                
                // Set initial rating
                $this->ratingService->setRating($track->id, $oldRating);
                
                // Verify no history exists yet (first rating doesn't create history)
                $historyBefore = RatingHistory::where('track_id', $track->id)->count();
                $this->assertEquals(0, $historyBefore);
                
                // Change the rating
                $this->ratingService->setRating($track->id, $newRating);
                
                // Verify the rating was updated
                $currentRating = $this->ratingService->getRating($track->id);
                $this->assertEquals($newRating, $currentRating);
                
                // Verify history entry was created
                $history = RatingHistory::where('track_id', $track->id)->get();
                $this->assertCount(1, $history);
                
                $historyEntry = $history->first();
                $this->assertEquals($track->id, $historyEntry->track_id);
                $this->assertEquals($oldRating, $historyEntry->old_rating);
                $this->assertEquals($newRating, $historyEntry->new_rating);
                $this->assertNotNull($historyEntry->changed_at);
            });
    }

    /**
     * @test
     * 
     * Property: Removing a rating deletes the TrackRating record
     * Validates: Requirement 11.3
     * 
     * WHEN пользователь удаляет рейтинг 
     * THEN THE Track_Rating SHALL установить значение в NULL (delete the record)
     */
    public function removing_rating_deletes_the_track_rating_record(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($rating) {
                // Create a fresh track
                $track = $this->createTrack("Track for rating removal {$rating}");
                
                // Set a rating first
                $this->ratingService->setRating($track->id, $rating);
                
                // Verify rating exists
                $this->assertNotNull($this->ratingService->getRating($track->id));
                $this->assertDatabaseHas('track_ratings', [
                    'track_id' => $track->id,
                    'rating' => $rating,
                ]);
                
                // Remove the rating
                $this->ratingService->removeRating($track->id);
                
                // Verify rating is now NULL (record deleted)
                $retrievedRating = $this->ratingService->getRating($track->id);
                $this->assertNull($retrievedRating);
                
                // Verify the record was deleted from database
                $this->assertDatabaseMissing('track_ratings', [
                    'track_id' => $track->id,
                ]);
                
                // Verify history entry was created for removal
                $history = RatingHistory::where('track_id', $track->id)
                    ->whereNull('new_rating')
                    ->get();
                $this->assertCount(1, $history);
                
                $historyEntry = $history->first();
                $this->assertEquals($rating, $historyEntry->old_rating);
                $this->assertNull($historyEntry->new_rating);
            });
    }

    /**
     * @test
     * 
     * Property: Invalid ratings (0, 6, negative) are rejected with InvalidArgumentException
     * Validates: Requirement 11.1 (implicit - ratings must be 1-5)
     * 
     * Ratings outside range 1-5 are rejected
     */
    public function invalid_ratings_are_rejected_with_exception(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\oneOf(
                    Generator\choose(-100, 0),  // Negative and zero
                    Generator\choose(6, 100)     // Above 5
                )
            )
            ->then(function ($invalidRating) {
                // Create a fresh track
                $track = $this->createTrack("Track for invalid rating {$invalidRating}");
                
                // Attempt to set invalid rating should throw exception
                $exceptionThrown = false;
                try {
                    $this->ratingService->setRating($track->id, $invalidRating);
                } catch (InvalidArgumentException $e) {
                    $exceptionThrown = true;
                    $this->assertStringContainsString('between', $e->getMessage());
                }
                
                $this->assertTrue(
                    $exceptionThrown,
                    "InvalidArgumentException should be thrown for rating {$invalidRating}"
                );
                
                // Verify no rating was saved
                $this->assertNull($this->ratingService->getRating($track->id));
                $this->assertDatabaseMissing('track_ratings', [
                    'track_id' => $track->id,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Multiple rating changes create multiple history entries
     * Validates: Requirements 11.2, 14.1, 14.2
     * 
     * THE Rating_History SHALL хранить все изменения рейтингов без ограничения по времени
     */
    public function multiple_rating_changes_create_multiple_history_entries(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(2, 5))
            ->then(function ($numChanges) {
                // Create a fresh track
                $track = $this->createTrack("Track for multiple changes");
                
                // Generate a sequence of different ratings
                $ratings = [];
                $currentRating = rand(1, 5);
                $ratings[] = $currentRating;
                
                for ($i = 1; $i < $numChanges; $i++) {
                    // Ensure each rating is different from the previous
                    do {
                        $nextRating = rand(1, 5);
                    } while ($nextRating === $currentRating);
                    
                    $ratings[] = $nextRating;
                    $currentRating = $nextRating;
                }
                
                // Apply all ratings
                foreach ($ratings as $rating) {
                    $this->ratingService->setRating($track->id, $rating);
                }
                
                // Verify history has (numChanges - 1) entries (first rating doesn't create history)
                $history = $this->ratingService->getHistory($track->id);
                $this->assertCount($numChanges - 1, $history);
                
                // Verify each history entry has correct old_rating and new_rating
                // Sort by ID (ascending) to get chronological order since timestamps may be identical
                $historyArray = $history->sortBy('id')->values();
                for ($i = 0; $i < $numChanges - 1; $i++) {
                    $entry = $historyArray[$i];
                    $this->assertEquals($ratings[$i], $entry->old_rating);
                    $this->assertEquals($ratings[$i + 1], $entry->new_rating);
                }
            });
    }

    /**
     * @test
     * 
     * Property: Setting the same rating twice does not create duplicate history
     * Validates: Requirement 11.2 (implicit - only actual changes are recorded)
     */
    public function setting_same_rating_twice_does_not_create_history(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($rating) {
                // Create a fresh track
                $track = $this->createTrack("Track for same rating test");
                
                // Set rating first time
                $this->ratingService->setRating($track->id, $rating);
                
                // Set same rating again
                $this->ratingService->setRating($track->id, $rating);
                
                // Verify no history was created (no actual change)
                $history = RatingHistory::where('track_id', $track->id)->get();
                $this->assertCount(0, $history);
                
                // Verify rating is still correct
                $this->assertEquals($rating, $this->ratingService->getRating($track->id));
            });
    }

    /**
     * @test
     * 
     * Property: Rating lifecycle round-trip (set -> change -> remove -> set again)
     * Validates: Requirements 11.1, 11.2, 11.3, 14.1
     */
    public function rating_lifecycle_round_trip(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(1, 5),
                Generator\choose(1, 5),
                Generator\choose(1, 5)
            )
            ->when(function ($rating1, $rating2, $rating3) {
                // Ensure at least rating1 != rating2 for meaningful test
                return $rating1 !== $rating2;
            })
            ->then(function ($rating1, $rating2, $rating3) {
                // Create a fresh track
                $track = $this->createTrack("Track for lifecycle test");
                
                // Step 1: Set initial rating
                $this->ratingService->setRating($track->id, $rating1);
                $this->assertEquals($rating1, $this->ratingService->getRating($track->id));
                
                // Step 2: Change rating
                $this->ratingService->setRating($track->id, $rating2);
                $this->assertEquals($rating2, $this->ratingService->getRating($track->id));
                
                // Step 3: Remove rating
                $this->ratingService->removeRating($track->id);
                $this->assertNull($this->ratingService->getRating($track->id));
                
                // Step 4: Set rating again
                $this->ratingService->setRating($track->id, $rating3);
                $this->assertEquals($rating3, $this->ratingService->getRating($track->id));
                
                // Verify history contains:
                // 1. Change from rating1 to rating2
                // 2. Removal (rating2 to null)
                // No history for initial set or re-set after removal
                $history = $this->ratingService->getHistory($track->id);
                $this->assertCount(2, $history);
                
                // History is ordered by changed_at desc, so newest first
                $historyArray = $history->sortBy('changed_at')->values();
                
                // First entry: change from rating1 to rating2
                $this->assertEquals($rating1, $historyArray[0]->old_rating);
                $this->assertEquals($rating2, $historyArray[0]->new_rating);
                
                // Second entry: removal (rating2 to null)
                $this->assertEquals($rating2, $historyArray[1]->old_rating);
                $this->assertNull($historyArray[1]->new_rating);
            });
    }
}
