<?php

namespace Tests\Feature\Properties;

use App\Models\Fragment;
use App\Models\Track;
use App\Services\FragmentService;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Support\Facades\Queue;
use InvalidArgumentException;
use Tests\TestCase;

/**
 * Property Test: Fragment Comment Round-Trip
 * 
 * @tag Feature: advanced-music-player, Property 10: Round-trip для комментариев к фрагментам
 * **Validates: Requirements 8.1, 8.2, 8.3, 8.4**
 * 
 * For any fragment and comment text (up to 500 characters):
 * - Saving a comment and then retrieving it returns the same text
 * - Comments longer than 500 characters are rejected
 * - Removing a comment sets comment=null
 */
class FragmentCommentRoundTripPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private FragmentService $fragmentService;

    protected function setUp(): void
    {
        parent::setUp();
        $this->fragmentService = new FragmentService();
        Queue::fake();
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
     * Helper to create a fragment for testing.
     */
    private function createFragment(Track $track, float $startTime = 10.0, float $endTime = 30.0): Fragment
    {
        return $this->fragmentService->rateFragment(
            $track->id,
            $startTime,
            $endTime,
            true // is_liked
        );
    }

    /**
     * @test
     * 
     * Property: Saving a comment and retrieving it returns the same text
     * Validates: Requirement 8.1
     * 
     * WHEN пользователь добавляет комментарий к фрагменту 
     * THEN THE Fragment_Comment SHALL сохранить текст с привязкой к fragment_id
     */
    public function saving_comment_and_retrieving_returns_same_text(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\string(), // random comment text
                Generator\choose(0, 100), // start_time
                Generator\choose(10, 50)  // duration
            )
            ->then(function ($commentText, $startTime, $fragmentDuration) {
                // Limit comment to 500 characters for valid input
                $validComment = mb_substr($commentText, 0, 500);
                
                // Skip empty comments (edge case)
                if (empty($validComment)) {
                    $validComment = 'Test comment';
                }
                
                // Create track and fragment
                $trackDuration = $startTime + $fragmentDuration + 10;
                $track = $this->createTrack("Track for comment test", $trackDuration);
                $fragment = $this->createFragment($track, (float) $startTime, (float) ($startTime + $fragmentDuration));
                
                // Add comment
                $updatedFragment = $this->fragmentService->addComment($fragment->id, $validComment);
                
                // Verify comment is saved correctly
                $this->assertEquals($validComment, $updatedFragment->comment);
                
                // Verify in database
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                    'comment' => $validComment,
                ]);
                
                // Retrieve fragment and verify comment
                $retrievedFragment = Fragment::find($fragment->id);
                $this->assertEquals($validComment, $retrievedFragment->comment);
            });
    }

    /**
     * @test
     * 
     * Property: Comments longer than 500 characters are rejected
     * Validates: Requirement 8.2
     * 
     * THE Fragment_Comment SHALL поддерживать текст до 500 символов
     */
    public function comments_longer_than_500_characters_are_rejected(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\choose(501, 1000) // length of comment (501-1000 chars)
            )
            ->then(function ($commentLength) {
                // Create track and fragment
                $track = $this->createTrack("Track for long comment test", 300.0);
                $fragment = $this->createFragment($track);
                
                // Generate a comment longer than 500 characters
                $longComment = str_repeat('a', $commentLength);
                
                // Attempt to add long comment should throw exception
                $this->expectException(InvalidArgumentException::class);
                $this->expectExceptionMessage('Comment cannot exceed 500 characters');
                
                $this->fragmentService->addComment($fragment->id, $longComment);
            });
    }

    /**
     * @test
     * 
     * Property: Updating comment changes the stored text
     * Validates: Requirement 8.3
     * 
     * WHEN пользователь редактирует комментарий 
     * THEN THE Fragment_Comment SHALL обновить текст
     */
    public function updating_comment_changes_stored_text(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\string(), // initial comment
                Generator\string()  // updated comment
            )
            ->then(function ($initialComment, $updatedComment) {
                // Limit comments to 500 characters
                $validInitial = mb_substr($initialComment, 0, 500);
                $validUpdated = mb_substr($updatedComment, 0, 500);
                
                // Ensure non-empty comments
                if (empty($validInitial)) {
                    $validInitial = 'Initial comment';
                }
                if (empty($validUpdated)) {
                    $validUpdated = 'Updated comment';
                }
                
                // Create track and fragment
                $track = $this->createTrack("Track for update comment test", 300.0);
                $fragment = $this->createFragment($track);
                
                // Add initial comment
                $this->fragmentService->addComment($fragment->id, $validInitial);
                
                // Update comment
                $updatedFragment = $this->fragmentService->updateComment($fragment->id, $validUpdated);
                
                // Verify comment was updated
                $this->assertEquals($validUpdated, $updatedFragment->comment);
                
                // Verify in database
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                    'comment' => $validUpdated,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Removing comment sets comment to null
     * Validates: Requirement 8.4
     * 
     * WHEN пользователь удаляет комментарий 
     * THEN THE Fragment_Comment SHALL удалить запись (set to null)
     */
    public function removing_comment_sets_comment_to_null(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\string() // comment text
            )
            ->then(function ($commentText) {
                // Limit comment to 500 characters
                $validComment = mb_substr($commentText, 0, 500);
                
                // Ensure non-empty comment
                if (empty($validComment)) {
                    $validComment = 'Comment to be removed';
                }
                
                // Create track and fragment
                $track = $this->createTrack("Track for remove comment test", 300.0);
                $fragment = $this->createFragment($track);
                
                // Add comment
                $this->fragmentService->addComment($fragment->id, $validComment);
                
                // Verify comment exists
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                    'comment' => $validComment,
                ]);
                
                // Remove comment
                $updatedFragment = $this->fragmentService->removeComment($fragment->id);
                
                // Verify comment is null
                $this->assertNull($updatedFragment->comment);
                
                // Verify in database
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                    'comment' => null,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Comment round-trip lifecycle (add -> update -> remove)
     * Validates: Requirements 8.1, 8.3, 8.4
     */
    public function comment_round_trip_lifecycle(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\string(), // initial comment
                Generator\string()  // updated comment
            )
            ->then(function ($initialComment, $updatedComment) {
                // Limit comments to 500 characters
                $validInitial = mb_substr($initialComment, 0, 500);
                $validUpdated = mb_substr($updatedComment, 0, 500);
                
                // Ensure non-empty comments
                if (empty($validInitial)) {
                    $validInitial = 'Initial lifecycle comment';
                }
                if (empty($validUpdated)) {
                    $validUpdated = 'Updated lifecycle comment';
                }
                
                // Create track and fragment
                $track = $this->createTrack("Track for lifecycle test", 300.0);
                $fragment = $this->createFragment($track);
                
                // Step 1: Add comment
                $fragmentWithComment = $this->fragmentService->addComment($fragment->id, $validInitial);
                $this->assertEquals($validInitial, $fragmentWithComment->comment);
                
                // Step 2: Update comment
                $fragmentUpdated = $this->fragmentService->updateComment($fragment->id, $validUpdated);
                $this->assertEquals($validUpdated, $fragmentUpdated->comment);
                
                // Step 3: Remove comment
                $fragmentNoComment = $this->fragmentService->removeComment($fragment->id);
                $this->assertNull($fragmentNoComment->comment);
                
                // Verify fragment still exists (only comment was removed)
                $this->assertDatabaseHas('fragments', [
                    'id' => $fragment->id,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Comments with exactly 500 characters are accepted
     * Validates: Requirement 8.2 (boundary test)
     */
    public function comments_with_exactly_500_characters_are_accepted(): void
    {
        // Create track and fragment
        $track = $this->createTrack("Track for boundary test", 300.0);
        $fragment = $this->createFragment($track);
        
        // Generate exactly 500 character comment
        $exactComment = str_repeat('a', 500);
        
        // Add comment should succeed
        $updatedFragment = $this->fragmentService->addComment($fragment->id, $exactComment);
        
        // Verify comment is saved
        $this->assertEquals($exactComment, $updatedFragment->comment);
        $this->assertEquals(500, mb_strlen($updatedFragment->comment));
        
        // Verify in database
        $this->assertDatabaseHas('fragments', [
            'id' => $fragment->id,
            'comment' => $exactComment,
        ]);
    }

    /**
     * @test
     * 
     * Property: Unicode comments are preserved correctly
     * Validates: Requirement 8.1, 8.2 (unicode support)
     */
    public function unicode_comments_are_preserved_correctly(): void
    {
        // Create track and fragment
        $track = $this->createTrack("Track for unicode test", 300.0);
        $fragment = $this->createFragment($track);
        
        // Unicode comment with various scripts
        $unicodeComment = 'Привет мир! 你好世界! مرحبا بالعالم! 🎵🎶🎸';
        
        // Add comment
        $updatedFragment = $this->fragmentService->addComment($fragment->id, $unicodeComment);
        
        // Verify comment is preserved exactly
        $this->assertEquals($unicodeComment, $updatedFragment->comment);
        
        // Retrieve and verify
        $retrievedFragment = Fragment::find($fragment->id);
        $this->assertEquals($unicodeComment, $retrievedFragment->comment);
    }
}
