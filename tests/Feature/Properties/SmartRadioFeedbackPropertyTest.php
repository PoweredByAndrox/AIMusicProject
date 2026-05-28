<?php

namespace Tests\Feature\Properties;

use App\Models\RadioFeedback;
use App\Models\Track;
use App\Services\ListeningHistoryService;
use App\Services\SkipTrackerService;
use App\Services\SmartRadioService;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Support\Facades\Cache;
use Tests\TestCase;

/**
 * Property Test: Smart Radio Feedback
 * 
 * @tag Feature: advanced-music-player, Property 17: Обратная связь Smart Radio
 * **Validates: Requirements 17.1, 17.2, 17.4**
 * 
 * For any track and feedback type:
 * - "Больше такого" increases track weight in feedbackWeights
 * - "Меньше такого" decreases track weight in feedbackWeights
 * - Feedback is saved to radio_feedback table
 */
class SmartRadioFeedbackPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private SmartRadioService $radioService;
    private ListeningHistoryService $historyService;
    private SkipTrackerService $skipService;

    protected function setUp(): void
    {
        parent::setUp();
        $this->historyService = new ListeningHistoryService();
        $this->skipService = new SkipTrackerService();
        $this->radioService = new SmartRadioService($this->historyService, $this->skipService);
        
        // Reset Smart Radio state before each test
        $this->radioService->resetState();
        Cache::flush();
    }

    protected function tearDown(): void
    {
        $this->radioService->resetState();
        Cache::flush();
        parent::tearDown();
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
            'analysis_status' => 'completed',
        ]);
    }

    /**
     * @test
     * 
     * Property: "More" feedback increases track weight in feedbackWeights
     * Validates: Requirement 17.1
     * 
     * WHEN пользователь нажимает "Больше такого" THEN THE Smart_Radio SHALL 
     * увеличить вес текущего трека в рекомендациях
     */
    public function more_feedback_increases_weight(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(1, 10) // number of "more" feedbacks
            )
            ->then(function ($numFeedbacks) {
                // Reset state for each iteration
                $this->radioService->resetState();
                Cache::flush();
                
                // Create a track
                $track = $this->createTrack("Track for more feedback");
                
                // Get initial weight (should be 0)
                $initialWeight = $this->radioService->getFeedbackWeight($track->id);
                $this->assertEquals(0.0, $initialWeight);
                
                // Record multiple "more" feedbacks
                $expectedWeight = 0.0;
                $moreAdjustment = SmartRadioService::getMoreWeightAdjustment();
                
                for ($i = 0; $i < $numFeedbacks; $i++) {
                    $result = $this->radioService->recordFeedback($track->id, 'more');
                    $expectedWeight += $moreAdjustment;
                    
                    // Verify weight adjustment is positive
                    $this->assertGreaterThan(
                        0,
                        $result['weight_adjustment'],
                        '"More" feedback should have positive weight adjustment'
                    );
                    
                    // Verify new weight matches expected
                    $this->assertEquals(
                        $expectedWeight,
                        $result['new_weight'],
                        "Weight after {$i} 'more' feedbacks should be {$expectedWeight}"
                    );
                }
                
                // Verify final weight is greater than initial
                $finalWeight = $this->radioService->getFeedbackWeight($track->id);
                $this->assertGreaterThan(
                    $initialWeight,
                    $finalWeight,
                    "Final weight should be greater than initial after 'more' feedbacks"
                );
                
                // Verify weight increased by expected amount
                $this->assertEquals(
                    $numFeedbacks * $moreAdjustment,
                    $finalWeight,
                    "Weight should increase by {$moreAdjustment} per 'more' feedback"
                );
            });
    }

    /**
     * @test
     * 
     * Property: "Less" feedback decreases track weight in feedbackWeights
     * Validates: Requirement 17.2
     * 
     * WHEN пользователь нажимает "Меньше такого" THEN THE Smart_Radio SHALL 
     * уменьшить вес текущего трека в рекомендациях
     */
    public function less_feedback_decreases_weight(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(1, 10) // number of "less" feedbacks
            )
            ->then(function ($numFeedbacks) {
                // Reset state for each iteration
                $this->radioService->resetState();
                Cache::flush();
                
                // Create a track
                $track = $this->createTrack("Track for less feedback");
                
                // Get initial weight (should be 0)
                $initialWeight = $this->radioService->getFeedbackWeight($track->id);
                $this->assertEquals(0.0, $initialWeight);
                
                // Record multiple "less" feedbacks
                $expectedWeight = 0.0;
                $lessAdjustment = SmartRadioService::getLessWeightAdjustment();
                
                for ($i = 0; $i < $numFeedbacks; $i++) {
                    $result = $this->radioService->recordFeedback($track->id, 'less');
                    $expectedWeight += $lessAdjustment;
                    
                    // Verify weight adjustment is negative
                    $this->assertLessThan(
                        0,
                        $result['weight_adjustment'],
                        '"Less" feedback should have negative weight adjustment'
                    );
                    
                    // Verify new weight matches expected
                    $this->assertEquals(
                        $expectedWeight,
                        $result['new_weight'],
                        "Weight after {$i} 'less' feedbacks should be {$expectedWeight}"
                    );
                }
                
                // Verify final weight is less than initial
                $finalWeight = $this->radioService->getFeedbackWeight($track->id);
                $this->assertLessThan(
                    $initialWeight,
                    $finalWeight,
                    "Final weight should be less than initial after 'less' feedbacks"
                );
                
                // Verify weight decreased by expected amount
                $this->assertEquals(
                    $numFeedbacks * $lessAdjustment,
                    $finalWeight,
                    "Weight should decrease by {$lessAdjustment} per 'less' feedback"
                );
            });
    }

    /**
     * @test
     * 
     * Property: Feedback is saved to radio_feedback table
     * Validates: Requirement 17.4
     * 
     * THE System SHALL сохранять обратную связь для обучения Preference_Model
     */
    public function feedback_saved_to_database(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\elements(['more', 'less'])
            )
            ->then(function ($feedbackType) {
                // Reset state for each iteration
                $this->radioService->resetState();
                Cache::flush();
                
                // Create a track
                $track = $this->createTrack("Track for database test");
                
                // Count existing feedback records
                $initialCount = RadioFeedback::where('track_id', $track->id)->count();
                $this->assertEquals(0, $initialCount);
                
                // Record feedback
                $this->radioService->recordFeedback($track->id, $feedbackType);
                
                // Verify feedback was saved to database
                $this->assertDatabaseHas('radio_feedback', [
                    'track_id' => $track->id,
                    'feedback_type' => $feedbackType,
                ]);
                
                // Verify count increased
                $newCount = RadioFeedback::where('track_id', $track->id)->count();
                $this->assertEquals(
                    $initialCount + 1,
                    $newCount,
                    "Feedback count should increase by 1"
                );
                
                // Verify the record has correct data
                $feedback = RadioFeedback::where('track_id', $track->id)
                    ->where('feedback_type', $feedbackType)
                    ->first();
                
                $this->assertNotNull($feedback);
                $this->assertEquals($track->id, $feedback->track_id);
                $this->assertEquals($feedbackType, $feedback->feedback_type);
                $this->assertNotNull($feedback->created_at);
            });
    }

    /**
     * @test
     * 
     * Property: Multiple feedbacks accumulate correctly
     * Validates: Requirements 17.1, 17.2
     */
    public function mixed_feedback_accumulates_correctly(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\choose(0, 5), // number of "more" feedbacks
                Generator\choose(0, 5)  // number of "less" feedbacks
            )
            ->then(function ($numMore, $numLess) {
                // Reset state for each iteration
                $this->radioService->resetState();
                Cache::flush();
                
                // Create a track
                $track = $this->createTrack("Track for mixed feedback");
                
                $moreAdjustment = SmartRadioService::getMoreWeightAdjustment();
                $lessAdjustment = SmartRadioService::getLessWeightAdjustment();
                
                // Record "more" feedbacks
                for ($i = 0; $i < $numMore; $i++) {
                    $this->radioService->recordFeedback($track->id, 'more');
                }
                
                // Record "less" feedbacks
                for ($i = 0; $i < $numLess; $i++) {
                    $this->radioService->recordFeedback($track->id, 'less');
                }
                
                // Calculate expected weight
                $expectedWeight = ($numMore * $moreAdjustment) + ($numLess * $lessAdjustment);
                
                // Verify final weight
                $finalWeight = $this->radioService->getFeedbackWeight($track->id);
                $this->assertEquals(
                    $expectedWeight,
                    $finalWeight,
                    "Weight should be sum of all adjustments: {$numMore} more + {$numLess} less = {$expectedWeight}"
                );
                
                // Verify database has all feedback records
                $totalFeedbacks = RadioFeedback::where('track_id', $track->id)->count();
                $this->assertEquals(
                    $numMore + $numLess,
                    $totalFeedbacks,
                    "Database should have all feedback records"
                );
                
                // Verify counts by type
                $moreCount = RadioFeedback::where('track_id', $track->id)
                    ->where('feedback_type', 'more')
                    ->count();
                $lessCount = RadioFeedback::where('track_id', $track->id)
                    ->where('feedback_type', 'less')
                    ->count();
                
                $this->assertEquals($numMore, $moreCount);
                $this->assertEquals($numLess, $lessCount);
            });
    }

    /**
     * @test
     * 
     * Property: Feedback for different tracks is independent
     * Validates: Requirements 17.1, 17.2
     */
    public function feedback_is_track_independent(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\choose(2, 5), // number of tracks
                Generator\choose(1, 3)  // feedbacks per track
            )
            ->then(function ($numTracks, $feedbacksPerTrack) {
                // Reset state for each iteration
                $this->radioService->resetState();
                Cache::flush();
                
                // Create tracks
                $tracks = [];
                for ($i = 0; $i < $numTracks; $i++) {
                    $tracks[] = $this->createTrack("Track {$i}");
                }
                
                $moreAdjustment = SmartRadioService::getMoreWeightAdjustment();
                
                // Record different amounts of feedback for each track
                foreach ($tracks as $index => $track) {
                    // Each track gets (index + 1) * feedbacksPerTrack "more" feedbacks
                    $numFeedbacks = ($index + 1) * $feedbacksPerTrack;
                    for ($i = 0; $i < $numFeedbacks; $i++) {
                        $this->radioService->recordFeedback($track->id, 'more');
                    }
                }
                
                // Verify each track has independent weight
                foreach ($tracks as $index => $track) {
                    $expectedFeedbacks = ($index + 1) * $feedbacksPerTrack;
                    $expectedWeight = $expectedFeedbacks * $moreAdjustment;
                    
                    $actualWeight = $this->radioService->getFeedbackWeight($track->id);
                    $this->assertEquals(
                        $expectedWeight,
                        $actualWeight,
                        "Track {$index} should have weight {$expectedWeight}"
                    );
                    
                    // Verify database count
                    $dbCount = RadioFeedback::where('track_id', $track->id)->count();
                    $this->assertEquals(
                        $expectedFeedbacks,
                        $dbCount,
                        "Track {$index} should have {$expectedFeedbacks} feedback records"
                    );
                }
            });
    }

    /**
     * @test
     * 
     * Property: Invalid feedback type throws exception
     * Validates: Input validation
     */
    public function invalid_feedback_type_throws_exception(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\suchThat(
                    fn($s) => !in_array($s, ['more', 'less']),
                    Generator\string()
                )
            )
            ->then(function ($invalidType) {
                // Reset state for each iteration
                $this->radioService->resetState();
                Cache::flush();
                
                // Create a track
                $track = $this->createTrack("Track for invalid feedback");
                
                // Attempt to record invalid feedback
                $this->expectException(\InvalidArgumentException::class);
                $this->radioService->recordFeedback($track->id, $invalidType);
            });
    }

    /**
     * @test
     * 
     * Property: Feedback for non-existent track throws exception
     * Validates: Input validation
     */
    public function feedback_for_nonexistent_track_throws_exception(): void
    {
        $this
            ->limitTo(50)
            ->forAll(
                Generator\choose(99999, 999999), // non-existent track IDs
                Generator\elements(['more', 'less'])
            )
            ->then(function ($nonExistentId, $feedbackType) {
                // Reset state for each iteration
                $this->radioService->resetState();
                Cache::flush();
                
                // Attempt to record feedback for non-existent track
                $this->expectException(\Illuminate\Database\Eloquent\ModelNotFoundException::class);
                $this->radioService->recordFeedback($nonExistentId, $feedbackType);
            });
    }
}
