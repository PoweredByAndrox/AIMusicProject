<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use App\Models\UserTrackPreference;
use App\Services\UserTrackPreferenceService;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use InvalidArgumentException;
use Tests\TestCase;

/**
 * Property Test: Round-trip for User Track Preferences
 * 
 * @tag Feature: advanced-music-player, Property 5: Round-trip для пользовательских настроек трека
 * **Validates: Requirements 4.1, 4.2, 4.5**
 * 
 * For any track and any settings (volume, position):
 * - Saving a setting followed by retrieval returns the same value
 * - Deleting a track cascades to delete associated preferences
 */
class UserTrackPreferenceRoundTripPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private UserTrackPreferenceService $preferenceService;

    protected function setUp(): void
    {
        parent::setUp();
        $this->preferenceService = new UserTrackPreferenceService();
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
     * Property: Saving volume followed by retrieval returns the same value
     * **Validates: Requirements 4.1**
     * 
     * WHEN пользователь изменяет громкость для трека 
     * THEN THE User_Preference SHALL сохранить настройку громкости
     */
    public function saving_volume_followed_by_retrieval_returns_same_value(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\map(
                    function ($int) {
                        // Generate volume between 0.0 and 1.0 with 2 decimal precision
                        return round($int / 100, 2);
                    },
                    Generator\choose(0, 100)
                )
            )
            ->then(function ($volume) {
                // Create a fresh track for each iteration
                $track = $this->createTrack("Track for volume {$volume}");
                
                // Save the volume preference
                $savedPreference = $this->preferenceService->saveVolume($track->id, $volume);
                
                // Verify the preference was saved correctly
                $this->assertNotNull($savedPreference);
                $this->assertEquals($volume, $savedPreference->volume);
                $this->assertEquals($track->id, $savedPreference->track_id);
                
                // Retrieve the preference
                $retrievedPreference = $this->preferenceService->getPreferences($track->id);
                
                // Verify round-trip: saved value equals retrieved value
                $this->assertNotNull($retrievedPreference);
                $this->assertEquals($volume, $retrievedPreference->volume);
                
                // Verify the preference exists in the database
                $this->assertDatabaseHas('user_track_preferences', [
                    'track_id' => $track->id,
                    'volume' => $volume,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Saving position followed by retrieval returns the same value
     * **Validates: Requirements 4.2**
     * 
     * WHEN пользователь останавливает воспроизведение 
     * THEN THE User_Preference SHALL сохранить текущую позицию
     */
    public function saving_position_followed_by_retrieval_returns_same_value(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\map(
                    function ($int) {
                        // Generate position between 0.0 and 300.0 (track duration) with 2 decimal precision
                        return round($int / 10, 2);
                    },
                    Generator\choose(0, 3000)
                )
            )
            ->then(function ($position) {
                // Create a fresh track with duration >= position
                $duration = max(300.0, $position + 10.0);
                $track = $this->createTrack("Track for position {$position}", $duration);
                
                // Save the position preference
                $savedPreference = $this->preferenceService->savePosition($track->id, $position);
                
                // Verify the preference was saved correctly
                $this->assertNotNull($savedPreference);
                $this->assertEquals($position, $savedPreference->last_position);
                $this->assertEquals($track->id, $savedPreference->track_id);
                
                // Retrieve the preference
                $retrievedPreference = $this->preferenceService->getPreferences($track->id);
                
                // Verify round-trip: saved value equals retrieved value
                $this->assertNotNull($retrievedPreference);
                $this->assertEquals($position, $retrievedPreference->last_position);
                
                // Verify the preference exists in the database
                $this->assertDatabaseHas('user_track_preferences', [
                    'track_id' => $track->id,
                    'last_position' => $position,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Saving both volume and position followed by retrieval returns same values
     * **Validates: Requirements 4.1, 4.2**
     */
    public function saving_both_preferences_followed_by_retrieval_returns_same_values(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\map(
                    function ($int) {
                        return round($int / 100, 2);
                    },
                    Generator\choose(0, 100)
                ),
                Generator\map(
                    function ($int) {
                        return round($int / 10, 2);
                    },
                    Generator\choose(0, 3000)
                )
            )
            ->then(function ($volume, $position) {
                // Create a fresh track with duration >= position
                $duration = max(300.0, $position + 10.0);
                $track = $this->createTrack("Track for volume {$volume} position {$position}", $duration);
                
                // Save both preferences at once
                $savedPreference = $this->preferenceService->savePreferences($track->id, $volume, $position);
                
                // Verify the preferences were saved correctly
                $this->assertNotNull($savedPreference);
                $this->assertEquals($volume, $savedPreference->volume);
                $this->assertEquals($position, $savedPreference->last_position);
                $this->assertEquals($track->id, $savedPreference->track_id);
                
                // Retrieve the preferences
                $retrievedPreference = $this->preferenceService->getPreferences($track->id);
                
                // Verify round-trip: saved values equal retrieved values
                $this->assertNotNull($retrievedPreference);
                $this->assertEquals($volume, $retrievedPreference->volume);
                $this->assertEquals($position, $retrievedPreference->last_position);
            });
    }

    /**
     * @test
     * 
     * Property: Deleting a track cascades to delete associated preferences
     * **Validates: Requirements 4.5**
     * 
     * WHEN трек удаляется THEN THE System SHALL удалить связанные User_Preference
     */
    public function deleting_track_cascades_to_delete_preferences(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\map(
                    function ($int) {
                        return round($int / 100, 2);
                    },
                    Generator\choose(0, 100)
                ),
                Generator\map(
                    function ($int) {
                        return round($int / 10, 2);
                    },
                    Generator\choose(0, 3000)
                )
            )
            ->then(function ($volume, $position) {
                // Create a fresh track with duration >= position
                $duration = max(300.0, $position + 10.0);
                $track = $this->createTrack("Track for cascade delete test", $duration);
                $trackId = $track->id;
                
                // Save preferences
                $this->preferenceService->savePreferences($trackId, $volume, $position);
                
                // Verify preferences exist
                $this->assertDatabaseHas('user_track_preferences', [
                    'track_id' => $trackId,
                ]);
                
                // Delete the track
                $track->delete();
                
                // Verify preferences were cascade deleted
                $this->assertDatabaseMissing('user_track_preferences', [
                    'track_id' => $trackId,
                ]);
                
                // Verify the track is also deleted
                $this->assertDatabaseMissing('tracks', [
                    'id' => $trackId,
                ]);
            });
    }

    /**
     * @test
     * 
     * Property: Invalid volume values (< 0 or > 1) are rejected
     * **Validates: Requirements 4.1 (implicit - volume must be valid)**
     */
    public function invalid_volume_values_are_rejected(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\oneOf(
                    Generator\map(
                        function ($int) {
                            // Negative values
                            return -1.0 * ($int / 100 + 0.01);
                        },
                        Generator\choose(0, 100)
                    ),
                    Generator\map(
                        function ($int) {
                            // Values > 1.0
                            return 1.0 + ($int / 100 + 0.01);
                        },
                        Generator\choose(0, 100)
                    )
                )
            )
            ->then(function ($invalidVolume) {
                // Create a fresh track
                $track = $this->createTrack("Track for invalid volume test");
                
                // Attempt to save invalid volume should throw exception
                $exceptionThrown = false;
                try {
                    $this->preferenceService->saveVolume($track->id, $invalidVolume);
                } catch (InvalidArgumentException $e) {
                    $exceptionThrown = true;
                    $this->assertStringContainsString('between', $e->getMessage());
                }
                
                $this->assertTrue(
                    $exceptionThrown,
                    "InvalidArgumentException should be thrown for volume {$invalidVolume}"
                );
                
                // Verify no preference was saved
                $this->assertNull($this->preferenceService->getPreferences($track->id));
            });
    }

    /**
     * @test
     * 
     * Property: Negative position values are rejected
     * **Validates: Requirements 4.2 (implicit - position must be valid)**
     */
    public function negative_position_values_are_rejected(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\map(
                    function ($int) {
                        // Negative values
                        return -1.0 * ($int / 10 + 0.01);
                    },
                    Generator\choose(0, 1000)
                )
            )
            ->then(function ($invalidPosition) {
                // Create a fresh track
                $track = $this->createTrack("Track for invalid position test");
                
                // Attempt to save invalid position should throw exception
                $exceptionThrown = false;
                try {
                    $this->preferenceService->savePosition($track->id, $invalidPosition);
                } catch (InvalidArgumentException $e) {
                    $exceptionThrown = true;
                    $this->assertStringContainsString('negative', $e->getMessage());
                }
                
                $this->assertTrue(
                    $exceptionThrown,
                    "InvalidArgumentException should be thrown for position {$invalidPosition}"
                );
                
                // Verify no preference was saved
                $this->assertNull($this->preferenceService->getPreferences($track->id));
            });
    }

    /**
     * @test
     * 
     * Property: Position exceeding track duration is rejected
     * **Validates: Requirements 4.2 (implicit - position must be within track)**
     */
    public function position_exceeding_track_duration_is_rejected(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(100, 500),  // Track duration
                Generator\choose(1, 100)      // Extra seconds beyond duration
            )
            ->then(function ($duration, $extra) {
                // Create a track with specific duration
                $track = $this->createTrack("Track for position overflow test", (float) $duration);
                
                // Position that exceeds duration
                $invalidPosition = $duration + $extra;
                
                // Attempt to save invalid position should throw exception
                $exceptionThrown = false;
                try {
                    $this->preferenceService->savePosition($track->id, $invalidPosition);
                } catch (InvalidArgumentException $e) {
                    $exceptionThrown = true;
                    $this->assertStringContainsString('exceed', $e->getMessage());
                }
                
                $this->assertTrue(
                    $exceptionThrown,
                    "InvalidArgumentException should be thrown for position {$invalidPosition} > duration {$duration}"
                );
                
                // Verify no preference was saved
                $this->assertNull($this->preferenceService->getPreferences($track->id));
            });
    }

    /**
     * @test
     * 
     * Property: Updating preferences preserves other values
     * **Validates: Requirements 4.1, 4.2**
     */
    public function updating_one_preference_preserves_other_values(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\map(
                    function ($int) {
                        return round($int / 100, 2);
                    },
                    Generator\choose(0, 100)
                ),
                Generator\map(
                    function ($int) {
                        return round($int / 100, 2);
                    },
                    Generator\choose(0, 100)
                ),
                Generator\map(
                    function ($int) {
                        return round($int / 10, 2);
                    },
                    Generator\choose(0, 2000)
                )
            )
            ->when(function ($volume1, $volume2, $position) {
                // Ensure volumes are different for meaningful test
                return $volume1 !== $volume2;
            })
            ->then(function ($volume1, $volume2, $position) {
                // Create a fresh track
                $duration = max(300.0, $position + 10.0);
                $track = $this->createTrack("Track for update preservation test", $duration);
                
                // Save initial preferences
                $this->preferenceService->savePreferences($track->id, $volume1, $position);
                
                // Update only volume
                $this->preferenceService->saveVolume($track->id, $volume2);
                
                // Retrieve and verify position was preserved
                $preference = $this->preferenceService->getPreferences($track->id);
                $this->assertEquals($volume2, $preference->volume);
                $this->assertEquals($position, $preference->last_position);
            });
    }

    /**
     * @test
     * 
     * Property: Preferences for non-existent track throws ModelNotFoundException
     * **Validates: Requirements 4.4 (implicit - track must exist)**
     */
    public function preferences_for_non_existent_track_throws_exception(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(999999, 9999999))
            ->then(function ($nonExistentTrackId) {
                // Ensure track doesn't exist
                $this->assertDatabaseMissing('tracks', ['id' => $nonExistentTrackId]);
                
                // Attempt to get preferences should throw exception
                $exceptionThrown = false;
                try {
                    $this->preferenceService->getPreferences($nonExistentTrackId);
                } catch (\Illuminate\Database\Eloquent\ModelNotFoundException $e) {
                    $exceptionThrown = true;
                }
                
                $this->assertTrue(
                    $exceptionThrown,
                    "ModelNotFoundException should be thrown for non-existent track {$nonExistentTrackId}"
                );
            });
    }
}
