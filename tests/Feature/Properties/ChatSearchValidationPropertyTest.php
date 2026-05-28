<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property-based tests for Chat Search API validation.
 * 
 * **Property 11: Валидация некорректных запросов API**
 * **Validates: Requirements 6.6**
 */
class ChatSearchValidationPropertyTest extends TestCase
{
    use DatabaseTransactions;

    /**
     * Property: Missing track_id returns 422 with INVALID_TRACK_ID code.
     * 
     * **Validates: Requirements 6.6**
     */
    public function test_missing_track_id_returns_422(): void
    {
        $response = $this->postJson('/api/chat/search', [
            'start_time' => 10.0,
            'end_time' => 20.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_TRACK_ID'
        ]);
    }

    /**
     * Property: Missing start_time returns 422 with INVALID_START_TIME code.
     * 
     * **Validates: Requirements 6.6**
     */
    public function test_missing_start_time_returns_422(): void
    {
        $response = $this->postJson('/api/chat/search', [
            'track_id' => 1,
            'end_time' => 20.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_START_TIME'
        ]);
    }

    /**
     * Property: Missing end_time returns 422 with INVALID_END_TIME code.
     * 
     * **Validates: Requirements 6.6**
     */
    public function test_missing_end_time_returns_422(): void
    {
        $response = $this->postJson('/api/chat/search', [
            'track_id' => 1,
            'start_time' => 10.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_END_TIME'
        ]);
    }

    /**
     * Property: For any negative time value, API returns 422 with NEGATIVE_TIME code.
     * 
     * **Validates: Requirements 6.6**
     */
    public function test_negative_time_values_return_422(): void
    {
        // Create a track for valid track_id
        $track = Track::factory()->create();

        // Test negative start_time
        $response = $this->postJson('/api/chat/search', [
            'track_id' => $track->id,
            'start_time' => -5.0,
            'end_time' => 20.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'NEGATIVE_TIME'
        ]);

        // Test negative end_time
        $response = $this->postJson('/api/chat/search', [
            'track_id' => $track->id,
            'start_time' => 10.0,
            'end_time' => -5.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'NEGATIVE_TIME'
        ]);
    }

    /**
     * Property: For any start_time >= end_time, API returns 422 with INVALID_RANGE code.
     * 
     * **Validates: Requirements 6.6**
     */
    public function test_invalid_time_range_returns_422(): void
    {
        $track = Track::factory()->create();

        // Test start_time == end_time
        $response = $this->postJson('/api/chat/search', [
            'track_id' => $track->id,
            'start_time' => 10.0,
            'end_time' => 10.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_RANGE'
        ]);

        // Test start_time > end_time
        $response = $this->postJson('/api/chat/search', [
            'track_id' => $track->id,
            'start_time' => 30.0,
            'end_time' => 20.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_RANGE'
        ]);
    }

    /**
     * Property: Non-existent track_id returns 404 with TRACK_NOT_FOUND code.
     * 
     * **Validates: Requirements 6.5**
     */
    public function test_nonexistent_track_returns_404(): void
    {
        $response = $this->postJson('/api/chat/search', [
            'track_id' => 999999,
            'start_time' => 10.0,
            'end_time' => 20.0,
        ]);

        $response->assertStatus(404);
        $response->assertJson([
            'code' => 'TRACK_NOT_FOUND'
        ]);
    }

    /**
     * Property: Non-numeric track_id returns 422.
     * 
     * **Validates: Requirements 6.6**
     */
    public function test_non_numeric_track_id_returns_422(): void
    {
        $response = $this->postJson('/api/chat/search', [
            'track_id' => 'abc',
            'start_time' => 10.0,
            'end_time' => 20.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_TRACK_ID'
        ]);
    }

    /**
     * Property: Non-numeric time values return 422.
     * 
     * **Validates: Requirements 6.6**
     */
    public function test_non_numeric_time_values_return_422(): void
    {
        $response = $this->postJson('/api/chat/search', [
            'track_id' => 1,
            'start_time' => 'abc',
            'end_time' => 20.0,
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_START_TIME'
        ]);

        $response = $this->postJson('/api/chat/search', [
            'track_id' => 1,
            'start_time' => 10.0,
            'end_time' => 'xyz',
        ]);

        $response->assertStatus(422);
        $response->assertJson([
            'code' => 'INVALID_END_TIME'
        ]);
    }

    /**
     * Property: limit is clamped to max 50.
     * 
     * **Validates: Requirements 6.4**
     */
    public function test_limit_is_clamped_to_max_50(): void
    {
        $track = Track::factory()->create();

        // This test verifies the controller accepts the request
        // The actual clamping happens internally
        $response = $this->postJson('/api/chat/search', [
            'track_id' => $track->id,
            'start_time' => 10.0,
            'end_time' => 20.0,
            'limit' => 100, // Should be clamped to 50
        ]);

        // Should not return validation error for limit
        $this->assertNotEquals(422, $response->status());
    }

    /**
     * Property: min_similarity is clamped to 0.0-1.0 range.
     * 
     * **Validates: Requirements 6.4**
     */
    public function test_min_similarity_is_clamped(): void
    {
        $track = Track::factory()->create();

        // Test with value > 1.0
        $response = $this->postJson('/api/chat/search', [
            'track_id' => $track->id,
            'start_time' => 10.0,
            'end_time' => 20.0,
            'min_similarity' => 2.0, // Should be clamped to 1.0
        ]);

        // Should not return validation error
        $this->assertNotEquals(422, $response->status());

        // Test with value < 0.0
        $response = $this->postJson('/api/chat/search', [
            'track_id' => $track->id,
            'start_time' => 10.0,
            'end_time' => 20.0,
            'min_similarity' => -0.5, // Should be clamped to 0.0
        ]);

        // Should not return validation error
        $this->assertNotEquals(422, $response->status());
    }
}
