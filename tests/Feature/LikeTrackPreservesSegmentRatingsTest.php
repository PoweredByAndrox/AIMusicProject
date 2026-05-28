<?php

namespace Tests\Feature;

use App\Models\ApiToken;
use App\Models\Pattern;
use App\Models\PatternExample;
use App\Models\Track;
use App\Models\TrackRating;
use App\Models\User;
use App\Models\UserPatternExampleRating;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Support\Facades\Queue;
use Tests\TestCase;

class LikeTrackPreservesSegmentRatingsTest extends TestCase
{
    use DatabaseTransactions;

    public function test_liking_track_does_not_overwrite_existing_or_unrated_segments(): void
    {
        Queue::fake();

        [$user, $plainToken] = $this->createUserWithToken();
        $track = Track::factory()->completed()->create([
            'duration' => 180,
        ]);
        $pattern = Pattern::create([
            'name' => 'verse',
            'representation_type' => 'chroma',
            'transpose_invariant' => true,
            'tempo_invariant' => true,
            'match_threshold' => 0.8,
            'importance' => 2,
        ]);

        $ratedSegment = PatternExample::create([
            'pattern_id' => $pattern->id,
            'track_id' => $track->id,
            'start_time' => 12.0,
            'end_time' => 24.0,
            'is_positive' => false,
        ]);
        $unratedSegment = PatternExample::create([
            'pattern_id' => $pattern->id,
            'track_id' => $track->id,
            'start_time' => 30.0,
            'end_time' => 42.0,
            'is_positive' => true,
        ]);

        UserPatternExampleRating::create([
            'user_id' => $user->id,
            'pattern_example_id' => $ratedSegment->id,
            'rating' => 1.0,
            'is_positive' => false,
        ]);

        $response = $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->postJson("/api/tracks/{$track->id}/like");

        $response->assertOk()
            ->assertJsonPath('is_liked', true)
            ->assertJsonPath('explicit_track_rating', 5)
            ->assertJsonPath('feedback.track_rating', 5)
            ->assertJsonPath('feedback.explicit_track_rating', 5);

        $segmentRatings = UserPatternExampleRating::query()
            ->where('user_id', $user->id)
            ->whereIn('pattern_example_id', [$ratedSegment->id, $unratedSegment->id])
            ->orderBy('pattern_example_id')
            ->get();

        $this->assertCount(1, $segmentRatings);
        $this->assertSame($ratedSegment->id, (int) $segmentRatings->first()->pattern_example_id);
        $this->assertSame(1.0, (float) $segmentRatings->first()->rating);
        $this->assertFalse((bool) $segmentRatings->first()->is_positive);

        $feedbackSegments = collect($response->json('feedback.segments'))
            ->keyBy(fn (array $segment): int => (int) $segment['id']);

        $this->assertSame(1.0, (float) $feedbackSegments[$ratedSegment->id]['rating']);
        $this->assertNull($feedbackSegments[$unratedSegment->id]['rating']);

        $this->assertDatabaseHas('track_ratings', [
            'user_id' => $user->id,
            'track_id' => $track->id,
            'rating' => 5,
        ]);

        $this->assertSame(
            5,
            (int) TrackRating::query()
                ->where('user_id', $user->id)
                ->where('track_id', $track->id)
                ->value('rating')
        );
    }

    /**
     * @return array{0: User, 1: string}
     */
    private function createUserWithToken(): array
    {
        $user = User::create([
            'name' => 'Segment Test User',
            'email' => 'segment-test-' . uniqid('', true) . '@example.com',
            'password' => 'secret-password',
            'role' => User::ROLE_USER,
        ]);

        $plainToken = bin2hex(random_bytes(20));

        ApiToken::create([
            'user_id' => $user->id,
            'name' => 'test-token',
            'token_hash' => hash('sha256', $plainToken),
            'created_at' => now(),
        ]);

        return [$user, $plainToken];
    }
}
