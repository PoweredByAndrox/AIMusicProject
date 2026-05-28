<?php

namespace Tests\Feature;

use App\Models\ApiToken;
use App\Models\Friendship;
use App\Models\Track;
use App\Models\User;
use App\Models\UserTrackLike;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

class TrackFriendLikesTest extends TestCase
{
    use DatabaseTransactions;

    public function test_track_detail_includes_likes_from_accepted_friends_only(): void
    {
        [$viewer, $plainToken] = $this->createUserWithToken();
        $acceptedFriend = User::create([
            'name' => 'Accepted Friend',
            'email' => 'friend-' . uniqid('', true) . '@example.com',
            'password' => 'secret-password',
            'role' => User::ROLE_USER,
        ]);
        $pendingFriend = User::create([
            'name' => 'Pending Friend',
            'email' => 'pending-' . uniqid('', true) . '@example.com',
            'password' => 'secret-password',
            'role' => User::ROLE_USER,
        ]);
        $stranger = User::create([
            'name' => 'Stranger',
            'email' => 'stranger-' . uniqid('', true) . '@example.com',
            'password' => 'secret-password',
            'role' => User::ROLE_USER,
        ]);

        Friendship::create([
            'requester_id' => $viewer->id,
            'addressee_id' => $acceptedFriend->id,
            'status' => Friendship::STATUS_ACCEPTED,
            'accepted_at' => now(),
            'responded_at' => now(),
        ]);

        Friendship::create([
            'requester_id' => $pendingFriend->id,
            'addressee_id' => $viewer->id,
            'status' => Friendship::STATUS_PENDING,
        ]);

        $track = Track::factory()->completed()->create();

        UserTrackLike::create([
            'user_id' => $acceptedFriend->id,
            'track_id' => $track->id,
        ]);
        UserTrackLike::create([
            'user_id' => $pendingFriend->id,
            'track_id' => $track->id,
        ]);
        UserTrackLike::create([
            'user_id' => $stranger->id,
            'track_id' => $track->id,
        ]);

        $response = $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->getJson("/api/tracks/{$track->id}");

        $response->assertOk()
            ->assertJsonCount(1, 'friend_likes')
            ->assertJsonPath('friend_likes.0.id', $acceptedFriend->id)
            ->assertJsonPath('friend_likes.0.name', 'Accepted Friend');

        $friendLikes = $response->json('friend_likes');
        $this->assertIsArray($friendLikes);
        $this->assertNotEmpty($friendLikes[0]['liked_at']);
    }

    /**
     * @return array{0: User, 1: string}
     */
    private function createUserWithToken(): array
    {
        $user = User::create([
            'name' => 'Viewer User',
            'email' => 'viewer-' . uniqid('', true) . '@example.com',
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
