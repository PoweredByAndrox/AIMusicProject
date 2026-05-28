<?php

namespace Tests\Feature;

use App\Models\ApiToken;
use App\Models\Track;
use App\Models\User;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Support\Facades\Artisan;
use Tests\TestCase;

class ArtistPageTest extends TestCase
{
    use DatabaseTransactions;

    public function test_artist_endpoint_groups_primary_featured_and_remix_tracks(): void
    {
        [, $plainToken] = $this->createUserWithToken();

        Track::factory()->completed()->create([
            'title' => 'Skyline',
            'artist' => 'DJ Alpha',
            'artist_credits' => [
                ['name' => 'DJ Alpha', 'slug' => 'dj-alpha', 'roles' => ['primary']],
            ],
            'artist_slugs' => ['dj-alpha'],
        ]);

        Track::factory()->completed()->create([
            'title' => 'Guest Verse',
            'artist' => 'Other Artist ft. DJ Alpha',
            'artist_credits' => [
                ['name' => 'Other Artist', 'slug' => 'other-artist', 'roles' => ['primary']],
                ['name' => 'DJ Alpha', 'slug' => 'dj-alpha', 'roles' => ['featured']],
            ],
            'artist_slugs' => ['other-artist', 'dj-alpha'],
        ]);

        Track::factory()->completed()->create([
            'title' => 'Anthem (DJ Alpha Remix)',
            'artist' => 'Gamma',
            'artist_credits' => [
                ['name' => 'Gamma', 'slug' => 'gamma', 'roles' => ['primary']],
                ['name' => 'DJ Alpha', 'slug' => 'dj-alpha', 'roles' => ['remixer']],
            ],
            'artist_slugs' => ['gamma', 'dj-alpha'],
        ]);

        $response = $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->getJson('/api/artists/dj-alpha');

        $response->assertOk()
            ->assertJsonPath('artist.name', 'DJ Alpha')
            ->assertJsonCount(1, 'primary_tracks')
            ->assertJsonCount(1, 'featured_tracks')
            ->assertJsonCount(1, 'remix_tracks')
            ->assertJsonPath('primary_tracks.0.title', 'Skyline')
            ->assertJsonPath('featured_tracks.0.title', 'Guest Verse')
            ->assertJsonPath('remix_tracks.0.title', 'Anthem (DJ Alpha Remix)');
    }

    public function test_artist_endpoint_finds_backfilled_existing_tracks_by_artist_slug(): void
    {
        [, $plainToken] = $this->createUserWithToken();

        Track::factory()->completed()->create([
            'source' => 'upload',
            'title' => 'Skyline',
            'artist' => 'DJ Alpha feat. Beta',
            'artist_credits' => null,
            'artist_slugs' => null,
        ]);

        Track::factory()->completed()->create([
            'source' => 'upload',
            'title' => 'Guest Verse',
            'artist' => 'Other Artist feat. DJ Alpha',
            'artist_credits' => null,
            'artist_slugs' => null,
        ]);

        Track::factory()->completed()->create([
            'source' => 'youtube',
            'title' => 'Gamma - Anthem (DJ Alpha Remix)',
            'artist' => null,
            'youtube_channel' => 'Gamma',
            'artist_credits' => null,
            'artist_slugs' => null,
        ]);

        Artisan::call('tracks:reparse-artists');

        $response = $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->getJson('/api/artists/dj-alpha');

        $response->assertOk()
            ->assertJsonPath('artist.name', 'DJ Alpha')
            ->assertJsonCount(1, 'primary_tracks')
            ->assertJsonCount(1, 'featured_tracks')
            ->assertJsonCount(1, 'remix_tracks');
    }

    public function test_admin_can_rename_artist_across_tracks(): void
    {
        [, $plainToken] = $this->createUserWithToken(User::ROLE_ADMIN);

        $primaryTrack = Track::factory()->completed()->create([
            'title' => 'Skyline',
            'artist' => 'DJ Alpha',
            'artist_credits' => [
                ['name' => 'DJ Alpha', 'slug' => 'dj-alpha', 'roles' => ['primary']],
            ],
            'artist_slugs' => ['dj-alpha'],
        ]);

        $remixTrack = Track::factory()->completed()->create([
            'title' => 'Anthem (DJ Alpha Remix)',
            'artist' => 'Gamma',
            'artist_credits' => [
                ['name' => 'Gamma', 'slug' => 'gamma', 'roles' => ['primary']],
                ['name' => 'DJ Alpha', 'slug' => 'dj-alpha', 'roles' => ['remixer']],
            ],
            'artist_slugs' => ['gamma', 'dj-alpha'],
        ]);

        $response = $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->putJson('/api/artists/dj-alpha', [
                'name' => 'DJ Omega',
            ]);

        $response->assertOk()
            ->assertJsonPath('artist.name', 'DJ Omega')
            ->assertJsonPath('artist.slug', 'dj-omega')
            ->assertJsonPath('updated_tracks', 2);

        $primaryTrack->refresh();
        $remixTrack->refresh();

        $this->assertSame('DJ Omega', $primaryTrack->artist_credits[0]['name']);
        $this->assertSame('dj-omega', $primaryTrack->artist_credits[0]['slug']);
        $this->assertSame(['dj-omega'], $primaryTrack->artist_slugs);

        $this->assertSame('DJ Omega', $remixTrack->artist_credits[1]['name']);
        $this->assertSame('dj-omega', $remixTrack->artist_credits[1]['slug']);
        $this->assertSame(['gamma', 'dj-omega'], $remixTrack->artist_slugs);

        $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->getJson('/api/artists/dj-omega')
            ->assertOk()
            ->assertJsonPath('artist.name', 'DJ Omega')
            ->assertJsonCount(1, 'primary_tracks')
            ->assertJsonCount(1, 'remix_tracks');

        $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->getJson('/api/artists/dj-alpha')
            ->assertNotFound();
    }

    public function test_non_admin_cannot_rename_artist(): void
    {
        [, $plainToken] = $this->createUserWithToken();

        Track::factory()->completed()->create([
            'title' => 'Skyline',
            'artist' => 'DJ Alpha',
            'artist_credits' => [
                ['name' => 'DJ Alpha', 'slug' => 'dj-alpha', 'roles' => ['primary']],
            ],
            'artist_slugs' => ['dj-alpha'],
        ]);

        $this
            ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
            ->putJson('/api/artists/dj-alpha', [
                'name' => 'DJ Omega',
            ])
            ->assertForbidden();
    }

    /**
     * @return array{0: User, 1: string}
     */
    private function createUserWithToken(string $role = User::ROLE_USER): array
    {
        $user = User::create([
            'name' => 'Artist Test User',
            'email' => 'artist-test-' . uniqid('', true) . '@example.com',
            'password' => 'secret-password',
            'role' => $role,
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
