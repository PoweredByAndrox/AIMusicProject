<?php

namespace Tests\Feature;

use App\Models\Fragment;
use App\Models\Track;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

class TrackBatchDeleteTest extends TestCase
{
    use DatabaseTransactions;

    /**
     * @test
     */
    public function it_deletes_multiple_tracks_in_one_request(): void
    {
        $tracks = Track::factory()->count(2)->create();

        foreach ($tracks as $index => $track) {
            Fragment::create([
                'track_id' => $track->id,
                'start_time' => $index * 10.0,
                'end_time' => ($index + 1) * 10.0,
            ]);
        }

        $response = $this->deleteJson('/api/tracks/batch', [
            'ids' => $tracks->pluck('id')->all(),
        ]);

        $response
            ->assertOk()
            ->assertJson([
                'message' => 'Tracks deleted successfully',
                'deleted_count' => 2,
                'deleted_ids' => $tracks->pluck('id')->all(),
            ]);

        foreach ($tracks as $track) {
            $this->assertDatabaseMissing('tracks', ['id' => $track->id]);
            $this->assertDatabaseMissing('fragments', ['track_id' => $track->id]);
        }
    }
}
