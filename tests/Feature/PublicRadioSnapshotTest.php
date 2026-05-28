<?php

namespace Tests\Feature;

use App\Jobs\TrackHlsPrewarmJob;
use App\Jobs\TrackWaveformPrewarmJob;
use App\Models\Track;
use Illuminate\Foundation\Testing\RefreshDatabase;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Queue;
use Illuminate\Support\Facades\Storage;
use Tests\TestCase;

class PublicRadioSnapshotTest extends TestCase
{
    use RefreshDatabase;

    public function test_public_radio_snapshot_does_not_fail_when_queue_prewarm_runs(): void
    {
        Cache::flush();
        Queue::fake();
        Storage::fake('local');

        $track = Track::factory()->completed()->create([
            'title' => 'Radio Snapshot Track',
            'artist' => 'Codex',
            'duration' => 180,
            'file_path' => 'audio/radio-snapshot-track.mp3',
            'genre' => 'electronic',
        ]);

        Storage::disk('local')->put($track->file_path, 'fake-audio');

        $response = $this->getJson('/api/public/radio/test');

        $response->assertOk()
            ->assertJsonPath('station.slug', 'test')
            ->assertJsonPath('listener_count', 0)
            ->assertJsonPath('now_playing.track_id', $track->id);

        Queue::assertPushed(TrackWaveformPrewarmJob::class, function (TrackWaveformPrewarmJob $job) use ($track): bool {
            return $job->trackId === $track->id;
        });

        Queue::assertPushed(TrackHlsPrewarmJob::class, function (TrackHlsPrewarmJob $job) use ($track): bool {
            return $job->trackId === $track->id;
        });
    }
}
