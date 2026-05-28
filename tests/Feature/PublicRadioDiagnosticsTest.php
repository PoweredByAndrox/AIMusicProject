<?php

namespace Tests\Feature;

use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;
use Tests\TestCase;

class PublicRadioDiagnosticsTest extends TestCase
{
    public function test_public_radio_diagnostics_endpoint_accepts_batched_events_and_logs_them(): void
    {
        Log::shouldReceive('channel')
            ->once()
            ->with('public_radio_diagnostics')
            ->andReturnSelf();

        Log::shouldReceive('info')
            ->once()
            ->withArgs(function (string $message, array $context): bool {
                return $message === 'public-radio-client-diagnostics'
                    && $context['station_slug'] === 'test'
                    && $context['event_count'] === 2
                    && $context['events'][0]['event'] === 'waiting'
                    && $context['events'][1]['event'] === 'recovery:waiting';
            });

        $response = $this->postJson('/api/public/radio/test/diagnostics', [
            'client_id' => 'listener-test-1',
            'events' => [
                [
                    'event' => 'waiting',
                    'timestamp' => now()->toIso8601String(),
                    'playbackMode' => 'direct',
                    'transportPreference' => 'direct',
                    'selectedBitrate' => '64',
                    'trackId' => 14054,
                    'currentSrc' => 'http://localhost/api/tracks/14054/stream?bitrate=64',
                    'streamUrl' => 'http://localhost/api/tracks/14054/stream?bitrate=64',
                    'hlsSourceUrl' => '',
                    'currentTime' => 31.2,
                    'liveOffset' => 32.0,
                    'drift' => 0.8,
                    'playbackRate' => 1.04,
                    'mediaErrorCode' => null,
                    'waiting' => true,
                    'paused' => false,
                    'readyState' => 2,
                ],
                [
                    'event' => 'recovery:waiting',
                    'timestamp' => now()->addSecond()->toIso8601String(),
                    'playbackMode' => 'direct',
                    'transportPreference' => 'direct',
                    'selectedBitrate' => '64',
                    'trackId' => 14054,
                    'currentSrc' => 'http://localhost/api/tracks/14054/stream?bitrate=64',
                    'streamUrl' => 'http://localhost/api/tracks/14054/stream?bitrate=64',
                    'hlsSourceUrl' => '',
                    'currentTime' => 32.1,
                    'liveOffset' => 32.4,
                    'drift' => 0.3,
                    'playbackRate' => 1.0,
                    'mediaErrorCode' => null,
                    'waiting' => false,
                    'paused' => false,
                    'readyState' => 4,
                ],
            ],
        ]);

        $response->assertOk()
            ->assertJsonPath('accepted', 2)
            ->assertJsonPath('dropped', 0);
    }

    public function test_public_radio_diagnostics_endpoint_throttles_excess_events_per_client(): void
    {
        $rateKey = 'public_radio_diagnostics_rate:' . sha1('test|listener-test-2|' . now()->format('YmdHi'));
        Cache::put($rateKey, 239, now()->addMinutes(2));

        Log::shouldReceive('channel')
            ->twice()
            ->with('public_radio_diagnostics')
            ->andReturnSelf();

        Log::shouldReceive('info')
            ->once()
            ->withArgs(function (string $message, array $context): bool {
                return $message === 'public-radio-client-diagnostics'
                    && $context['client_id'] === 'listener-test-2'
                    && $context['event_count'] === 1;
            });

        Log::shouldReceive('warning')
            ->once()
            ->withArgs(function (string $message, array $context): bool {
                return $message === 'public-radio-client-diagnostics-throttled'
                    && $context['client_id'] === 'listener-test-2'
                    && $context['accepted_count'] === 1
                    && $context['dropped_count'] === 1;
            });

        $response = $this->postJson('/api/public/radio/test/diagnostics', [
            'client_id' => 'listener-test-2',
            'events' => [
                [
                    'event' => 'waiting',
                    'timestamp' => now()->toIso8601String(),
                ],
                [
                    'event' => 'stalled',
                    'timestamp' => now()->addSecond()->toIso8601String(),
                ],
            ],
        ]);

        $response->assertOk()
            ->assertJsonPath('accepted', 1)
            ->assertJsonPath('dropped', 1);
    }
}
