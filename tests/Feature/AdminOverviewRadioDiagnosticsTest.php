<?php

namespace Tests\Feature;

use App\Models\ApiToken;
use App\Models\User;
use Illuminate\Support\Facades\File;
use Tests\TestCase;

class AdminOverviewRadioDiagnosticsTest extends TestCase
{
    public function test_admin_overview_includes_recent_radio_diagnostics_summary(): void
    {
        [, $plainToken] = $this->createAdminWithToken();
        $logPath = storage_path('logs/public-radio-diagnostics-codex-test.log');

        File::ensureDirectoryExists(dirname($logPath));
        File::put($logPath, implode(PHP_EOL, [
            json_encode([
                'message' => 'public-radio-client-diagnostics',
                'datetime' => '2099-03-26T10:10:10+00:00',
                'context' => [
                    'station_slug' => 'radio-test-station',
                    'viewer_id' => 1,
                    'client_id' => 'listener-alpha',
                    'client_ip' => '127.0.0.1',
                    'user_agent' => 'Test Browser',
                    'event_count' => 2,
                    'events' => [
                        [
                            'event' => 'waiting',
                            'timestamp' => '2099-03-26T10:10:09+00:00',
                            'playback_mode' => 'direct',
                            'transport_preference' => 'direct',
                            'selected_bitrate' => '64',
                            'track_id' => 14054,
                            'drift' => 0.8,
                            'playback_rate' => 1.04,
                            'media_error_code' => null,
                            'waiting' => true,
                            'paused' => false,
                            'ready_state' => 2,
                            'current_src' => '/api/tracks/14054/stream?bitrate=64',
                        ],
                        [
                            'event' => 'native-hls-fallback-armed',
                            'timestamp' => '2099-03-26T10:10:10+00:00',
                            'playback_mode' => 'hls-native',
                            'transport_preference' => 'direct',
                            'selected_bitrate' => '64',
                            'track_id' => 14054,
                            'drift' => 1.2,
                            'playback_rate' => 1.0,
                            'media_error_code' => 4,
                            'waiting' => false,
                            'paused' => false,
                            'ready_state' => 1,
                            'current_src' => '/api/tracks/14054/hls/64k/index.m3u8',
                        ],
                    ],
                ],
            ], JSON_UNESCAPED_SLASHES),
            json_encode([
                'message' => 'public-radio-client-diagnostics-throttled',
                'datetime' => '2099-03-26T10:10:11+00:00',
                'context' => [
                    'station_slug' => 'radio-test-station',
                    'viewer_id' => 1,
                    'client_id' => 'listener-alpha',
                    'accepted_count' => 1,
                    'dropped_count' => 3,
                    'limit_per_minute' => 240,
                ],
            ], JSON_UNESCAPED_SLASHES),
        ]) . PHP_EOL);

        try {
            $response = $this
                ->withHeaders(['Authorization' => 'Bearer ' . $plainToken])
                ->getJson('/api/admin/overview');

            $response->assertOk()
                ->assertJsonPath('radio_diagnostics.available', true)
                ->assertJsonPath('radio_diagnostics.dropped_events', 3)
                ->assertJsonPath('radio_diagnostics.fallback_events', 1)
                ->assertJsonPath('radio_diagnostics.waiting_events', 1)
                ->assertJsonPath('radio_diagnostics.recent_events.0.station_slug', 'radio-test-station');
        } finally {
            File::delete($logPath);
        }
    }

    /**
     * @return array{0: User, 1: string}
     */
    private function createAdminWithToken(): array
    {
        $user = User::create([
            'name' => 'Admin Overview Tester',
            'email' => 'admin-overview-' . uniqid('', true) . '@example.com',
            'password' => 'secret-password',
            'role' => User::ROLE_ADMIN,
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
