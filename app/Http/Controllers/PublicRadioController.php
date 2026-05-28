<?php

namespace App\Http\Controllers;

use App\Services\PersonalRadioAccessService;
use App\Services\PublicRadioStationService;
use App\Services\RadioAccessTokenService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Http\Exceptions\HttpResponseException;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;
use InvalidArgumentException;

class PublicRadioController extends Controller
{
    private const MAX_DIAGNOSTIC_EVENTS_PER_MINUTE = 240;

    public function __construct(
        private readonly PublicRadioStationService $stations,
        private readonly PersonalRadioAccessService $personalRadioAccess,
        private readonly RadioAccessTokenService $radioAccessTokens,
    ) {}

    public function show(Request $request, string $slug): JsonResponse
    {
        try {
            $viewer = $this->authorizePersonalStation($request, $slug);

            return response()->json($this->stations->snapshot($slug, $viewer));
        } catch (InvalidArgumentException) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Radio station not found',
            ], 404);
        }
    }

    public function history(Request $request, string $slug): JsonResponse
    {
        try {
            $viewer = $this->authorizePersonalStation($request, $slug);

            return response()->json($this->stations->history($slug, $viewer));
        } catch (InvalidArgumentException) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Radio station not found',
            ], 404);
        }
    }

    public function diagnostics(Request $request, string $slug): JsonResponse
    {
        try {
            $viewer = $this->authorizePersonalStation($request, $slug);
        } catch (InvalidArgumentException) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Radio station not found',
            ], 404);
        }

        $validated = $request->validate([
            'client_id' => ['required', 'string', 'max:120'],
            'events' => ['required', 'array', 'min:1', 'max:20'],
            'events.*.event' => ['required', 'string', 'max:120'],
            'events.*.timestamp' => ['required', 'string', 'max:80'],
            'events.*.playbackMode' => ['nullable', 'string', 'max:40'],
            'events.*.transportPreference' => ['nullable', 'string', 'max:20'],
            'events.*.selectedBitrate' => ['nullable', 'string', 'max:20'],
            'events.*.trackId' => ['nullable', 'integer'],
            'events.*.currentSrc' => ['nullable', 'string', 'max:2048'],
            'events.*.streamUrl' => ['nullable', 'string', 'max:2048'],
            'events.*.hlsSourceUrl' => ['nullable', 'string', 'max:2048'],
            'events.*.currentTime' => ['nullable', 'numeric'],
            'events.*.liveOffset' => ['nullable', 'numeric'],
            'events.*.drift' => ['nullable', 'numeric'],
            'events.*.playbackRate' => ['nullable', 'numeric'],
            'events.*.mediaErrorCode' => ['nullable', 'integer'],
            'events.*.waiting' => ['nullable', 'boolean'],
            'events.*.paused' => ['nullable', 'boolean'],
            'events.*.readyState' => ['nullable', 'integer'],
        ]);

        $clientId = trim((string) $validated['client_id']);
        $rateKey = $this->diagnosticsRateLimitKey($slug, $clientId);
        $currentAccepted = (int) Cache::get($rateKey, 0);
        $remainingCapacity = max(0, self::MAX_DIAGNOSTIC_EVENTS_PER_MINUTE - $currentAccepted);
        $acceptedCount = min(count($validated['events']), $remainingCapacity);
        $acceptedInput = array_slice($validated['events'], 0, $acceptedCount);
        $droppedCount = count($validated['events']) - $acceptedCount;

        if ($acceptedCount > 0) {
            Cache::put($rateKey, $currentAccepted + $acceptedCount, now()->addMinutes(2));
        }

        $events = array_map(static function (array $event): array {
            return [
                'event' => (string) ($event['event'] ?? ''),
                'timestamp' => (string) ($event['timestamp'] ?? ''),
                'playback_mode' => isset($event['playbackMode']) ? (string) $event['playbackMode'] : null,
                'transport_preference' => isset($event['transportPreference']) ? (string) $event['transportPreference'] : null,
                'selected_bitrate' => isset($event['selectedBitrate']) ? (string) $event['selectedBitrate'] : null,
                'track_id' => isset($event['trackId']) ? (int) $event['trackId'] : null,
                'current_src' => isset($event['currentSrc']) ? (string) $event['currentSrc'] : null,
                'stream_url' => isset($event['streamUrl']) ? (string) $event['streamUrl'] : null,
                'hls_source_url' => isset($event['hlsSourceUrl']) ? (string) $event['hlsSourceUrl'] : null,
                'current_time' => isset($event['currentTime']) ? (float) $event['currentTime'] : null,
                'live_offset' => isset($event['liveOffset']) ? (float) $event['liveOffset'] : null,
                'drift' => isset($event['drift']) ? (float) $event['drift'] : null,
                'playback_rate' => isset($event['playbackRate']) ? (float) $event['playbackRate'] : null,
                'media_error_code' => isset($event['mediaErrorCode']) ? (int) $event['mediaErrorCode'] : null,
                'waiting' => isset($event['waiting']) ? (bool) $event['waiting'] : null,
                'paused' => isset($event['paused']) ? (bool) $event['paused'] : null,
                'ready_state' => isset($event['readyState']) ? (int) $event['readyState'] : null,
            ];
        }, $acceptedInput);

        if ($events !== []) {
            Log::channel('public_radio_diagnostics')->info('public-radio-client-diagnostics', [
                'station_slug' => $slug,
                'viewer_id' => $viewer?->id,
                'client_id' => $clientId,
                'client_ip' => $request->ip(),
                'user_agent' => $request->userAgent(),
                'event_count' => count($events),
                'events' => $events,
            ]);
        }

        if ($droppedCount > 0) {
            Log::channel('public_radio_diagnostics')->warning('public-radio-client-diagnostics-throttled', [
                'station_slug' => $slug,
                'viewer_id' => $viewer?->id,
                'client_id' => $clientId,
                'client_ip' => $request->ip(),
                'user_agent' => $request->userAgent(),
                'accepted_count' => count($events),
                'dropped_count' => $droppedCount,
                'limit_per_minute' => self::MAX_DIAGNOSTIC_EVENTS_PER_MINUTE,
            ]);
        }

        return response()->json([
            'accepted' => count($events),
            'dropped' => $droppedCount,
        ]);
    }

    private function authorizePersonalStation(Request $request, string $slug): ?\App\Models\User
    {
        if (!$this->personalRadioAccess->isPersonalStationSlug($slug)) {
            return $this->radioAccessTokens->resolveFromRequest($request);
        }

        $owner = $this->personalRadioAccess->ownerForSlug($slug);
        if (!$owner) {
            throw new InvalidArgumentException('Radio station not found');
        }

        $viewer = $this->radioAccessTokens->resolveFromRequest($request);
        if (!$this->personalRadioAccess->canAccessStation($owner, $viewer)) {
            throw new HttpResponseException(response()->json([
                'error' => 'Forbidden',
                'message' => 'You do not have access to this personal radio station',
            ], 403));
        }

        return $viewer;
    }

    private function diagnosticsRateLimitKey(string $slug, string $clientId): string
    {
        return 'public_radio_diagnostics_rate:' . sha1($slug . '|' . $clientId . '|' . now()->format('YmdHi'));
    }
}
