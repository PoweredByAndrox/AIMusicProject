<?php

namespace App\Services;

use App\Jobs\TrackHlsPrewarmJob;
use App\Jobs\TrackWaveformPrewarmJob;
use App\Models\Track;
use App\Models\User;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Redis;
use Illuminate\Support\Facades\Storage;
use InvalidArgumentException;

class PublicRadioStationService
{
    private const STATE_TTL_SECONDS = 86400;
    private const LOCK_TTL_SECONDS = 5;
    private const LOCK_WAIT_SECONDS = 3;
    private const PERSONAL_DEEP_FETCH_LIMIT = 48;
    private const PERSONAL_RECOMMENDATION_FETCH_MIN = 180;
    private const PERSONAL_DIVERSITY_POOL_MIN = 24;
    private const RECENT_HISTORY_REPLAY_GUARD = 5;

    /**
     * @var array<string, array{name:string,description:string,lookahead:int,history_limit:int,genres:array<int, string>}>
     */
    private const STATIONS = [
        'test' => [
            'name' => 'Radio Test',
            'description' => 'Shared all-genre station for public radio testing.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => [],
        ],
        'hardstyle' => [
            'name' => 'Hardstyle Radio',
            'description' => 'Shared hardstyle-focused radio station.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['hardstyle', 'hands up', 'jumpstyle', 'hardcore', 'happy hardcore'],
        ],
        'dubstep' => [
            'name' => 'Dubstep Radio',
            'description' => 'Shared bass-heavy station for dubstep and adjacent styles.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['dubstep', 'grime', 'halftime', 'trap'],
        ],
        'house' => [
            'name' => 'House Radio',
            'description' => 'Shared house station spanning house and electro house cuts.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['house', 'electro house', 'tropical house', 'electro'],
        ],
        'trance' => [
            'name' => 'Trance Radio',
            'description' => 'Shared trance station for trance and tech trance tracks.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['trance', 'tech trance'],
        ],
        'dnb' => [
            'name' => 'Drum & Bass Radio',
            'description' => 'Shared drum and bass station.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['drum & bass'],
        ],
        'bass-mix' => [
            'name' => 'Bass Mix',
            'description' => 'Mixed bass station for dubstep, drum & bass, grime, halftime, trap, and bassline cuts.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['dubstep', 'drum & bass', 'grime', 'halftime', 'trap', 'bassline'],
        ],
        'hard-dance' => [
            'name' => 'Hard Dance Mix',
            'description' => 'Mixed hard dance station spanning hardstyle, hardcore, jumpstyle, donk, and hands up tracks.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['hardstyle', 'hardcore', 'jumpstyle', 'hands up', 'happy hardcore', 'donk'],
        ],
        'club-mix' => [
            'name' => 'Club Mix',
            'description' => 'Mixed club station for house, electro house, progressive house, euro house, electro, and dance-pop.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['house', 'electro house', 'progressive house', 'euro house', 'electro', 'dance-pop'],
        ],
        'euphoric' => [
            'name' => 'Euphoric Mix',
            'description' => 'Mixed euphoric station for trance, tech trance, progressive house, hands up, and chillwave energy.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['trance', 'tech trance', 'progressive house', 'hands up', 'chillwave'],
        ],
        'afterhours' => [
            'name' => 'Afterhours Mix',
            'description' => 'Mixed late-night station combining ambient, techno, experimental, synth-pop, and chillwave tracks.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['ambient', 'techno', 'experimental', 'synth-pop', 'chillwave'],
        ],
        'festival-mix' => [
            'name' => 'Festival Mix',
            'description' => 'High-energy mixed station spanning hardstyle, trance, electro house, big-room club energy, and drum & bass.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['hardstyle', 'trance', 'electro house', 'progressive house', 'drum & bass'],
        ],
        'uk-rave' => [
            'name' => 'UK Rave Mix',
            'description' => 'Mixed UK-focused station for grime, dubstep, drum & bass, trap, and halftime cuts.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['grime', 'dubstep', 'drum & bass', 'trap', 'halftime'],
        ],
        'melodic-rush' => [
            'name' => 'Melodic Rush',
            'description' => 'Mixed melodic station for trance, progressive house, hands up, chillwave, and euphoric hard dance crossover.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['trance', 'progressive house', 'hands up', 'chillwave', 'hardstyle'],
        ],
        'warehouse-mix' => [
            'name' => 'Warehouse Mix',
            'description' => 'Mixed underground station for techno, electro, house, experimental, and bass-heavy night tracks.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => ['techno', 'electro', 'house', 'experimental', 'dubstep'],
        ],
    ];

    public function __construct(
        private readonly SmartRadioService $smartRadio,
        private readonly PersonalRadioAccessService $personalRadioAccess,
        private readonly PublicRadioWaveformService $publicRadioWaveformService,
        private readonly PublicRadioHlsService $publicRadioHlsService,
    ) {}

    /**
     * @return array<string, mixed>
     */
    public function snapshot(string $slug, ?User $viewer = null): array
    {
        $station = $this->resolveStation($slug);

        return $this->synchronizeState($slug, $station, function (array $state) use ($slug, $station, $viewer): array {
            $nowTs = now()->timestamp;
            $queue = $state['queue'] ?? [];
            $history = $state['history'] ?? [];

            return [
                'station' => $this->serializeStation($station, $viewer),
                'viewer' => $this->serializeViewer($viewer),
                'stations' => $this->serializeStations($slug, $viewer, $station),
                'listener_count' => $this->listenerCount($slug),
                'now_playing' => isset($queue[0]) ? $this->serializeEntry($queue[0], $nowTs) : null,
                'up_next' => array_map(
                    fn (array $entry): array => $this->serializeEntry($entry, $nowTs),
                    array_slice($queue, 1, 4),
                ),
                'history' => array_map(
                    fn (array $entry): array => $this->serializeEntry($entry, $nowTs),
                    array_slice(array_reverse($history), 0, (int) $station['history_limit']),
                ),
                'generated_at' => now()->toIso8601String(),
            ];
        });
    }

    /**
     * @return array<string, mixed>
     */
    public function history(string $slug, ?User $viewer = null): array
    {
        $station = $this->resolveStation($slug);

        return $this->synchronizeState($slug, $station, function (array $state) use ($slug, $station, $viewer): array {
            $nowTs = now()->timestamp;
            $history = $state['history'] ?? [];

            return [
                'station' => $this->serializeStation($station, $viewer),
                'stations' => $this->serializeStations($slug, $viewer, $station),
                'listener_count' => $this->listenerCount($slug),
                'history' => array_map(
                    fn (array $entry): array => $this->serializeEntry($entry, $nowTs),
                    array_slice(array_reverse($history), 0, (int) $station['history_limit']),
                ),
            ];
        });
    }

    /**
     * @param array<string, mixed> $station
     * @param callable(array<string, mixed>): array<string, mixed> $callback
     * @return array<string, mixed>
     */
    private function synchronizeState(string $slug, array $station, callable $callback): array
    {
        $lock = Cache::lock($this->lockKey($slug), self::LOCK_TTL_SECONDS);

        return $lock->block(self::LOCK_WAIT_SECONDS, function () use ($slug, $station, $callback): array {
            $state = $this->normalizeState(Cache::get($this->stateKey($slug), []));
            $state = $this->advanceState($slug, $station, $state);

            Cache::put($this->stateKey($slug), $state, now()->addSeconds(self::STATE_TTL_SECONDS));

            return $callback($state);
        });
    }

    /**
     * @param array<string, mixed> $station
     * @param array<string, mixed> $state
     * @return array<string, mixed>
     */
    private function advanceState(string $slug, array $station, array $state): array
    {
        $nowTs = now()->timestamp;
        $queue = array_values($state['queue'] ?? []);
        $history = array_values($state['history'] ?? []);

        while (!empty($queue) && (int) ($queue[0]['end_at'] ?? 0) <= $nowTs) {
            $history[] = array_shift($queue);
        }

        $history = array_slice($history, -((int) $station['history_limit']));
        $lookahead = max(1, (int) $station['lookahead']);
        $missing = $lookahead - count($queue);

        if ($missing > 0) {
            $startAt = !empty($queue)
                ? (int) $queue[count($queue) - 1]['end_at']
                : $nowTs;

            foreach ($this->pickTracks($missing, $queue, $history, $station) as $track) {
                $duration = max(1, (int) ceil((float) $track->duration));
                $queue[] = [
                    'track_id' => (int) $track->id,
                    'title' => (string) $track->title,
                    'artist' => $track->artist,
                    'duration' => $duration,
                    'start_at' => $startAt,
                    'end_at' => $startAt + $duration,
                ];
                $startAt += $duration;
            }
        }

        $this->dispatchQueuePrewarmJobs($queue);

        return [
            'slug' => $slug,
            'queue' => $queue,
            'history' => $history,
            'updated_at' => now()->toIso8601String(),
        ];
    }

    /**
     * @param array<int, array<string, mixed>> $queue
     * @param array<int, array<string, mixed>> $history
     * @param array<string, mixed> $station
     * @return array<int, Track>
     */
    private function pickTracks(int $needed, array $queue, array $history, array $station): array
    {
        if ($needed <= 0) {
            return [];
        }

        $excludeTrackIds = array_values(array_unique(array_map(
            static fn (array $entry): int => (int) ($entry['track_id'] ?? 0),
            array_merge($queue, $history),
        )));

        $tracks = $this->fetchPlayableTracks($excludeTrackIds, max(12, $needed * 4), $station);

        if (count($tracks) < $needed && ($station['type'] ?? 'shared') === 'personal') {
            $tracks = $this->mergeUniqueTracks(
                $tracks,
                $this->fetchPlayableTracks($excludeTrackIds, max(self::PERSONAL_DEEP_FETCH_LIMIT, $needed * 12), $station),
            );
        }

        if (count($tracks) < $needed) {
            $queuedTrackIds = array_values(array_unique(array_map(
                static fn (array $entry): int => (int) ($entry['track_id'] ?? 0),
                $queue,
            )));
            $recentHistoryTrackIds = array_values(array_unique(array_map(
                static fn (array $entry): int => (int) ($entry['track_id'] ?? 0),
                array_slice($history, -min(self::RECENT_HISTORY_REPLAY_GUARD, count($history))),
            )));
            $fallbackExcludeTrackIds = array_values(array_unique(array_merge($queuedTrackIds, $recentHistoryTrackIds)));
            $fallbackTracks = $this->fetchPlayableTracks($fallbackExcludeTrackIds, max(24, $needed * 8), $station);

            $tracks = $this->mergeUniqueTracks($tracks, $fallbackTracks);

            if (count($tracks) < $needed) {
                $tracks = $this->mergeUniqueTracks(
                    $tracks,
                    $this->fetchPlayableTracks($queuedTrackIds, max(12, $needed * 6), $station),
                );
            }
        }

        return array_slice($tracks, 0, $needed);
    }

    /**
     * @param array<int, int> $excludeTrackIds
     * @param array<string, mixed> $station
     * @return array<int, Track>
     */
    private function fetchPlayableTracks(array $excludeTrackIds, int $limit, array $station): array
    {
        if (($station['type'] ?? 'shared') === 'personal') {
            return $this->fetchPersonalTracks(
                $excludeTrackIds,
                $limit,
                (int) ($station['owner_id'] ?? 0),
                is_array($station['genres'] ?? null) ? $station['genres'] : [],
            );
        }

        $query = Track::query()
            ->where('analysis_status', 'completed')
            ->whereNotNull('file_path')
            ->whereNotNull('duration')
            ->where('duration', '>', 0);

        $stationGenres = array_values(array_unique(array_map(
            static fn (string $genre): string => mb_strtolower(trim($genre)),
            $station['genres'] ?? [],
        )));

        if ($stationGenres !== []) {
            $query->where(function ($builder) use ($stationGenres): void {
                foreach ($stationGenres as $genre) {
                    $builder->orWhere(DB::raw('LOWER(TRIM(genre))'), '=', $genre);
                }
            });
        }

        if (!empty($excludeTrackIds)) {
            $query->whereNotIn('id', $excludeTrackIds);
        }

        /** @var array<int, Track> $tracks */
        $tracks = $query
            ->inRandomOrder()
            ->limit($limit)
            ->get(['id', 'title', 'artist', 'duration', 'file_path'])
            ->filter(fn (Track $track): bool => is_string($track->file_path) && Storage::disk('local')->exists($track->file_path))
            ->values()
            ->all();

        return $tracks;
    }

    /**
     * @param array<int, int> $excludeTrackIds
     * @return array<int, Track>
     */
    private function fetchPersonalTracks(array $excludeTrackIds, int $limit, int $ownerId, array $preferredGenres = []): array
    {
        if ($ownerId <= 0) {
            return [];
        }

        $normalizedGenres = $this->normalizeGenres($preferredGenres);

        /** @var array<int, Track> $tracks */
        $tracks = $this->smartRadio
            ->getPreferenceBasedRecommendations(max($limit * 18, self::PERSONAL_RECOMMENDATION_FETCH_MIN), $ownerId)
            ->filter(fn (Track $track): bool => !in_array((int) $track->id, $excludeTrackIds, true))
            ->filter(fn (Track $track): bool => is_string($track->file_path) && Storage::disk('local')->exists($track->file_path))
            ->filter(fn (Track $track): bool => $this->trackMatchesGenres($track, $normalizedGenres))
            ->values()
            ->all();

        $diversifiedPoolSize = min(count($tracks), max($limit * 5, self::PERSONAL_DIVERSITY_POOL_MIN));

        if ($diversifiedPoolSize > $limit) {
            /** @var array<int, Track> $diversifiedTracks */
            $diversifiedTracks = collect($tracks)
                ->take($diversifiedPoolSize)
                ->shuffle()
                ->values()
                ->all();

            return array_slice($diversifiedTracks, 0, $limit);
        }

        return array_slice($tracks, 0, $limit);
    }

    /**
     * @param array<int, Track> $tracks
     * @param array<int, Track> $candidates
     * @return array<int, Track>
     */
    private function mergeUniqueTracks(array $tracks, array $candidates): array
    {
        foreach ($candidates as $track) {
            $alreadyIncluded = collect($tracks)->contains(
                fn (Track $existing): bool => (int) $existing->id === (int) $track->id,
            );

            if (!$alreadyIncluded) {
                $tracks[] = $track;
            }
        }

        return $tracks;
    }

    /**
     * @param array<string, mixed> $entry
     * @return array<string, mixed>
     */
    private function serializeEntry(array $entry, int $nowTs): array
    {
        $trackId = (int) ($entry['track_id'] ?? 0);
        $startAt = (int) ($entry['start_at'] ?? $nowTs);
        $endAt = (int) ($entry['end_at'] ?? $startAt);
        $duration = max(1, (int) ($entry['duration'] ?? ($endAt - $startAt)));
        $offsetSeconds = min($duration, max(0, $nowTs - $startAt));

        return [
            'track_id' => $trackId,
            'title' => (string) ($entry['title'] ?? 'Unknown Track'),
            'artist' => $entry['artist'] ?? null,
            'duration' => $duration,
            'started_at' => now()->setTimestamp($startAt)->toIso8601String(),
            'ends_at' => now()->setTimestamp($endAt)->toIso8601String(),
            'offset_seconds' => $offsetSeconds,
            'remaining_seconds' => max(0, $endAt - $nowTs),
            'stream_url' => '/api/tracks/' . $trackId . '/stream',
            'waveform_url' => $this->publicRadioWaveformService->currentPublicUrl($trackId) ?? '/api/tracks/' . $trackId . '/waveform',
            'hls_master_url' => $this->publicRadioHlsService->currentMasterPublicUrl($trackId),
        ];
    }

    /**
     * @param array<string, mixed> $station
     * @return array<string, mixed>
     */
    private function serializeStation(array $station, ?User $viewer = null): array
    {
        $owner = isset($station['owner']) && $station['owner'] instanceof User
            ? $station['owner']
            : null;
        $isPersonal = ($station['type'] ?? 'shared') === 'personal';
        $canManage = $owner ? $this->personalRadioAccess->canManageStation($owner, $viewer) : false;

        return [
            'slug' => (string) $station['slug'],
            'name' => (string) $station['name'],
            'description' => (string) $station['description'],
            'public_url' => '/radio/' . $station['slug'],
            'embed_url' => '/embed/radio/' . $station['slug'],
            'genres' => $station['genres'] ?? [],
            'available_genres' => $canManage ? $this->personalRadioAccess->availableGenres() : [],
            'can_manage' => $canManage,
            'is_personal' => $isPersonal,
            'owner' => $owner
                ? [
                    'id' => (int) $owner->id,
                    'name' => (string) $owner->name,
                    'email' => (string) $owner->email,
                ]
                : null,
        ];
    }

    private function serializeViewer(?User $viewer): ?array
    {
        if (!$viewer) {
            return null;
        }

        return [
            'id' => (int) $viewer->id,
            'name' => (string) $viewer->name,
            'email' => (string) $viewer->email,
        ];
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    private function serializeStations(string $selectedSlug, ?User $viewer = null, ?array $selectedStation = null): array
    {
        $stations = [];

        foreach (self::STATIONS as $slug => $station) {
            $stations[] = [
                ...$this->serializeStation([
                    ...$station,
                    'slug' => $slug,
                    'type' => 'shared',
                ], $viewer),
                'selected' => $slug === $selectedSlug,
            ];
        }

        if ($viewer) {
            $viewerStation = $this->resolveStation($this->personalRadioAccess->stationSlugForUser((int) $viewer->id));
            $stations[] = [
                ...$this->serializeStation($viewerStation, $viewer),
                'selected' => $viewerStation['slug'] === $selectedSlug,
            ];
        }

        if (
            $selectedStation !== null
            && ($selectedStation['type'] ?? 'shared') === 'personal'
            && !collect($stations)->contains(fn (array $station): bool => $station['slug'] === $selectedStation['slug'])
        ) {
            $stations[] = [
                ...$this->serializeStation($selectedStation, $viewer),
                'selected' => true,
            ];
        }

        return $stations;
    }

    /**
     * @param array<string, mixed> $state
     * @return array<string, mixed>
     */
    private function normalizeState(array $state): array
    {
        return [
            'queue' => is_array($state['queue'] ?? null) ? array_values($state['queue']) : [],
            'history' => is_array($state['history'] ?? null) ? array_values($state['history']) : [],
            'updated_at' => $state['updated_at'] ?? null,
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function resolveStation(string $slug): array
    {
        $normalizedSlug = trim(mb_strtolower($slug));

        if (isset(self::STATIONS[$normalizedSlug])) {
            return [
                ...self::STATIONS[$normalizedSlug],
                'slug' => $normalizedSlug,
                'type' => 'shared',
            ];
        }

        $owner = $this->personalRadioAccess->ownerForSlug($normalizedSlug);
        if (!$owner) {
            throw new InvalidArgumentException('Radio station not found');
        }

        return [
            'slug' => $normalizedSlug,
            'type' => 'personal',
            'owner_id' => (int) $owner->id,
            'owner' => $owner,
            'name' => $owner->name . '\'s Wave',
            'description' => 'Private preference-driven radio based on ' . $owner->name . '\'s listening taste.',
            'lookahead' => 5,
            'history_limit' => 10,
            'genres' => $this->personalRadioAccess->preferredGenresForOwner($owner),
        ];
    }

    public function refreshUpcomingQueue(string $slug): void
    {
        $station = $this->resolveStation($slug);
        $lock = Cache::lock($this->lockKey($slug), self::LOCK_TTL_SECONDS);

        $lock->block(self::LOCK_WAIT_SECONDS, function () use ($slug, $station): void {
            $state = $this->normalizeState(Cache::get($this->stateKey($slug), []));
            $state = $this->advanceState($slug, $station, $state);
            $currentEntry = $state['queue'][0] ?? null;

            Cache::put($this->stateKey($slug), [
                'queue' => $currentEntry ? [$currentEntry] : [],
                'history' => $state['history'] ?? [],
                'updated_at' => now()->toIso8601String(),
            ], now()->addSeconds(self::STATE_TTL_SECONDS));
        });
    }

    private function stateKey(string $slug): string
    {
        return 'public_radio_station:' . $slug;
    }

    private function lockKey(string $slug): string
    {
        return 'public_radio_station_lock:' . $slug;
    }

    private function listenerCount(string $slug): int
    {
        $value = Redis::get($this->listenerCountKey($slug));

        return max(0, (int) $value);
    }

    /**
     * @param array<int, array<string, mixed>> $queue
     */
    private function dispatchQueuePrewarmJobs(array $queue): void
    {
        $entries = array_slice($queue, 0, 2);
        $trackIds = array_values(array_filter(array_map(
            static fn (array $entry): int => (int) ($entry['track_id'] ?? 0),
            $entries,
        )));

        /** @var \Illuminate\Support\Collection<int, Track> $tracks */
        $tracks = Track::query()
            ->whereIn('id', $trackIds)
            ->get(['id', 'file_path'])
            ->keyBy('id');

        foreach ($entries as $entry) {
            $trackId = (int) ($entry['track_id'] ?? 0);
            if ($trackId <= 0) {
                continue;
            }

            $track = $tracks->get($trackId);
            if (!$track instanceof Track) {
                continue;
            }

            if ($this->publicRadioWaveformService->currentPublicUrl($trackId) === null) {
                TrackWaveformPrewarmJob::dispatch($trackId);
            }

            if (!$this->publicRadioHlsService->hasCurrentPackage($track)) {
                TrackHlsPrewarmJob::dispatch($trackId);
            }
        }
    }

    private function listenerCountKey(string $slug): string
    {
        return 'public_radio_listener_count:' . $slug;
    }

    /**
     * @param array<int, string> $genres
     * @return array<int, string>
     */
    private function normalizeGenres(array $genres): array
    {
        return array_values(array_unique(array_map(
            static fn (string $genre): string => mb_strtolower(trim($genre)),
            array_values(array_filter($genres, static fn (mixed $genre): bool => is_string($genre) && trim($genre) !== '')),
        )));
    }

    /**
     * @param array<int, string> $genres
     */
    private function trackMatchesGenres(Track $track, array $genres): bool
    {
        if ($genres === []) {
            return true;
        }

        $trackGenre = is_string($track->genre) ? mb_strtolower(trim($track->genre)) : '';

        return $trackGenre !== '' && in_array($trackGenre, $genres, true);
    }
}
