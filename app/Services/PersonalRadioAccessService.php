<?php

namespace App\Services;

use App\Models\Track;
use App\Models\User;
use App\Models\UserSetting;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Str;

class PersonalRadioAccessService
{
    private const KEY_PREFIX = 'personal_radio_access:';
    private const ACCESS_MODE_ALL = 'all';
    private const ACCESS_MODE_SELECTED_USERS = 'selected_users';
    private const AVAILABLE_GENRES_CACHE_KEY = 'personal_radio_available_genres';
    private const AVAILABLE_GENRES_CACHE_TTL_SECONDS = 600;

    /**
     * @return array{
     *   enabled: bool,
     *   access_mode: string,
     *   allowed_user_ids: array<int, int>,
     *   preferred_genres: array<int, string>,
     *   available_genres: array<int, string>,
     *   available_users: array<int, array{id:int,name:string,email:string}>,
     *   station_slug: string,
     *   public_url: string,
     *   owner: array{id:int,name:string,email:string}
     * }
     */
    public function getSettingsForOwner(User $owner): array
    {
        $stored = UserSetting::getValue($this->settingsKey((int) $owner->id), []);
        $allowedUserIds = $this->normalizeAllowedUserIds(
            $owner,
            is_array($stored['allowed_user_ids'] ?? null) ? $stored['allowed_user_ids'] : [],
            is_array($stored['allowed_emails'] ?? null) ? $stored['allowed_emails'] : [],
        );

        $accessMode = $this->normalizeAccessMode((string) ($stored['access_mode'] ?? self::ACCESS_MODE_SELECTED_USERS));
        $preferredGenres = $this->normalizePreferredGenres(
            is_array($stored['preferred_genres'] ?? null) ? $stored['preferred_genres'] : [],
        );

        $availableUsers = User::query()
            ->where('id', '!=', $owner->id)
            ->orderByRaw('LOWER(name)')
            ->orderBy('id')
            ->get(['id', 'name', 'email'])
            ->map(fn (User $user): array => [
                'id' => (int) $user->id,
                'name' => (string) $user->name,
                'email' => (string) $user->email,
            ])
            ->values()
            ->all();

        return [
            'enabled' => (bool) ($stored['enabled'] ?? false),
            'access_mode' => $accessMode,
            'allowed_user_ids' => $allowedUserIds,
            'preferred_genres' => $preferredGenres,
            'available_genres' => $this->availableGenres(),
            'available_users' => $availableUsers,
            'station_slug' => $this->stationSlugForUser((int) $owner->id),
            'public_url' => '/radio/' . $this->stationSlugForUser((int) $owner->id),
            'owner' => [
                'id' => (int) $owner->id,
                'name' => (string) $owner->name,
                'email' => (string) $owner->email,
            ],
        ];
    }

    /**
     * @return array{
     *   enabled: bool,
     *   access_mode: string,
     *   allowed_user_ids: array<int, int>,
     *   preferred_genres: array<int, string>,
     *   available_genres: array<int, string>,
     *   available_users: array<int, array{id:int,name:string,email:string}>,
     *   station_slug: string,
     *   public_url: string,
     *   owner: array{id:int,name:string,email:string}
     * }
     */
    public function updateSettingsForOwner(User $owner, bool $enabled, string $accessMode, array $allowedUserIds, ?array $preferredGenres = null): array
    {
        $stored = UserSetting::getValue($this->settingsKey((int) $owner->id), []);
        $normalizedUserIds = collect($allowedUserIds)
            ->map(fn (mixed $userId): int => (int) $userId)
            ->filter(fn (int $userId): bool => $userId > 0 && $userId !== (int) $owner->id)
            ->unique()
            ->values()
            ->all();

        UserSetting::setValue($this->settingsKey((int) $owner->id), [
            'enabled' => $enabled,
            'access_mode' => $this->normalizeAccessMode($accessMode),
            'allowed_user_ids' => $normalizedUserIds,
            'preferred_genres' => $this->normalizePreferredGenres(
                is_array($preferredGenres)
                    ? $preferredGenres
                    : (is_array($stored['preferred_genres'] ?? null) ? $stored['preferred_genres'] : []),
            ),
        ]);

        return $this->getSettingsForOwner($owner);
    }

    /**
     * @return array<int, string>
     */
    public function preferredGenresForOwner(User $owner): array
    {
        $stored = UserSetting::getValue($this->settingsKey((int) $owner->id), []);

        return $this->normalizePreferredGenres(
            is_array($stored['preferred_genres'] ?? null) ? $stored['preferred_genres'] : [],
        );
    }

    /**
     * @return array<int, string>
     */
    public function availableGenres(): array
    {
        /** @var array<int, string> $genres */
        $genres = Cache::remember(
            self::AVAILABLE_GENRES_CACHE_KEY,
            now()->addSeconds(self::AVAILABLE_GENRES_CACHE_TTL_SECONDS),
            function (): array {
                return Track::query()
                    ->selectRaw('LOWER(TRIM(genre)) as genre_name, COUNT(*) as aggregate')
                    ->whereNotNull('genre')
                    ->whereRaw("TRIM(genre) <> ''")
                    ->groupByRaw('LOWER(TRIM(genre))')
                    ->orderByDesc('aggregate')
                    ->pluck('genre_name')
                    ->map(fn (mixed $genre): string => is_string($genre) ? trim($genre) : '')
                    ->filter(fn (string $genre): bool => $genre !== '')
                    ->values()
                    ->all();
            },
        );

        return $genres;
    }

    /**
     * @return array{
     *   enabled: bool,
     *   access_mode: string,
     *   allowed_user_ids: array<int, int>,
     *   preferred_genres: array<int, string>,
     *   available_genres: array<int, string>,
     *   available_users: array<int, array{id:int,name:string,email:string}>,
     *   station_slug: string,
     *   public_url: string,
     *   owner: array{id:int,name:string,email:string}
     * }
     */
    public function updatePreferredGenresForOwner(User $owner, array $preferredGenres): array
    {
        $settings = $this->getSettingsForOwner($owner);

        return $this->updateSettingsForOwner(
            $owner,
            (bool) $settings['enabled'],
            (string) $settings['access_mode'],
            is_array($settings['allowed_user_ids'] ?? null) ? $settings['allowed_user_ids'] : [],
            $preferredGenres,
        );
    }

    public function isPersonalStationSlug(string $slug): bool
    {
        return $this->ownerIdFromSlug($slug) !== null;
    }

    public function ownerIdFromSlug(string $slug): ?int
    {
        if (!preg_match('/^personal-(\d+)$/', trim(Str::lower($slug)), $matches)) {
            return null;
        }

        return max(1, (int) ($matches[1] ?? 0));
    }

    public function stationSlugForUser(int $userId): string
    {
        return 'personal-' . $userId;
    }

    public function ownerForSlug(string $slug): ?User
    {
        $ownerId = $this->ownerIdFromSlug($slug);
        if ($ownerId === null) {
            return null;
        }

        return User::find($ownerId);
    }

    public function canAccessStation(User $owner, ?User $viewer): bool
    {
        if ($viewer && (int) $viewer->id === (int) $owner->id) {
            return true;
        }

        $settings = $this->getSettingsForOwner($owner);
        if (!$settings['enabled'] || !$viewer) {
            return false;
        }

        if (($settings['access_mode'] ?? self::ACCESS_MODE_SELECTED_USERS) === self::ACCESS_MODE_ALL) {
            return true;
        }

        return in_array((int) $viewer->id, $settings['allowed_user_ids'] ?? [], true);
    }

    public function canManageStation(User $owner, ?User $viewer): bool
    {
        return $viewer !== null && (int) $viewer->id === (int) $owner->id;
    }

    private function settingsKey(int $ownerId): string
    {
        return self::KEY_PREFIX . $ownerId;
    }

    /**
     * @param array<int, mixed> $allowedUserIds
     * @param array<int, mixed> $legacyAllowedEmails
     * @return array<int, int>
     */
    private function normalizeAllowedUserIds(User $owner, array $allowedUserIds, array $legacyAllowedEmails = []): array
    {
        $normalizedUserIds = collect($allowedUserIds)
            ->map(fn (mixed $userId): int => (int) $userId)
            ->filter(fn (int $userId): bool => $userId > 0 && $userId !== (int) $owner->id)
            ->unique();

        if ($normalizedUserIds->isNotEmpty()) {
            return $normalizedUserIds->values()->all();
        }

        $normalizedEmails = collect($legacyAllowedEmails)
            ->filter(fn (mixed $email): bool => is_string($email) && trim($email) !== '')
            ->map(fn (string $email): string => Str::lower(trim($email)))
            ->unique()
            ->values()
            ->all();

        if ($normalizedEmails === []) {
            return [];
        }

        return User::query()
            ->whereIn(DB::raw('LOWER(email)'), $normalizedEmails)
            ->where('id', '!=', $owner->id)
            ->pluck('id')
            ->map(fn (mixed $userId): int => (int) $userId)
            ->unique()
            ->values()
            ->all();
    }

    private function normalizeAccessMode(string $accessMode): string
    {
        return trim(Str::lower($accessMode)) === self::ACCESS_MODE_ALL
            ? self::ACCESS_MODE_ALL
            : self::ACCESS_MODE_SELECTED_USERS;
    }

    /**
     * @param array<int, mixed> $genres
     * @return array<int, string>
     */
    private function normalizePreferredGenres(array $genres): array
    {
        $allowedGenres = array_fill_keys($this->availableGenres(), true);

        return collect($genres)
            ->map(fn (mixed $genre): string => is_string($genre) ? trim(Str::lower($genre)) : '')
            ->filter(fn (string $genre): bool => $genre !== '' && isset($allowedGenres[$genre]))
            ->unique()
            ->values()
            ->all();
    }
}
