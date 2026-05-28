<?php

namespace App\Services;

use App\Models\Track;
use Illuminate\Support\Str;

class ArtistCreditParser
{
    private const ROLE_PRIMARY = 'primary';
    private const ROLE_FEATURED = 'featured';
    private const ROLE_REMIXER = 'remixer';

    /**
     * @return array{title: string, artist: ?string, artist_credits: array<int, array{name: string, slug: string, roles: array<int, string>}>, artist_slugs: array<int, string>}
     */
    public function parseUploadMetadata(?string $artist, ?string $title, string $fallbackFileName): array
    {
        $fallbackTitle = $this->normalizeFallbackTitle($fallbackFileName);
        $normalizedArtist = $this->sanitizeText($artist);
        $normalizedTitle = $this->sanitizeText($title);

        if (($normalizedArtist === null || $normalizedArtist === '') && $normalizedTitle !== null && $normalizedTitle !== '') {
            return $this->parseRawTitle($normalizedTitle, null, $fallbackTitle);
        }

        return $this->buildParsedTrack($normalizedArtist, $normalizedTitle ?: $fallbackTitle, null);
    }

    /**
     * @return array{title: string, artist: ?string, artist_credits: array<int, array{name: string, slug: string, roles: array<int, string>}>, artist_slugs: array<int, string>}
     */
    public function parseYouTubeMetadata(string $rawTitle, ?string $channelName = null): array
    {
        return $this->parseRawTitle($rawTitle, $channelName);
    }

    /**
     * @param array<string, mixed> $attributes
     * @return array{title: string, artist: ?string, artist_credits: array<int, array{name: string, slug: string, roles: array<int, string>}>, artist_slugs: array<int, string>}
     */
    public function resolveFromAttributes(array $attributes): array
    {
        $storedCredits = $this->normalizeStoredCredits($attributes['artist_credits'] ?? null);
        if ($storedCredits !== []) {
            $artist = $this->sanitizeText($attributes['artist'] ?? null) ?? $this->buildLegacyArtistString($storedCredits);
            $title = $this->sanitizeText($attributes['title'] ?? null) ?? 'Unknown Title';

            return [
                'title' => $title,
                'artist' => $artist,
                'artist_credits' => $storedCredits,
                'artist_slugs' => $this->normalizeStoredSlugs($attributes['artist_slugs'] ?? null, $storedCredits),
            ];
        }

        $source = $this->sanitizeText($attributes['source'] ?? null);
        $title = $this->sanitizeText($attributes['title'] ?? null) ?? 'Unknown Title';
        $artist = $this->sanitizeText($attributes['artist'] ?? null);
        $channelName = $this->sanitizeText($attributes['youtube_channel'] ?? null);
        $filePath = $this->sanitizeText($attributes['file_path'] ?? null) ?? $title;

        return $source === 'youtube'
            ? $this->parseYouTubeMetadata($title, $channelName)
            : $this->parseUploadMetadata($artist, $title, basename($filePath));
    }

    /**
     * @return array{title: string, artist: ?string, artist_credits: array<int, array{name: string, slug: string, roles: array<int, string>}>, artist_slugs: array<int, string>}
     */
    public function resolveTrack(Track $track): array
    {
        return $this->resolveFromAttributes($track->getAttributes());
    }

    /**
     * @param array<int, array{name: string, slug: string, roles: array<int, string>}> $artistCredits
     * @return array<int, string>
     */
    public function rolesForSlug(array $artistCredits, string $slug): array
    {
        foreach ($artistCredits as $credit) {
            if (($credit['slug'] ?? null) !== $slug) {
                continue;
            }

            $roles = array_values(array_unique(array_filter(
                array_map(
                    fn (mixed $role): ?string => is_string($role) ? trim($role) : null,
                    is_array($credit['roles'] ?? null) ? $credit['roles'] : [],
                ),
            )));

            return $this->sortRoles($roles);
        }

        return [];
    }

    /**
     * @param array<int, array{name: string, slug: string, roles: array<int, string>}> $artistCredits
     */
    public function findNameForSlug(array $artistCredits, string $slug): ?string
    {
        foreach ($artistCredits as $credit) {
            if (($credit['slug'] ?? null) === $slug && is_string($credit['name'] ?? null)) {
                return trim($credit['name']);
            }
        }

        return null;
    }

    /**
     * @return array{title: string, artist: ?string, artist_credits: array<int, array{name: string, slug: string, roles: array<int, string>}>, artist_slugs: array<int, string>}
     */
    private function parseRawTitle(string $rawTitle, ?string $channelName = null, ?string $fallbackTitle = null): array
    {
        $cleanTitle = $this->removeNoiseSuffixes($rawTitle);
        $channelName = $this->normalizeArtistName($channelName);

        if (preg_match('/^(.+?)\s+(?:-|\x{2013}|\x{2014}|\x{2212})\s+(.+)$/u', $cleanTitle, $matches) === 1) {
            $artistPart = trim($matches[1]);
            $titlePart = trim($matches[2]);

            if ($this->looksLikeArtistPart($artistPart)) {
                return $this->buildParsedTrack($artistPart, $titlePart, $channelName);
            }
        }

        if (preg_match('/^(.+?):\s*(.+)$/u', $cleanTitle, $matches) === 1 && $this->looksLikeArtistPart($matches[1])) {
            return $this->buildParsedTrack($matches[1], $matches[2], $channelName);
        }

        return $this->buildParsedTrack($channelName, $cleanTitle ?: ($fallbackTitle ?: $rawTitle), $channelName);
    }

    /**
     * @return array{title: string, artist: ?string, artist_credits: array<int, array{name: string, slug: string, roles: array<int, string>}>, artist_slugs: array<int, string>}
     */
    private function buildParsedTrack(?string $artistPart, string $titlePart, ?string $fallbackPrimaryArtist): array
    {
        $artistPart = $this->sanitizeText($artistPart);
        $titlePart = $this->cleanupTrackTitle($titlePart);

        [$primaryArtists, $featuredFromArtist] = $this->parseArtistPart($artistPart);
        [$titleWithoutFeatured, $featuredFromTitle] = $this->extractFeaturedArtistsFromTitle($titlePart);
        $remixers = $this->extractRemixersFromTitle($titleWithoutFeatured);

        if ($primaryArtists === [] && $fallbackPrimaryArtist !== null) {
            $primaryArtists = $this->splitArtistNames($fallbackPrimaryArtist);
        }

        $credits = [];
        foreach ($primaryArtists as $name) {
            $credits = $this->upsertCredit($credits, $name, self::ROLE_PRIMARY);
        }
        foreach (array_merge($featuredFromArtist, $featuredFromTitle) as $name) {
            $credits = $this->upsertCredit($credits, $name, self::ROLE_FEATURED);
        }
        foreach ($remixers as $name) {
            $credits = $this->upsertCredit($credits, $name, self::ROLE_REMIXER);
        }

        return [
            'title' => $titleWithoutFeatured ?: 'Unknown Title',
            'artist' => $this->buildLegacyArtistString($credits),
            'artist_credits' => array_values($credits),
            'artist_slugs' => array_values(array_map(
                static fn (array $credit): string => $credit['slug'],
                array_values($credits),
            )),
        ];
    }

    private function normalizeFallbackTitle(string $fallbackFileName): string
    {
        $withoutExtension = pathinfo($fallbackFileName, PATHINFO_FILENAME);

        return $this->cleanupTrackTitle(str_replace('_', ' ', $withoutExtension)) ?: 'Unknown Title';
    }

    private function removeNoiseSuffixes(string $rawTitle): string
    {
        $cleanTitle = $this->sanitizeText($rawTitle) ?? '';

        $suffixPatterns = [
            '/\s*\[Official\s*(Music\s*)?Video\]/iu',
            '/\s*\(Official\s*(Music\s*)?Video\)/iu',
            '/\s*\[Official\s*Audio\]/iu',
            '/\s*\(Official\s*Audio\)/iu',
            '/\s*\[Lyric\s*Video\]/iu',
            '/\s*\(Lyric\s*Video\)/iu',
            '/\s*\[Lyrics\]/iu',
            '/\s*\(Lyrics\)/iu',
            '/\s*\[Visualizer\]/iu',
            '/\s*\(Visualizer\)/iu',
            '/\s*\[Audio\]/iu',
            '/\s*\(Audio\)/iu',
            '/\s*\[(HD|HQ|4K)\]/iu',
            '/\s*\((HD|HQ|4K)\)/iu',
            '/\s*\|\s*.*$/u',
            '/\s*\/\/\s*.*$/u',
        ];

        foreach ($suffixPatterns as $pattern) {
            $cleanTitle = preg_replace($pattern, '', $cleanTitle) ?? $cleanTitle;
        }

        return $this->cleanupTrackTitle($cleanTitle);
    }

    private function looksLikeArtistPart(string $artistPart): bool
    {
        $artistPart = trim($artistPart);

        if ($artistPart === '' || mb_strlen($artistPart) > 120) {
            return false;
        }

        if (preg_match('/^\d{4}$/u', $artistPart) === 1) {
            return false;
        }

        return preg_match('/^(episode|part|vol|chapter)\s*\d/iu', $artistPart) !== 1;
    }

    /**
     * @return array{0: array<int, string>, 1: array<int, string>}
     */
    private function parseArtistPart(?string $artistPart): array
    {
        if ($artistPart === null || $artistPart === '') {
            return [[], []];
        }

        if (preg_match('/^(.+?)\s+(feat\.?|ft\.?|featuring)\s+(.+)$/iu', $artistPart, $matches) === 1) {
            return [
                $this->splitArtistNames($matches[1]),
                $this->splitArtistNames($matches[3]),
            ];
        }

        return [
            $this->splitArtistNames($artistPart),
            [],
        ];
    }

    /**
     * @return array{0: string, 1: array<int, string>}
     */
    private function extractFeaturedArtistsFromTitle(string $title): array
    {
        $featured = [];
        $cleanTitle = $title;

        $patterns = [
            '/\s*\((feat\.?|ft\.?|featuring)\s+([^)]+)\)/iu',
            '/\s*\[(feat\.?|ft\.?|featuring)\s+([^\]]+)\]/iu',
            '/\s+(feat\.?|ft\.?|featuring)\s+([^\(\[]+?)(?=\s*(\(|\[)|$)/iu',
        ];

        foreach ($patterns as $pattern) {
            if (preg_match($pattern, $cleanTitle, $matches) !== 1) {
                continue;
            }

            $featured = array_merge($featured, $this->splitArtistNames($matches[2]));
            $cleanTitle = trim((string) preg_replace($pattern, '', $cleanTitle, 1));
        }

        return [$this->cleanupTrackTitle($cleanTitle), array_values(array_unique($featured))];
    }

    /**
     * @return array<int, string>
     */
    private function extractRemixersFromTitle(string $title): array
    {
        preg_match_all('/[\(\[]([^\)\]]+)[\)\]]/u', $title, $matches);
        $remixers = [];

        foreach ($matches[1] ?? [] as $segment) {
            $segment = trim((string) $segment);
            if ($segment === '') {
                continue;
            }

            if (preg_match('/^(.+?)\s+remix$/iu', $segment, $remixMatch) === 1) {
                $candidate = $this->normalizeArtistName($remixMatch[1]);
                if ($candidate !== null && !$this->isGenericRemixLabel($candidate)) {
                    $remixers = array_merge($remixers, $this->splitArtistNames($candidate));
                }
                continue;
            }

            if (preg_match('/^remix\s+by\s+(.+)$/iu', $segment, $remixByMatch) === 1) {
                $candidate = $this->normalizeArtistName($remixByMatch[1]);
                if ($candidate !== null) {
                    $remixers = array_merge($remixers, $this->splitArtistNames($candidate));
                }
            }
        }

        return array_values(array_unique($remixers));
    }

    private function isGenericRemixLabel(string $candidate): bool
    {
        $normalized = Str::of($candidate)->lower()->squish()->value();

        return in_array($normalized, [
            'official',
            'extended',
            'club',
            'radio',
            'original',
            'vip',
            'dub',
            'mix',
            'edit',
            'bootleg',
            'version',
        ], true);
    }

    /**
     * @return array<int, string>
     */
    private function splitArtistNames(?string $value): array
    {
        $value = $this->sanitizeText($value);
        if ($value === null || $value === '') {
            return [];
        }

        $normalized = preg_replace('/\s+(x|\x{00D7}|vs\.?|and|with)\s+/iu', ',', $value) ?? $value;
        $normalized = preg_replace('/\s*&\s*/u', ',', $normalized) ?? $normalized;
        $normalized = preg_replace('/\s*\/\s*/u', ',', $normalized) ?? $normalized;
        $normalized = preg_replace('/\s*;\s*/u', ',', $normalized) ?? $normalized;

        $parts = preg_split('/\s*,\s*/u', $normalized) ?: [];
        $artists = [];

        foreach ($parts as $part) {
            $name = $this->normalizeArtistName($part);
            if ($name === null) {
                continue;
            }

            $artists[$this->slugifyArtist($name)] = $name;
        }

        return array_values($artists);
    }

    /**
     * @param array<int, array{name: string, slug: string, roles: array<int, string>}> $credits
     * @return array<int, array{name: string, slug: string, roles: array<int, string>}>
     */
    private function upsertCredit(array $credits, string $name, string $role): array
    {
        $normalizedName = $this->normalizeArtistName($name);
        if ($normalizedName === null) {
            return $credits;
        }

        $slug = $this->slugifyArtist($normalizedName);
        $existing = $credits[$slug] ?? [
            'name' => $normalizedName,
            'slug' => $slug,
            'roles' => [],
        ];

        $existing['roles'][] = $role;
        $existing['roles'] = $this->sortRoles(array_values(array_unique($existing['roles'])));
        $credits[$slug] = $existing;

        return $credits;
    }

    /**
     * @param array<int, string> $roles
     * @return array<int, string>
     */
    private function sortRoles(array $roles): array
    {
        $order = [
            self::ROLE_PRIMARY => 0,
            self::ROLE_FEATURED => 1,
            self::ROLE_REMIXER => 2,
        ];

        usort($roles, static fn (string $left, string $right): int => ($order[$left] ?? 99) <=> ($order[$right] ?? 99));

        return $roles;
    }

    /**
     * @param array<int, array{name: string, slug: string, roles: array<int, string>}> $credits
     */
    public function buildLegacyArtistString(array $credits): ?string
    {
        $primary = [];
        $featured = [];

        foreach ($credits as $credit) {
            $roles = is_array($credit['roles'] ?? null) ? $credit['roles'] : [];
            if (in_array(self::ROLE_PRIMARY, $roles, true)) {
                $primary[] = $credit['name'];
            }
            if (in_array(self::ROLE_FEATURED, $roles, true)) {
                $featured[] = $credit['name'];
            }
        }

        if ($primary === [] && $featured === []) {
            return null;
        }

        $artist = $primary !== [] ? implode(' & ', $primary) : implode(' & ', $featured);
        if ($primary !== [] && $featured !== []) {
            $artist .= ' ft. ' . implode(', ', array_values(array_unique($featured)));
        }

        return $artist;
    }

    /**
     * @param mixed $storedCredits
     * @return array<int, array{name: string, slug: string, roles: array<int, string>}>
     */
    private function normalizeStoredCredits(mixed $storedCredits): array
    {
        if (is_string($storedCredits) && $storedCredits !== '') {
            $decoded = json_decode($storedCredits, true);
            $storedCredits = is_array($decoded) ? $decoded : [];
        }

        if (!is_array($storedCredits)) {
            return [];
        }

        $normalized = [];
        foreach ($storedCredits as $credit) {
            if (!is_array($credit) || !is_string($credit['name'] ?? null)) {
                continue;
            }

            $name = $this->normalizeArtistName($credit['name']);
            if ($name === null) {
                continue;
            }

            $slug = is_string($credit['slug'] ?? null) && trim($credit['slug']) !== ''
                ? trim($credit['slug'])
                : $this->slugifyArtist($name);

            $roles = array_values(array_unique(array_filter(
                array_map(
                    fn (mixed $role): ?string => is_string($role) ? trim($role) : null,
                    is_array($credit['roles'] ?? null) ? $credit['roles'] : [],
                ),
            )));

            $normalized[$slug] = [
                'name' => $name,
                'slug' => $slug,
                'roles' => $this->sortRoles($roles === [] ? [self::ROLE_PRIMARY] : $roles),
            ];
        }

        return array_values($normalized);
    }

    /**
     * @param mixed $storedSlugs
     * @param array<int, array{name: string, slug: string, roles: array<int, string>}> $credits
     * @return array<int, string>
     */
    private function normalizeStoredSlugs(mixed $storedSlugs, array $credits): array
    {
        if (is_string($storedSlugs) && $storedSlugs !== '') {
            $decoded = json_decode($storedSlugs, true);
            $storedSlugs = is_array($decoded) ? $decoded : [];
        }

        if (!is_array($storedSlugs) || $storedSlugs === []) {
            return array_values(array_map(
                static fn (array $credit): string => $credit['slug'],
                $credits,
            ));
        }

        $slugs = [];
        foreach ($storedSlugs as $slug) {
            if (!is_string($slug)) {
                continue;
            }

            $slug = trim($slug);
            if ($slug !== '') {
                $slugs[$slug] = $slug;
            }
        }

        return array_values($slugs);
    }

    private function normalizeArtistName(?string $value): ?string
    {
        $value = $this->sanitizeText($value);
        if ($value === null) {
            return null;
        }

        $value = preg_replace('/\s*-\s*topic$/iu', '', $value) ?? $value;
        $value = preg_replace('/^[\s,\;\|\-\x{2013}\x{2014}]+|[\s,\;\|\-\x{2013}\x{2014}]+$/u', '', $value) ?? $value;

        return $value === '' ? null : $value;
    }

    private function cleanupTrackTitle(string $value): string
    {
        $value = preg_replace('/\s+/u', ' ', trim($value)) ?? trim($value);

        return preg_replace('/^[\s\-\x{2013}\x{2014}]+|[\s\-\x{2013}\x{2014}]+$/u', '', $value) ?? $value;
    }

    private function sanitizeText(mixed $value): ?string
    {
        if (!is_string($value)) {
            return null;
        }

        $value = trim($value);

        return $value === '' ? null : $value;
    }

    public function slugifyArtist(string $artistName): string
    {
        $slug = Str::slug($artistName);

        if ($slug !== '') {
            return $slug;
        }

        return 'artist-' . substr(sha1(mb_strtolower($artistName)), 0, 12);
    }
}
