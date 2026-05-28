<?php

namespace App\Services;

use App\Models\Track;
use Illuminate\Support\Facades\Storage;

class TrackPlaybackManifestService
{
    private const MANIFEST_VERSION = 1;

    /**
     * @var array<int, array{default_mode_key: string, modes: array<string, array<string, mixed>>, stems: array<string, array<string, mixed>>, lyrics: array<string, mixed>}>
     */
    private array $catalogCache = [];

    /**
     * @return array<string, mixed>
     */
    public function manifestForTrack(Track $track): array
    {
        $catalog = $this->catalogForTrack($track);

        return [
            'track_id' => (int) $track->id,
            'manifest_version' => self::MANIFEST_VERSION,
            'default_mode_key' => $catalog['default_mode_key'],
            'modes' => array_values(array_map(
                fn (array $mode): array => $this->publicModePayload($track, $mode),
                $catalog['modes'],
            )),
            'stems' => array_values(array_map(
                fn (array $stem): array => $this->publicStemPayload($track, $stem),
                $catalog['stems'],
            )),
            'lyrics' => $catalog['lyrics'],
            'features' => [
                'karaoke' => count($catalog['modes']) > 1,
                'stem_mixing' => count($catalog['stems']) > 0,
                'lyrics' => (bool) ($catalog['lyrics']['available'] ?? false),
                'synced_lyrics' => ($catalog['lyrics']['kind'] ?? 'none') === 'timed_text',
            ],
        ];
    }

    /**
     * @return array<string, mixed>|null
     */
    public function resolveModeSource(Track $track, string $modeKey): ?array
    {
        $catalog = $this->catalogForTrack($track);
        return $catalog['modes'][$modeKey] ?? null;
    }

    /**
     * @return array<string, mixed>|null
     */
    public function resolveStemSource(Track $track, string $stemKey): ?array
    {
        $catalog = $this->catalogForTrack($track);
        return $catalog['stems'][$stemKey] ?? null;
    }

    /**
     * @return array{default_mode_key: string, modes: array<string, array<string, mixed>>, stems: array<string, array<string, mixed>>, lyrics: array<string, mixed>}
     */
    private function catalogForTrack(Track $track): array
    {
        $cacheKey = (int) $track->id;
        if (array_key_exists($cacheKey, $this->catalogCache)) {
            return $this->catalogCache[$cacheKey];
        }

        $absoluteTrackPath = $this->absoluteTrackPath($track);
        if ($absoluteTrackPath === null) {
            return $this->catalogCache[$cacheKey] = [
                'default_mode_key' => 'original',
                'modes' => [],
                'stems' => [],
                'lyrics' => $this->emptyLyricsPayload(),
            ];
        }

        $rawManifest = $this->readRawManifest($absoluteTrackPath);
        $modes = [
            'original' => $this->buildOriginalMode($track, $absoluteTrackPath),
        ];

        foreach ($this->definitionMap($rawManifest['modes'] ?? null) as $modeKey => $modeConfig) {
            if ($modeKey === 'original') {
                continue;
            }

            $mode = $this->buildConfiguredMode($modeKey, $modeConfig, $absoluteTrackPath);
            if ($mode !== null) {
                $modes[$modeKey] = $mode;
            }
        }

        $defaultModeKey = (string) ($rawManifest['default_mode_key'] ?? 'original');
        if (!array_key_exists($defaultModeKey, $modes)) {
            $defaultModeKey = 'original';
        }

        $stems = [];
        foreach ($this->definitionMap($rawManifest['stems'] ?? null) as $stemKey => $stemConfig) {
            $stem = $this->buildConfiguredStem($stemKey, $stemConfig, $absoluteTrackPath);
            if ($stem !== null) {
                $stems[$stemKey] = $stem;
            }
        }

        return $this->catalogCache[$cacheKey] = [
            'default_mode_key' => $defaultModeKey,
            'modes' => $modes,
            'stems' => $stems,
            'lyrics' => $this->buildLyricsPayload($rawManifest['lyrics'] ?? null, $absoluteTrackPath),
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function buildOriginalMode(Track $track, string $absoluteTrackPath): array
    {
        return [
            'key' => 'original',
            'label' => 'Original',
            'description' => 'Primary full mix source',
            'kind' => 'full_mix',
            'recommended' => true,
            'is_original' => true,
            'path' => $absoluteTrackPath,
            'file_name' => basename($absoluteTrackPath),
            'file_size_bytes' => filesize($absoluteTrackPath) ?: 0,
            'mime_type' => $this->detectMimeType($absoluteTrackPath, null),
            'extension' => strtolower(pathinfo($absoluteTrackPath, PATHINFO_EXTENSION)),
        ];
    }

    /**
     * @param array<string, mixed> $modeConfig
     * @return array<string, mixed>|null
     */
    private function buildConfiguredMode(string $modeKey, array $modeConfig, string $absoluteTrackPath): ?array
    {
        $resolvedPath = $this->resolveConfiguredFile($absoluteTrackPath, $modeConfig['path'] ?? null);
        if ($resolvedPath === null) {
            return null;
        }

        return [
            'key' => $modeKey,
            'label' => (string) ($modeConfig['label'] ?? $this->humanizeKey($modeKey)),
            'description' => (string) ($modeConfig['description'] ?? ''),
            'kind' => (string) ($modeConfig['kind'] ?? 'alternate_mix'),
            'recommended' => (bool) ($modeConfig['recommended'] ?? false),
            'is_original' => false,
            'path' => $resolvedPath,
            'file_name' => basename($resolvedPath),
            'file_size_bytes' => filesize($resolvedPath) ?: 0,
            'mime_type' => $this->detectMimeType($resolvedPath, $modeConfig['mime_type'] ?? null),
            'extension' => strtolower(pathinfo($resolvedPath, PATHINFO_EXTENSION)),
        ];
    }

    /**
     * @param array<string, mixed> $stemConfig
     * @return array<string, mixed>|null
     */
    private function buildConfiguredStem(string $stemKey, array $stemConfig, string $absoluteTrackPath): ?array
    {
        $resolvedPath = $this->resolveConfiguredFile($absoluteTrackPath, $stemConfig['path'] ?? null);
        if ($resolvedPath === null) {
            return null;
        }

        return [
            'key' => $stemKey,
            'label' => (string) ($stemConfig['label'] ?? $this->humanizeKey($stemKey)),
            'description' => (string) ($stemConfig['description'] ?? ''),
            'family' => (string) ($stemConfig['family'] ?? 'default'),
            'kind' => (string) ($stemConfig['kind'] ?? 'stem'),
            'default_enabled' => (bool) ($stemConfig['default_enabled'] ?? true),
            'path' => $resolvedPath,
            'file_name' => basename($resolvedPath),
            'file_size_bytes' => filesize($resolvedPath) ?: 0,
            'mime_type' => $this->detectMimeType($resolvedPath, $stemConfig['mime_type'] ?? null),
            'extension' => strtolower(pathinfo($resolvedPath, PATHINFO_EXTENSION)),
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function publicModePayload(Track $track, array $mode): array
    {
        return [
            'key' => (string) $mode['key'],
            'label' => (string) $mode['label'],
            'description' => (string) $mode['description'],
            'kind' => (string) $mode['kind'],
            'recommended' => (bool) ($mode['recommended'] ?? false),
            'is_original' => (bool) ($mode['is_original'] ?? false),
            'file_name' => (string) ($mode['file_name'] ?? ''),
            'file_size_bytes' => (int) ($mode['file_size_bytes'] ?? 0),
            'mime_type' => $mode['mime_type'] ?? null,
            'extension' => $mode['extension'] ?? null,
            'stream_url' => '/api/tracks/' . $track->id . '/playback/modes/' . $mode['key'] . '/stream',
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function publicStemPayload(Track $track, array $stem): array
    {
        return [
            'key' => (string) $stem['key'],
            'label' => (string) $stem['label'],
            'description' => (string) $stem['description'],
            'family' => (string) ($stem['family'] ?? 'default'),
            'kind' => (string) ($stem['kind'] ?? 'stem'),
            'default_enabled' => (bool) ($stem['default_enabled'] ?? true),
            'file_name' => (string) ($stem['file_name'] ?? ''),
            'file_size_bytes' => (int) ($stem['file_size_bytes'] ?? 0),
            'mime_type' => $stem['mime_type'] ?? null,
            'extension' => $stem['extension'] ?? null,
            'stream_url' => '/api/tracks/' . $track->id . '/playback/stems/' . $stem['key'] . '/stream',
        ];
    }

    private function absoluteTrackPath(Track $track): ?string
    {
        if (!is_string($track->file_path) || trim($track->file_path) === '') {
            return null;
        }

        if (!Storage::disk('local')->exists($track->file_path)) {
            return null;
        }

        return Storage::disk('local')->path($track->file_path);
    }

    /**
     * @return array<string, mixed>
     */
    private function readRawManifest(string $absoluteTrackPath): array
    {
        $manifestPath = $absoluteTrackPath . '.playback.json';
        if (!is_file($manifestPath)) {
            return [];
        }

        $decoded = json_decode((string) file_get_contents($manifestPath), true);
        return is_array($decoded) ? $decoded : [];
    }

    /**
     * @return array<string, array<string, mixed>>
     */
    private function definitionMap(mixed $rawDefinitions): array
    {
        if (!is_array($rawDefinitions)) {
            return [];
        }

        $definitions = [];
        foreach ($rawDefinitions as $key => $definition) {
            if (!is_array($definition)) {
                continue;
            }

            $resolvedKey = is_string($key)
                ? trim($key)
                : trim((string) ($definition['key'] ?? ''));

            if ($resolvedKey === '') {
                continue;
            }

            $definitions[$resolvedKey] = $definition;
        }

        return $definitions;
    }

    private function resolveConfiguredFile(string $absoluteTrackPath, mixed $candidate): ?string
    {
        if (!is_string($candidate) || trim($candidate) === '') {
            return null;
        }

        $normalized = str_replace('\\', '/', trim($candidate));
        if ($normalized === '' || str_contains($normalized, "\0")) {
            return null;
        }

        $storageRoot = realpath(storage_path('app'));
        if ($storageRoot === false) {
            return null;
        }

        $relativeCandidate = ltrim($normalized, '/');
        $searchPaths = [
            storage_path('app/' . str_replace('/', DIRECTORY_SEPARATOR, $relativeCandidate)),
            dirname($absoluteTrackPath) . DIRECTORY_SEPARATOR . str_replace('/', DIRECTORY_SEPARATOR, $relativeCandidate),
        ];

        foreach ($searchPaths as $searchPath) {
            if (!is_file($searchPath)) {
                continue;
            }

            $realPath = realpath($searchPath);
            if ($realPath === false || !$this->pathIsWithinRoot($realPath, $storageRoot)) {
                continue;
            }

            return $realPath;
        }

        return null;
    }

    private function pathIsWithinRoot(string $candidatePath, string $rootPath): bool
    {
        $candidate = str_replace('\\', '/', $candidatePath);
        $root = rtrim(str_replace('\\', '/', $rootPath), '/');

        return $candidate === $root || str_starts_with($candidate, $root . '/');
    }

    private function detectMimeType(string $path, mixed $preferredMimeType): string
    {
        if (is_string($preferredMimeType) && trim($preferredMimeType) !== '') {
            return trim($preferredMimeType);
        }

        return mime_content_type($path) ?: 'audio/mpeg';
    }

    private function humanizeKey(string $key): string
    {
        return ucwords(str_replace(['_', '-'], ' ', $key));
    }

    /**
     * @return array<string, mixed>
     */
    private function buildLyricsPayload(mixed $rawLyrics, string $absoluteTrackPath): array
    {
        if (!is_array($rawLyrics)) {
            return $this->emptyLyricsPayload();
        }

        $inlineCues = $this->normalizeCueCollection($rawLyrics['cues'] ?? null);
        if ($inlineCues !== []) {
            return $this->timedLyricsPayload($inlineCues, $rawLyrics['format'] ?? 'json');
        }

        if (is_string($rawLyrics['text'] ?? null) && trim($rawLyrics['text']) !== '') {
            return $this->plainLyricsPayload((string) $rawLyrics['text'], $rawLyrics['format'] ?? 'plain_text');
        }

        $resolvedPath = $this->resolveConfiguredFile($absoluteTrackPath, $rawLyrics['path'] ?? null);
        if ($resolvedPath === null) {
            return $this->emptyLyricsPayload();
        }

        $extension = strtolower(pathinfo($resolvedPath, PATHINFO_EXTENSION));
        $contents = file_get_contents($resolvedPath);
        if ($contents === false) {
            return $this->emptyLyricsPayload();
        }

        if ($extension === 'json') {
            $decoded = json_decode($contents, true);
            if (is_array($decoded)) {
                $jsonCues = $this->normalizeCueCollection($decoded['cues'] ?? $decoded);
                if ($jsonCues !== []) {
                    return $this->timedLyricsPayload($jsonCues, (string) ($decoded['format'] ?? 'json'));
                }

                if (is_string($decoded['text'] ?? null) && trim($decoded['text']) !== '') {
                    return $this->plainLyricsPayload((string) $decoded['text'], (string) ($decoded['format'] ?? 'plain_text'));
                }
            }

            return $this->emptyLyricsPayload();
        }

        if ($extension === 'lrc') {
            $parsed = $this->parseLrc($contents);
            return $parsed !== [] ? $this->timedLyricsPayload($parsed, 'lrc') : $this->emptyLyricsPayload();
        }

        return trim($contents) !== ''
            ? $this->plainLyricsPayload($contents, $extension !== '' ? $extension : 'plain_text')
            : $this->emptyLyricsPayload();
    }

    /**
     * @return array<int, array{start_ms: int, end_ms: int|null, text: string}>
     */
    private function normalizeCueCollection(mixed $rawCues): array
    {
        if (!is_array($rawCues)) {
            return [];
        }

        $normalized = [];
        foreach ($rawCues as $rawCue) {
            if (!is_array($rawCue)) {
                continue;
            }

            $text = trim((string) ($rawCue['text'] ?? ''));
            if ($text === '') {
                continue;
            }

            $startMs = $this->extractMilliseconds($rawCue, ['start_ms', 'startMs', 'start']);
            if ($startMs === null) {
                $startSec = $this->extractFloat($rawCue, ['start_sec', 'startSec']);
                if ($startSec !== null) {
                    $startMs = (int) round($startSec * 1000);
                }
            }

            if ($startMs === null) {
                continue;
            }

            $endMs = $this->extractMilliseconds($rawCue, ['end_ms', 'endMs', 'end']);
            if ($endMs === null) {
                $endSec = $this->extractFloat($rawCue, ['end_sec', 'endSec']);
                if ($endSec !== null) {
                    $endMs = (int) round($endSec * 1000);
                }
            }

            if ($endMs !== null && $endMs < $startMs) {
                $endMs = $startMs;
            }

            $normalized[] = [
                'start_ms' => $startMs,
                'end_ms' => $endMs,
                'text' => $text,
            ];
        }

        usort($normalized, static fn (array $left, array $right): int => $left['start_ms'] <=> $right['start_ms']);

        for ($index = 0; $index < count($normalized); $index++) {
            if ($normalized[$index]['end_ms'] !== null) {
                continue;
            }

            $nextCue = $normalized[$index + 1] ?? null;
            $normalized[$index]['end_ms'] = is_array($nextCue) ? $nextCue['start_ms'] : null;
        }

        return $normalized;
    }

    /**
     * @return array<int, array{start_ms: int, end_ms: int|null, text: string}>
     */
    private function parseLrc(string $contents): array
    {
        $parsed = [];
        $lines = preg_split('/\r\n|\r|\n/', $contents) ?: [];

        foreach ($lines as $line) {
            if (!is_string($line) || trim($line) === '') {
                continue;
            }

            preg_match_all('/\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]/', $line, $matches, PREG_SET_ORDER);
            if ($matches === []) {
                continue;
            }

            $text = trim((string) preg_replace('/\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]/', '', $line));
            if ($text === '') {
                continue;
            }

            foreach ($matches as $match) {
                $minutes = (int) ($match[1] ?? 0);
                $seconds = (int) ($match[2] ?? 0);
                $fraction = (string) ($match[3] ?? '');
                $millis = match (strlen($fraction)) {
                    1 => (int) $fraction * 100,
                    2 => (int) $fraction * 10,
                    3 => (int) $fraction,
                    default => 0,
                };

                $parsed[] = [
                    'start_ms' => ($minutes * 60 * 1000) + ($seconds * 1000) + $millis,
                    'end_ms' => null,
                    'text' => $text,
                ];
            }
        }

        usort($parsed, static fn (array $left, array $right): int => $left['start_ms'] <=> $right['start_ms']);

        for ($index = 0; $index < count($parsed); $index++) {
            $nextCue = $parsed[$index + 1] ?? null;
            $parsed[$index]['end_ms'] = is_array($nextCue) ? $nextCue['start_ms'] : null;
        }

        return $parsed;
    }

    /**
     * @param array<int, array{start_ms: int, end_ms: int|null, text: string}> $cues
     * @return array<string, mixed>
     */
    private function timedLyricsPayload(array $cues, mixed $format): array
    {
        return [
            'available' => true,
            'kind' => 'timed_text',
            'format' => is_string($format) && trim($format) !== '' ? trim($format) : 'timed_text',
            'text' => implode("\n", array_map(static fn (array $cue): string => $cue['text'], $cues)),
            'cues' => $cues,
            'line_count' => count($cues),
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function plainLyricsPayload(string $text, mixed $format): array
    {
        $trimmed = trim($text);

        return [
            'available' => $trimmed !== '',
            'kind' => 'plain_text',
            'format' => is_string($format) && trim($format) !== '' ? trim($format) : 'plain_text',
            'text' => $trimmed,
            'cues' => [],
            'line_count' => $trimmed === '' ? 0 : count(preg_split('/\r\n|\r|\n/', $trimmed) ?: []),
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function emptyLyricsPayload(): array
    {
        return [
            'available' => false,
            'kind' => 'none',
            'format' => null,
            'text' => null,
            'cues' => [],
            'line_count' => 0,
        ];
    }

    /**
     * @param array<string, mixed> $rawCue
     */
    private function extractMilliseconds(array $rawCue, array $keys): ?int
    {
        foreach ($keys as $key) {
            if (!array_key_exists($key, $rawCue)) {
                continue;
            }

            $value = $rawCue[$key];
            if (is_int($value)) {
                return $value;
            }

            if (is_float($value) || (is_string($value) && is_numeric($value))) {
                return (int) round((float) $value);
            }
        }

        return null;
    }

    /**
     * @param array<string, mixed> $rawCue
     */
    private function extractFloat(array $rawCue, array $keys): ?float
    {
        foreach ($keys as $key) {
            if (!array_key_exists($key, $rawCue)) {
                continue;
            }

            $value = $rawCue[$key];
            if (is_int($value) || is_float($value)) {
                return (float) $value;
            }

            if (is_string($value) && is_numeric($value)) {
                return (float) $value;
            }
        }

        return null;
    }
}
