<?php

namespace App\Services;

use App\Models\Track;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Storage;
use RuntimeException;

class PublicRadioHlsService
{
    /**
     * @var array<int, string>
     */
    private const RADIO_BITRATE_LADDER = ['32k', '48k', '64k', '96k', '128k', '160k', '192k', '256k', '320k'];
    private const HLS_PACKAGE_VERSION = 'v6';
    private const HLS_SEGMENT_SECONDS = 6;

    /**
     * @return array{directory:string,master:string}
     */
    public function ensureHlsPackage(Track $track): array
    {
        if (!function_exists('proc_open')) {
            throw new RuntimeException('HLS packaging is not available in this runtime');
        }

        $sourcePath = $this->resolveSourcePath($track);
        $signature = $this->packageSignature($track, $sourcePath);

        $packageDirectory = storage_path('app/public_radio_hls/' . (int) $track->id . '/' . $signature);
        $masterPath = $packageDirectory . DIRECTORY_SEPARATOR . 'master.m3u8';

        if (is_file($masterPath)) {
            $this->publishCurrentHlsPackage((int) $track->id, $packageDirectory);

            return [
                'directory' => $packageDirectory,
                'master' => $masterPath,
            ];
        }

        $lockKey = 'public_radio_hls_package:' . $track->id . ':' . $signature;

        return Cache::lock($lockKey, 600)->block(15, function () use ($track, $packageDirectory, $masterPath, $sourcePath): array {
            if (is_file($masterPath)) {
                $this->publishCurrentHlsPackage((int) $track->id, $packageDirectory);

                return [
                    'directory' => $packageDirectory,
                    'master' => $masterPath,
                ];
            }

            if (!is_dir($packageDirectory) && !mkdir($packageDirectory, 0775, true) && !is_dir($packageDirectory)) {
                throw new RuntimeException('Failed to create HLS package directory');
            }

            foreach (self::RADIO_BITRATE_LADDER as $profile) {
                $profileDirectory = $packageDirectory . DIRECTORY_SEPARATOR . $profile;
                $playlistPath = $profileDirectory . DIRECTORY_SEPARATOR . 'index.m3u8';

                if (is_file($playlistPath)) {
                    continue;
                }

                if (!is_dir($profileDirectory) && !mkdir($profileDirectory, 0775, true) && !is_dir($profileDirectory)) {
                    throw new RuntimeException('Failed to create HLS profile directory');
                }

                $segmentPattern = $profileDirectory . DIRECTORY_SEPARATOR . 'segment_%03d.ts';
                $this->runFfmpegCommand([
                    'ffmpeg',
                    '-y',
                    '-v', 'error',
                    '-nostdin',
                    '-i', $sourcePath,
                    '-vn',
                    '-map', 'a:0',
                    '-c:a', 'aac',
                    '-b:a', $profile,
                    '-ac', '2',
                    '-ar', '44100',
                    '-f', 'hls',
                    '-hls_time', (string) self::HLS_SEGMENT_SECONDS,
                    '-hls_playlist_type', 'vod',
                    '-hls_flags', 'independent_segments',
                    '-hls_segment_filename', $segmentPattern,
                    $playlistPath,
                ], 'HLS profile generation failed');
            }

            $masterManifest = $this->buildHlsMasterManifest();
            if (@file_put_contents($masterPath, $masterManifest) === false) {
                throw new RuntimeException('Failed to write HLS master manifest');
            }

            $this->publishCurrentHlsPackage((int) $track->id, $packageDirectory);

            return [
                'directory' => $packageDirectory,
                'master' => $masterPath,
            ];
        });
    }

    public function currentMasterPublicUrl(int $trackId): ?string
    {
        return is_file($this->currentMasterPath($trackId))
            ? '/storage/public_radio_hls/' . $trackId . '/current/master.m3u8'
            : null;
    }

    public function currentMasterPath(int $trackId): string
    {
        return storage_path('app/public_radio_hls/' . $trackId . '/current/master.m3u8');
    }

    public function hasCurrentPackage(Track $track): bool
    {
        $trackId = (int) $track->id;
        $currentMasterPath = $this->currentMasterPath($trackId);
        if (!is_file($currentMasterPath)) {
            return false;
        }

        $currentSignatureMarker = dirname($currentMasterPath) . DIRECTORY_SEPARATOR . '.package-signature';
        if (!is_file($currentSignatureMarker)) {
            return false;
        }

        $currentSignature = trim((string) @file_get_contents($currentSignatureMarker));
        if ($currentSignature === '') {
            return false;
        }

        return hash_equals($this->packageSignature($track), $currentSignature);
    }

    public function masterPath(Track $track): string
    {
        return $this->ensureHlsPackage($track)['master'];
    }

    public function variantPath(Track $track, string $profile): string
    {
        $normalizedProfile = $this->normalizeHlsProfile($profile);
        if ($normalizedProfile === null) {
            throw new RuntimeException('HLS profile not found');
        }

        ['directory' => $packageDirectory] = $this->ensureHlsPackage($track);
        $playlistPath = $packageDirectory . DIRECTORY_SEPARATOR . $normalizedProfile . DIRECTORY_SEPARATOR . 'index.m3u8';

        if (!is_file($playlistPath)) {
            throw new RuntimeException('HLS playlist not found');
        }

        return $playlistPath;
    }

    public function segmentPath(Track $track, string $profile, string $segment): string
    {
        $normalizedProfile = $this->normalizeHlsProfile($profile);
        if ($normalizedProfile === null) {
            throw new RuntimeException('HLS profile not found');
        }

        if (basename($segment) !== $segment) {
            throw new RuntimeException('Invalid segment name');
        }

        ['directory' => $packageDirectory] = $this->ensureHlsPackage($track);
        $segmentPath = $packageDirectory . DIRECTORY_SEPARATOR . $normalizedProfile . DIRECTORY_SEPARATOR . $segment;

        if (!is_file($segmentPath)) {
            throw new RuntimeException('HLS segment not found');
        }

        return $segmentPath;
    }

    public function normalizeHlsProfile(string $profile): ?string
    {
        $normalized = strtolower(trim($profile));
        if ($normalized === '') {
            return null;
        }

        if (!str_ends_with($normalized, 'k')) {
            $normalized .= 'k';
        }

        return in_array($normalized, self::RADIO_BITRATE_LADDER, true) ? $normalized : null;
    }

    private function resolveSourcePath(Track $track): string
    {
        if (!$track->file_path || !Storage::disk('local')->exists($track->file_path)) {
            throw new RuntimeException('Audio file not found');
        }

        return Storage::disk('local')->path($track->file_path);
    }

    private function buildHlsMasterManifest(): string
    {
        $lines = ['#EXTM3U', '#EXT-X-VERSION:3'];

        foreach (self::RADIO_BITRATE_LADDER as $profile) {
            $bitrateKbps = (int) str_replace('k', '', $profile);
            $bandwidth = $bitrateKbps * 1024;
            $lines[] = sprintf('#EXT-X-STREAM-INF:BANDWIDTH=%d,CODECS="mp4a.40.2"', $bandwidth);
            $lines[] = $profile . '/index.m3u8';
        }

        return implode("\n", $lines) . "\n";
    }

    private function packageSignature(Track $track, ?string $sourcePath = null): string
    {
        $resolvedSourcePath = $sourcePath ?? $this->resolveSourcePath($track);

        return sha1(implode('|', [
            self::HLS_PACKAGE_VERSION,
            (string) $track->id,
            (string) (filesize($resolvedSourcePath) ?: 0),
            (string) (filemtime($resolvedSourcePath) ?: 0),
            implode(',', self::RADIO_BITRATE_LADDER),
            (string) self::HLS_SEGMENT_SECONDS,
        ]));
    }

    /**
     * @param array<int, string> $command
     */
    private function runFfmpegCommand(array $command, string $fallbackMessage): void
    {
        $descriptors = [
            0 => ['pipe', 'r'],
            1 => ['pipe', 'w'],
            2 => ['pipe', 'w'],
        ];

        $process = proc_open($command, $descriptors, $pipes);
        if (!is_resource($process)) {
            throw new RuntimeException($fallbackMessage);
        }

        fclose($pipes[0]);
        stream_get_contents($pipes[1]);
        fclose($pipes[1]);
        $stderr = stream_get_contents($pipes[2]);
        fclose($pipes[2]);
        $exitCode = proc_close($process);

        if ($exitCode !== 0) {
            throw new RuntimeException(trim((string) $stderr) !== '' ? trim((string) $stderr) : $fallbackMessage);
        }
    }

    private function publishCurrentHlsPackage(int $trackId, string $packageDirectory): void
    {
        $trackDirectory = storage_path('app/public_radio_hls/' . $trackId);
        $currentDirectory = $trackDirectory . DIRECTORY_SEPARATOR . 'current';
        $packageSignature = basename($packageDirectory);
        $currentSignatureMarker = $currentDirectory . DIRECTORY_SEPARATOR . '.package-signature';

        if (realpath($currentDirectory) === realpath($packageDirectory)) {
            return;
        }

        if (is_dir($currentDirectory) && is_file($currentSignatureMarker)) {
            $currentSignature = trim((string) @file_get_contents($currentSignatureMarker));
            if ($currentSignature === $packageSignature) {
                return;
            }
        }

        if (is_link($currentDirectory) || is_file($currentDirectory)) {
            @unlink($currentDirectory);
        } elseif (is_dir($currentDirectory)) {
            $this->deleteDirectory($currentDirectory);
        }

        if (function_exists('symlink') && @symlink($packageSignature, $currentDirectory)) {
            return;
        }

        $this->mirrorDirectory($packageDirectory, $currentDirectory);
        @file_put_contents($currentSignatureMarker, $packageSignature);
    }

    private function mirrorDirectory(string $sourceDirectory, string $targetDirectory): void
    {
        if (!is_dir($targetDirectory) && !mkdir($targetDirectory, 0775, true) && !is_dir($targetDirectory)) {
            throw new RuntimeException('Failed to create static HLS mirror directory');
        }

        $items = scandir($sourceDirectory);
        if ($items === false) {
            throw new RuntimeException('Failed to read HLS package directory');
        }

        foreach ($items as $item) {
            if ($item === '.' || $item === '..') {
                continue;
            }

            $sourcePath = $sourceDirectory . DIRECTORY_SEPARATOR . $item;
            $targetPath = $targetDirectory . DIRECTORY_SEPARATOR . $item;

            if (is_dir($sourcePath)) {
                $this->mirrorDirectory($sourcePath, $targetPath);
                continue;
            }

            if (@copy($sourcePath, $targetPath) === false) {
                throw new RuntimeException('Failed to copy HLS package into static mirror');
            }
        }
    }

    private function deleteDirectory(string $directory): void
    {
        $items = scandir($directory);
        if ($items === false) {
            return;
        }

        foreach ($items as $item) {
            if ($item === '.' || $item === '..') {
                continue;
            }

            $path = $directory . DIRECTORY_SEPARATOR . $item;
            if (is_dir($path) && !is_link($path)) {
                $this->deleteDirectory($path);
                continue;
            }

            @unlink($path);
        }

        @rmdir($directory);
    }
}
