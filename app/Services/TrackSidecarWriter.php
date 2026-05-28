<?php

namespace App\Services;

use App\Models\Track;

class TrackSidecarWriter
{
    public function writeForTrack(Track $track, bool $withSha256 = false, ?string $generatedAt = null): ?string
    {
        $relativePath = $this->normalizeRelativePath($track->file_path);
        if ($relativePath === null) {
            return null;
        }

        $absolutePath = $this->absoluteAudioPath($relativePath);
        if (!is_file($absolutePath)) {
            return null;
        }

        return $this->writeForFile($absolutePath, $track, $withSha256, $generatedAt);
    }

    public function writeForFile(string $absolutePath, ?Track $track, bool $withSha256 = false, ?string $generatedAt = null): string
    {
        $relativePath = $this->relativePathFromAbsolute($absolutePath);
        $payload = $this->buildPayload(
            $absolutePath,
            $relativePath,
            $track,
            $withSha256,
            $generatedAt ?? now()->toAtomString(),
        );

        $sidecarPath = $this->sidecarPathFromAbsolute($absolutePath);
        file_put_contents(
            $sidecarPath,
            json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE) . PHP_EOL
        );

        return $sidecarPath;
    }

    public function deleteForRelativePath(?string $relativePath): void
    {
        $normalized = $this->normalizeRelativePath($relativePath);
        if ($normalized === null) {
            return;
        }

        $sidecarPath = $this->sidecarPathFromAbsolute($this->absoluteAudioPath($normalized));
        if (is_file($sidecarPath)) {
            unlink($sidecarPath);
        }
    }

    private function buildPayload(
        string $absolutePath,
        string $relativePath,
        ?Track $track,
        bool $withSha256,
        string $generatedAt
    ): array {
        return [
            'sidecar_version' => 1,
            'generated_at' => $generatedAt,
            'audio' => [
                'relative_path' => $relativePath,
                'basename' => basename($absolutePath),
                'size_bytes' => filesize($absolutePath) ?: null,
                'last_modified_utc' => $this->formatMtime($absolutePath),
                'sha256' => $withSha256 ? hash_file('sha256', $absolutePath) : null,
            ],
            'track' => $track?->toArray(),
            'missing_track_row' => $track === null,
        ];
    }

    private function normalizeRelativePath(?string $relativePath): ?string
    {
        if (!is_string($relativePath) || trim($relativePath) === '') {
            return null;
        }

        return str_replace('\\', '/', ltrim(trim($relativePath), '/'));
    }

    private function absoluteAudioPath(string $relativePath): string
    {
        return storage_path('app/' . $relativePath);
    }

    private function relativePathFromAbsolute(string $absolutePath): string
    {
        return 'audio/' . basename($absolutePath);
    }

    private function sidecarPathFromAbsolute(string $absolutePath): string
    {
        return $absolutePath . '.meta.json';
    }

    private function formatMtime(string $filePath): ?string
    {
        $mtime = @filemtime($filePath);
        if ($mtime === false) {
            return null;
        }

        return gmdate('c', $mtime);
    }
}
