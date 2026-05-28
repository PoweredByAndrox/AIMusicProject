<?php

namespace App\Services;

use getID3;

/**
 * MetadataExtractor Service
 * 
 * Extracts metadata from audio files using ID3 tags.
 * Requirements: 1.2
 */
class MetadataExtractor
{
    private getID3 $getID3;

    public function __construct()
    {
        $this->getID3 = new getID3();
    }

    /**
     * Extract metadata from an audio file.
     *
     * @param string $filePath Full path to the audio file
     * @return array{title: ?string, artist: ?string, album: ?string, year: ?int, duration: float}
     */
    public function extract(string $filePath): array
    {
        $fileInfo = $this->getID3->analyze($filePath);

        // Get duration
        $duration = $fileInfo['playtime_seconds'] ?? 0;

        // Get ID3 tags
        $tags = $fileInfo['tags']['id3v2'] ?? $fileInfo['tags']['id3v1'] ?? [];

        return [
            'title' => $this->getTag($tags, 'title'),
            'artist' => $this->getTag($tags, 'artist'),
            'album' => $this->getTag($tags, 'album'),
            'year' => $this->getYearTag($tags),
            'duration' => (float) $duration,
        ];
    }

    /**
     * Get a tag value from the tags array.
     *
     * @param array $tags
     * @param string $key
     * @return string|null
     */
    private function getTag(array $tags, string $key): ?string
    {
        if (isset($tags[$key]) && is_array($tags[$key]) && !empty($tags[$key][0])) {
            return trim($tags[$key][0]);
        }
        return null;
    }

    /**
     * Get year tag as integer.
     *
     * @param array $tags
     * @return int|null
     */
    private function getYearTag(array $tags): ?int
    {
        $year = $this->getTag($tags, 'year') ?? $this->getTag($tags, 'recording_time');
        
        if ($year !== null) {
            // Extract 4-digit year from various formats
            if (preg_match('/(\d{4})/', $year, $matches)) {
                return (int) $matches[1];
            }
        }
        
        return null;
    }
}
