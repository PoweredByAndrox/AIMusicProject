<?php

namespace App\Services;

/**
 * YouTubeImporter Service
 * 
 * Validates and parses YouTube URLs for import.
 * Requirements: 2.1, 2.2, 2.3, 2.5
 */
class YouTubeImporter
{
    /**
     * URL type constants
     */
    public const TYPE_VIDEO = 'video';
    public const TYPE_PLAYLIST = 'playlist';
    public const TYPE_CHANNEL = 'channel';
    public const TYPE_INVALID = 'invalid';

    /**
     * YouTube URL patterns
     */
    private const PATTERNS = [
        // Video URLs
        'video' => [
            '/^https?:\/\/(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})/',
            '/^https?:\/\/youtu\.be\/([a-zA-Z0-9_-]{11})/',
            '/^https?:\/\/(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})/',
            '/^https?:\/\/(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]{11})/',
        ],
        // Playlist URLs
        'playlist' => [
            '/^https?:\/\/(?:www\.)?youtube\.com\/playlist\?list=([a-zA-Z0-9_-]+)/',
            '/^https?:\/\/(?:www\.)?youtube\.com\/watch\?.*list=([a-zA-Z0-9_-]+)/',
        ],
        // Channel URLs
        'channel' => [
            '/^https?:\/\/(?:www\.)?youtube\.com\/channel\/([a-zA-Z0-9_-]+)/',
            '/^https?:\/\/(?:www\.)?youtube\.com\/@([a-zA-Z0-9_-]+)/',
            '/^https?:\/\/(?:www\.)?youtube\.com\/c\/([a-zA-Z0-9_-]+)/',
            '/^https?:\/\/(?:www\.)?youtube\.com\/user\/([a-zA-Z0-9_-]+)/',
        ],
    ];

    /**
     * Validate and determine the type of YouTube URL.
     *
     * @param string $url
     * @return array{type: string, id: ?string, valid: bool}
     */
    public function validateUrl(string $url): array
    {
        $url = trim($url);

        // Check for playlist first (can contain video ID too)
        foreach (self::PATTERNS['playlist'] as $pattern) {
            if (preg_match($pattern, $url, $matches)) {
                return [
                    'type' => self::TYPE_PLAYLIST,
                    'id' => $matches[1],
                    'valid' => true,
                ];
            }
        }

        // Check for video
        foreach (self::PATTERNS['video'] as $pattern) {
            if (preg_match($pattern, $url, $matches)) {
                return [
                    'type' => self::TYPE_VIDEO,
                    'id' => $matches[1],
                    'valid' => true,
                ];
            }
        }

        // Check for channel
        foreach (self::PATTERNS['channel'] as $pattern) {
            if (preg_match($pattern, $url, $matches)) {
                return [
                    'type' => self::TYPE_CHANNEL,
                    'id' => $matches[1],
                    'valid' => true,
                ];
            }
        }

        return [
            'type' => self::TYPE_INVALID,
            'id' => null,
            'valid' => false,
        ];
    }

    /**
     * Check if URL is a valid YouTube URL.
     *
     * @param string $url
     * @return bool
     */
    public function isValidUrl(string $url): bool
    {
        return $this->validateUrl($url)['valid'];
    }

    /**
     * Get the type of YouTube URL.
     *
     * @param string $url
     * @return string
     */
    public function getUrlType(string $url): string
    {
        return $this->validateUrl($url)['type'];
    }

    /**
     * Extract video ID from URL.
     *
     * @param string $url
     * @return string|null
     */
    public function extractVideoId(string $url): ?string
    {
        $result = $this->validateUrl($url);
        return $result['type'] === self::TYPE_VIDEO ? $result['id'] : null;
    }

    /**
     * Extract playlist ID from URL.
     *
     * @param string $url
     * @return string|null
     */
    public function extractPlaylistId(string $url): ?string
    {
        $result = $this->validateUrl($url);
        return $result['type'] === self::TYPE_PLAYLIST ? $result['id'] : null;
    }

    /**
     * Extract channel ID from URL.
     *
     * @param string $url
     * @return string|null
     */
    public function extractChannelId(string $url): ?string
    {
        $result = $this->validateUrl($url);
        return $result['type'] === self::TYPE_CHANNEL ? $result['id'] : null;
    }
}
