<?php

namespace App\Jobs;

use App\Models\QueueTask;
use App\Models\Track;
use App\Models\UserTrackLike;
use App\Services\ArtistCreditParser;
use App\Services\YouTubeImporter;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Process;

/**
 * YouTubeDownloadJob
 *
 * Downloads audio from YouTube using yt-dlp.
 * Requirements: 2.1, 2.2, 2.3, 2.4
 */
class YouTubeDownloadJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public $tries = 3;
    public $backoff = 120;
    public $timeout = 600; // 10 minutes

    public function __construct(
        public string $url,
        public string $urlType,
        public ?string $resourceId = null,
        public int $maxVideos = 200,
        public ?int $userId = null,
        public bool $likeAll = false
    ) {
        $this->onQueue('youtube');
    }

    public function handle(): void
    {
        $importer = new YouTubeImporter();

        $this->logInfo('job_started', [
            'video_id' => $this->extractVideoId($this->url),
        ]);

        if ($this->urlType === YouTubeImporter::TYPE_VIDEO) {
            $this->downloadVideo($this->url);
        } elseif ($this->urlType === YouTubeImporter::TYPE_PLAYLIST) {
            $this->downloadPlaylist($this->url);
        } elseif ($this->urlType === YouTubeImporter::TYPE_CHANNEL) {
            $this->downloadChannel($this->url);
        }
    }

    private function downloadVideo(string $url): void
    {
        $videoId = $this->extractVideoId($url);

        $this->logInfo('video_import_started', [
            'video_id' => $videoId,
        ]);

        if ($videoId) {
            $existingTrack = Track::where('youtube_video_id', $videoId)->first();
            if ($existingTrack) {
                $this->logInfo('duplicate_by_video_id', [
                    'video_id' => $videoId,
                    'existing_track_id' => $existingTrack->id,
                    'existing_file_path' => $existingTrack->file_path,
                    'existing_title' => $existingTrack->title,
                    'existing_artist' => $existingTrack->artist,
                ]);
                $this->likeTrackForUser($existingTrack);
                return;
            }
        }

        $infoArgs = [
            'yt-dlp',
            '--no-cookies',
            '--extractor-args', 'youtube:player_client=android',
            '--print', '%(title)s',
            '--print', '%(uploader)s',
            '--print', '%(duration)s',
            '--no-download',
            $url,
        ];

        $infoResult = Process::timeout(60)->run($infoArgs);

        if (!$infoResult->successful()) {
            $this->logError('video_metadata_fetch_failed', [
                'video_id' => $videoId,
                'error_output' => $infoResult->errorOutput(),
            ]);
        }

        $info = explode("\n", trim($infoResult->output()));
        $rawTitle = $info[0] ?? 'Unknown Title';
        $channelName = $info[1] ?? null;
        $duration = (float) ($info[2] ?? 0);

        $parsed = app(ArtistCreditParser::class)->parseYouTubeMetadata($rawTitle, $channelName);
        $title = $parsed['title'];
        $artist = $parsed['artist'];

        $duplicateByContent = Track::where('title', $title)
            ->whereBetween('duration', [$duration - 5, $duration + 5])
            ->first();

        if ($duplicateByContent) {
            $this->logInfo('duplicate_by_content', [
                'video_id' => $videoId,
                'existing_track_id' => $duplicateByContent->id,
                'existing_file_path' => $duplicateByContent->file_path,
                'existing_title' => $duplicateByContent->title,
                'existing_artist' => $duplicateByContent->artist,
                'matched_title' => $title,
                'matched_artist' => $artist,
                'matched_duration' => $duration,
            ]);
            $this->likeTrackForUser($duplicateByContent);
            return;
        }

        $outputDir = storage_path('app/audio');

        if (!is_dir($outputDir) && !mkdir($outputDir, 0755, true) && !is_dir($outputDir)) {
            $this->logError('audio_output_dir_create_failed', [
                'video_id' => $videoId,
                'output_dir' => $outputDir,
            ]);
            throw new \RuntimeException('Failed to create audio output directory');
        }

        $probeFile = $outputDir . '/.write_probe_' . uniqid('', true);
        $probeHandle = @fopen($probeFile, 'wb');

        if ($probeHandle === false) {
            $this->logError('audio_output_dir_not_writable', [
                'video_id' => $videoId,
                'output_dir' => $outputDir,
            ]);
            throw new \RuntimeException('Audio output directory is not writable');
        }

        fclose($probeHandle);
        @unlink($probeFile);

        $filename = uniqid('yt_');
        $outputPath = $outputDir . '/' . $filename . '.%(ext)s';

        $args = [
            'yt-dlp',
            '--no-cookies',
            '--extractor-args', 'youtube:player_client=android',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--audio-quality', '0',
            '--output', $outputPath,
            '--no-playlist',
            $url,
        ];

        $result = Process::timeout(300)->run($args);

        if (!$result->successful()) {
            $this->logError('video_download_failed', [
                'video_id' => $videoId,
                'output_template' => $outputPath,
                'error_output' => $result->errorOutput(),
            ]);
            throw new \Exception('yt-dlp failed: ' . $result->errorOutput());
        }

        $expectedFile = $outputDir . '/' . $filename . '.mp3';

        if (!file_exists($expectedFile)) {
            $files = glob($outputDir . '/' . $filename . '.*');
            if (empty($files)) {
                $this->logError('downloaded_file_not_found', [
                    'video_id' => $videoId,
                    'output_template' => $outputPath,
                ]);
                throw new \Exception('Downloaded file not found');
            }
            $expectedFile = $files[0];
        }

        $relativePath = 'audio/' . basename($expectedFile);

        $track = Track::create([
            'title' => $title,
            'artist' => $artist,
            'artist_credits' => $parsed['artist_credits'],
            'artist_slugs' => $parsed['artist_slugs'],
            'duration' => $duration,
            'file_path' => $relativePath,
            'source' => 'youtube',
            'youtube_url' => $url,
            'youtube_video_id' => $videoId,
            'youtube_channel' => $channelName,
            'analysis_status' => 'pending',
        ]);

        if ($this->likeAll && $this->userId !== null) {
            $this->likeTrackForUser($track);
        }

        $this->logInfo('track_created', [
            'track_id' => $track->id,
            'video_id' => $videoId,
            'title' => $track->title,
            'artist' => $track->artist,
            'duration' => $track->duration,
            'file_path' => $track->file_path,
            'youtube_url' => $track->youtube_url,
            'youtube_channel' => $track->youtube_channel,
            'analysis_status' => $track->analysis_status,
        ]);

        TrackAnalysisJob::dispatch($track);

        $this->logInfo('track_analysis_dispatched', [
            'track_id' => $track->id,
            'video_id' => $videoId,
            'file_path' => $track->file_path,
        ]);
    }

    private function downloadPlaylist(string $url): void
    {
        $this->logInfo('playlist_expand_started');

        $result = Process::timeout(120)->run([
            'yt-dlp',
            '--js-runtimes', 'node',
            '--flat-playlist',
            '--print', 'url',
            $url,
        ]);

        if (!$result->successful()) {
            $this->logError('playlist_expand_failed', [
                'error_output' => $result->errorOutput(),
            ]);
            throw new \Exception('Failed to fetch playlist: ' . $result->errorOutput());
        }

        $videoUrls = array_filter(explode("\n", trim($result->output())));

        $this->logInfo('playlist_expand_completed', [
            'dispatched_count' => count($videoUrls),
        ]);

        foreach ($videoUrls as $index => $videoUrl) {
            self::dispatch($videoUrl, YouTubeImporter::TYPE_VIDEO, null, 200, $this->userId, $this->likeAll)
                ->delay(now()->addSeconds(5));

            $this->logInfo('playlist_child_dispatched', [
                'child_index' => $index + 1,
                'child_url' => $videoUrl,
                'child_video_id' => $this->extractVideoId($videoUrl),
            ]);
        }
    }

    private function downloadChannel(string $url): void
    {
        $this->logInfo('channel_expand_started');

        $args = [
            'yt-dlp',
            '--js-runtimes', 'node',
            '--flat-playlist',
            '--print', 'url',
        ];

        if ($this->maxVideos > 0) {
            $args[] = '--playlist-end';
            $args[] = (string) $this->maxVideos;
        }

        $args[] = $url;

        $result = Process::timeout(300)->run($args);

        if (!$result->successful()) {
            $this->logError('channel_expand_failed', [
                'error_output' => $result->errorOutput(),
            ]);
            throw new \Exception('Failed to fetch channel: ' . $result->errorOutput());
        }

        $videoUrls = array_filter(explode("\n", trim($result->output())));

        $this->logInfo('channel_expand_completed', [
            'dispatched_count' => count($videoUrls),
        ]);

        foreach ($videoUrls as $index => $videoUrl) {
            self::dispatch($videoUrl, YouTubeImporter::TYPE_VIDEO, null, 200, $this->userId, $this->likeAll)
                ->delay(now()->addSeconds(5));

            $this->logInfo('channel_child_dispatched', [
                'child_index' => $index + 1,
                'child_url' => $videoUrl,
                'child_video_id' => $this->extractVideoId($videoUrl),
            ]);
        }
    }

    public function failed(\Throwable $exception): void
    {
        $this->cleanupPartialDownloads();

        $this->logError('job_failed', [
            'video_id' => $this->extractVideoId($this->url),
            'exception_class' => $exception::class,
            'error_message' => $exception->getMessage(),
        ]);

        QueueTask::create([
            'task_type' => 'youtube_download',
            'payload' => [
                'url' => $this->url,
                'type' => $this->urlType,
            ],
            'status' => QueueTask::STATUS_FAILED,
            'error_message' => $exception->getMessage(),
        ]);
    }

    private function cleanupPartialDownloads(): void
    {
        $outputDir = storage_path('app/audio');
        if (!is_dir($outputDir)) {
            return;
        }

        $patterns = ['*.part', '*.ytdl', '*.webm', '*.mp4.part', '*.mp4.ytdl'];
        foreach ($patterns as $pattern) {
            $files = glob($outputDir . '/' . $pattern);
            foreach ($files as $file) {
                @unlink($file);
            }
        }
    }

    private function likeTrackForUser(Track $track): void
    {
        if (!$this->likeAll || $this->userId === null) {
            return;
        }

        UserTrackLike::updateOrCreate([
            'user_id' => $this->userId,
            'track_id' => $track->id,
        ], []);
    }

    private function extractVideoId(string $url): ?string
    {
        $patterns = [
            '/(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})/',
        ];

        foreach ($patterns as $pattern) {
            if (preg_match($pattern, $url, $matches)) {
                return $matches[1];
            }
        }

        return null;
    }

    private function logInfo(string $event, array $context = []): void
    {
        $this->writeStructuredLog('info', $event, $context);
    }

    private function logError(string $event, array $context = []): void
    {
        $this->writeStructuredLog('error', $event, $context);
    }

    private function writeStructuredLog(string $level, string $event, array $context = []): void
    {
        Log::channel('youtube_download')->{$level}($event, array_merge([
            'job' => self::class,
            'url' => $this->url,
            'url_type' => $this->urlType,
            'resource_id' => $this->resourceId,
            'max_videos' => $this->maxVideos,
            'user_id' => $this->userId,
            'like_all' => $this->likeAll,
        ], $context));
    }
}
