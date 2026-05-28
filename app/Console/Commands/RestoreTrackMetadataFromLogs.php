<?php

namespace App\Console\Commands;

use App\Models\Track;
use App\Support\AppStoragePath;
use Illuminate\Console\Command;

class RestoreTrackMetadataFromLogs extends Command
{
    protected $signature = 'tracks:restore-metadata-from-logs
                            {--dry-run : Show what would be restored without writing to the database}
                            {--limit=0 : Limit number of track rows to update (0 = no limit)}
                            {--force : Skip confirmation}';

    protected $description = 'Restore track metadata from runtime laravel.log SQL traces by matching file_path';

    public function handle(): int
    {
        $logPath = AppStoragePath::log('laravel.log');
        if (!is_file($logPath)) {
            $this->error("Runtime log not found: {$logPath}");
            return 1;
        }

        $recoveredByPath = $this->buildRecoveryMap($logPath);
        if ($recoveredByPath === []) {
            $this->warn('No recoverable track metadata found in runtime log.');
            return 0;
        }

        $query = Track::query()
            ->whereIn('file_path', array_keys($recoveredByPath))
            ->where(function ($builder) {
                $builder
                    ->where('source', 'import')
                    ->orWhere('title', 'like', 'YouTube Import %');
            })
            ->orderBy('id');

        $candidates = $query->get(['id', 'file_path', 'title', 'artist', 'source']);
        if ($candidates->isEmpty()) {
            $this->warn('No current placeholder tracks match the recoverable file paths.');
            return 0;
        }

        $limit = max(0, (int) $this->option('limit'));
        if ($limit > 0) {
            $candidates = $candidates->take($limit);
        }

        $this->info('Recoverable log entries: ' . count($recoveredByPath));
        $this->info('Matching placeholder tracks: ' . $candidates->count());

        if ($this->option('dry-run')) {
            foreach ($candidates->take(10) as $track) {
                $recovered = $recoveredByPath[$track->file_path];
                $this->line(sprintf(
                    '#%d %s => %s / %s',
                    $track->id,
                    $track->file_path,
                    $recovered['title'],
                    $recovered['artist'] ?? 'null'
                ));
            }

            return 0;
        }

        if (!$this->option('force') && !$this->confirm('Update matched tracks from runtime log metadata?')) {
            $this->info('Cancelled.');
            return 0;
        }

        $updated = 0;
        foreach ($candidates as $track) {
            $recovered = $recoveredByPath[$track->file_path];
            Track::query()
                ->whereKey($track->id)
                ->update([
                    'title' => $recovered['title'],
                    'artist' => $recovered['artist'],
                    'source' => 'youtube',
                    'youtube_url' => $recovered['youtube_url'],
                    'youtube_video_id' => $recovered['youtube_video_id'],
                    'youtube_channel' => $recovered['youtube_channel'],
                    'duration' => $recovered['duration'],
                    'updated_at' => now(),
                ]);

            $updated++;
        }

        $this->info("Updated {$updated} tracks from runtime log metadata.");

        return 0;
    }

    /**
     * @return array<string, array{title:string, artist:?string, youtube_url:string, youtube_video_id:string, youtube_channel:?string, duration:float}>
     */
    private function buildRecoveryMap(string $logPath): array
    {
        $handle = fopen($logPath, 'r');
        if ($handle === false) {
            return [];
        }

        $map = [];

        while (($line = fgets($handle)) !== false) {
            if (strpos($line, 'insert into "tracks"') === false || strpos($line, 'audio/yt_') === false) {
                continue;
            }

            $recovered = $this->extractRecoveryEntry($line);
            if ($recovered === null) {
                continue;
            }

            $map[$recovered['file_path']] = $recovered;
        }

        fclose($handle);

        return $map;
    }

    /**
     * @return array{title:string, artist:?string, youtube_url:string, youtube_video_id:string, youtube_channel:?string, duration:float, file_path:string}|null
     */
    private function extractRecoveryEntry(string $line): ?array
    {
        $patterns = [
            '/values \((.*), ([0-9]+(?:\.[0-9]+)?), (audio\/yt_[^,]+\.mp3), youtube, (https:\/\/(?:www\.)?youtube\.com\/watch\?v=[^,\s]+), ([^,]+), (.*), pending, ([0-9\-: ]+), ([0-9\-: ]+)\) returning "id"/',
            '/values \((.*), ([0-9]+(?:\.[0-9]+)?), (audio\/yt_[^,]+\.mp3), youtube, (https:\/\/(?:www\.)?youtube\.com\/watch\?v=[^,\s]+), ([^,]+), pending, ([0-9\-: ]+), ([0-9\-: ]+)\) returning "id"/',
        ];

        foreach ($patterns as $pattern) {
            if (!preg_match($pattern, $line, $matches)) {
                continue;
            }

            $prefix = trim($matches[1]);
            $split = strrpos($prefix, ',');

            if ($split === false) {
                $title = $prefix;
                $artist = null;
            } else {
                $title = trim(substr($prefix, 0, $split));
                $artist = trim(substr($prefix, $split + 1));
            }

            $artist = $this->normalizeNullableText($artist);
            $youtubeChannel = null;

            if (count($matches) === 9) {
                $youtubeChannel = $this->normalizeNullableText(trim($matches[6]));
            }

            return [
                'title' => $title,
                'artist' => $artist,
                'youtube_url' => trim($matches[4]),
                'youtube_video_id' => trim($matches[5]),
                'youtube_channel' => $youtubeChannel,
                'duration' => (float) $matches[2],
                'file_path' => $matches[3],
            ];
        }

        return null;
    }

    private function normalizeNullableText(?string $value): ?string
    {
        if ($value === null) {
            return null;
        }

        $normalized = trim($value);
        if ($normalized === '' || $normalized === '?') {
            return null;
        }

        return $normalized;
    }
}
