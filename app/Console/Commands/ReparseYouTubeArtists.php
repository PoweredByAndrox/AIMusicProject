<?php

namespace App\Console\Commands;

use App\Models\Track;
use App\Services\ArtistCreditParser;
use Illuminate\Console\Command;

class ReparseYouTubeArtists extends Command
{
    protected $signature = 'tracks:reparse-artists {--dry-run : Show changes without applying}';
    protected $description = 'Rebuild normalized artist credits for existing tracks';

    public function handle(): int
    {
        $dryRun = (bool) $this->option('dry-run');
        $parser = app(ArtistCreditParser::class);

        $tracks = Track::query()->orderBy('id')->get();

        $this->info("Found {$tracks->count()} tracks to process");

        $updated = 0;
        $skipped = 0;

        foreach ($tracks as $track) {
            $parsed = $parser->resolveTrack($track);

            $changes = array_filter([
                'title' => $parsed['title'] !== (string) $track->getRawOriginal('title') ? $parsed['title'] : null,
                'artist' => $parsed['artist'] !== $track->getRawOriginal('artist') ? $parsed['artist'] : null,
                'artist_credits' => $parsed['artist_credits'] !== ($track->getRawOriginal('artist_credits') ? json_decode((string) $track->getRawOriginal('artist_credits'), true) : null)
                    ? $parsed['artist_credits']
                    : null,
                'artist_slugs' => $parsed['artist_slugs'] !== ($track->getRawOriginal('artist_slugs') ? json_decode((string) $track->getRawOriginal('artist_slugs'), true) : null)
                    ? $parsed['artist_slugs']
                    : null,
            ], static fn (mixed $value): bool => $value !== null);

            if ($changes === []) {
                $skipped++;
                continue;
            }

            if ($dryRun) {
                $this->line("Would update track #{$track->id}:");
                $this->line("  Title: '{$track->getRawOriginal('title')}' => '{$parsed['title']}'");
                $this->line("  Artist: '{$track->getRawOriginal('artist')}' => '" . ($parsed['artist'] ?? '') . "'");
                $this->line('  Credits: ' . json_encode($parsed['artist_credits'], JSON_UNESCAPED_UNICODE));
            } else {
                $track->forceFill([
                    'title' => $parsed['title'],
                    'artist' => $parsed['artist'],
                    'artist_credits' => $parsed['artist_credits'],
                    'artist_slugs' => $parsed['artist_slugs'],
                ])->save();

                $this->line("Updated track #{$track->id}: {$parsed['title']}");
            }

            $updated++;
        }

        $this->info("Done! Updated: {$updated}, Skipped: {$skipped}");

        return Command::SUCCESS;
    }
}
