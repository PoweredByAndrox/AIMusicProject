<?php

namespace App\Console\Commands;

use App\Models\Track;
use App\Services\TrackSidecarWriter;
use Illuminate\Console\Command;

class WriteTrackSidecars extends Command
{
    protected $signature = 'tracks:write-sidecars
                            {--dry-run : Show what would be written without creating files}
                            {--limit=0 : Limit number of audio files processed (0 = no limit)}
                            {--with-sha256 : Include file sha256 in the sidecar payload}
                            {--force : Skip confirmation}';

    protected $description = 'Write JSON sidecar metadata files next to every audio file for easier future recovery';

    public function handle(): int
    {
        $audioDir = storage_path('app/audio');
        if (!is_dir($audioDir)) {
            $this->error("Audio directory not found: {$audioDir}");
            return 1;
        }

        $dryRun = (bool) $this->option('dry-run');
        $limit = max(0, (int) $this->option('limit'));
        $withSha256 = (bool) $this->option('with-sha256');
        $sidecarWriter = app(TrackSidecarWriter::class);

        $files = glob($audioDir . '/*.mp3');
        if ($files === false || $files === []) {
            $this->warn('No mp3 files found in audio directory.');
            return 0;
        }

        sort($files, SORT_STRING);
        if ($limit > 0) {
            $files = array_slice($files, 0, $limit);
        }

        $relativePaths = array_map(
            static fn (string $path): string => 'audio/' . basename($path),
            $files
        );

        /** @var array<string, Track> $tracksByPath */
        $tracksByPath = Track::query()
            ->whereIn('file_path', $relativePaths)
            ->get()
            ->keyBy('file_path')
            ->all();

        $this->info('Audio files to process: ' . count($files));
        $this->info('Matching track rows found: ' . count($tracksByPath));

        if (
            !$dryRun &&
            !$this->option('force') &&
            !$this->confirm('Write sidecar files for the selected audio files?')
        ) {
            $this->info('Cancelled.');
            return 0;
        }

        $bar = $this->output->createProgressBar(count($files));
        $bar->start();

        $written = 0;
        $orphans = 0;
        $errors = 0;
        $generatedAt = now()->toAtomString();

        foreach ($files as $filePath) {
            $relativePath = 'audio/' . basename($filePath);
            $track = $tracksByPath[$relativePath] ?? null;

            if ($track === null) {
                $orphans++;
            }

            try {
                if (!$dryRun) {
                    $sidecarWriter->writeForFile($filePath, $track, $withSha256, $generatedAt);
                }

                $written++;
            } catch (\Throwable $e) {
                $errors++;
                $this->newLine();
                $this->error("Failed to write sidecar for {$relativePath}: {$e->getMessage()}");
            }

            $bar->advance();
        }

        $bar->finish();
        $this->newLine();

        if ($dryRun) {
            $this->info("Dry run complete. Sidecars to write: {$written}. Orphans: {$orphans}. Errors: {$errors}.");
        } else {
            $this->info("Sidecar write complete. Written: {$written}. Orphans: {$orphans}. Errors: {$errors}.");
        }

        return $errors > 0 ? 1 : 0;
    }
}
