<?php

namespace App\Console\Commands;

use App\Jobs\TrackAnalysisJob;
use App\Models\Track;
use App\Services\ArtistCreditParser;
use App\Services\MetadataExtractor;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Storage;

class ImportOrphanFiles extends Command
{
    protected $signature = 'tracks:import-orphan-files 
                            {--dry-run : Show what would be imported without actually importing}
                            {--limit=0 : Limit number of files to import (0 = no limit)}
                            {--skip-analysis : Restore track rows without dispatching TrackAnalysisJob}
                            {--force : Skip confirmation}';
    
    protected $description = 'Import audio files that exist on disk but have no database record';

    public function handle(): int
    {
        $dryRun = $this->option('dry-run');
        $limit = (int) $this->option('limit');
        $skipAnalysis = (bool) $this->option('skip-analysis');

        $this->info('Scanning for orphan audio files...');

        // Get all mp3 files from storage
        $audioDir = storage_path('app/audio');
        if (!is_dir($audioDir)) {
            $this->error("Audio directory not found: {$audioDir}");
            return 1;
        }

        $diskFiles = glob($audioDir . '/*.mp3');
        $this->info("Found " . count($diskFiles) . " mp3 files on disk");

        // Get all file paths from database
        $dbFilePaths = Track::pluck('file_path')->map(function ($path) {
            return basename($path);
        })->toArray();

        $this->info("Found " . count($dbFilePaths) . " tracks in database");

        // Find orphan files (on disk but not in DB)
        $orphanFiles = [];
        foreach ($diskFiles as $filePath) {
            $filename = basename($filePath);
            if (!in_array($filename, $dbFilePaths)) {
                $orphanFiles[] = $filePath;
            }
        }

        if (empty($orphanFiles)) {
            $this->info('No orphan files found. All files have database records.');
            return 0;
        }

        $this->info("Found " . count($orphanFiles) . " orphan files to import");

        if ($limit > 0 && count($orphanFiles) > $limit) {
            $orphanFiles = array_slice($orphanFiles, 0, $limit);
            $this->info("Limited to {$limit} files");
        }

        if ($dryRun) {
            $this->info('Dry run - showing first 10 files:');
            foreach (array_slice($orphanFiles, 0, 10) as $file) {
                $this->line("  - " . basename($file));
            }
            return 0;
        }

        if (!$this->option('force') && !$this->confirm("Import " . count($orphanFiles) . " orphan files?")) {
            $this->info('Cancelled.');
            return 0;
        }

        $bar = $this->output->createProgressBar(count($orphanFiles));
        $bar->start();

        $imported = 0;
        $errors = 0;
        $artistCreditParser = app(ArtistCreditParser::class);
        $metadataExtractor = app(MetadataExtractor::class);

        foreach ($orphanFiles as $filePath) {
            try {
                $filename = basename($filePath);
                $relativePath = 'audio/' . $filename;

                // Get file duration using ffprobe if available
                $duration = $this->getAudioDuration($filePath);
                $fileMetadata = $metadataExtractor->extract($filePath);
                $parsedMetadata = $artistCreditParser->parseUploadMetadata(
                    null,
                    $this->extractTitleFromFilename($filename),
                    $filename,
                );

                // Create track record with minimal data
                $attributes = [
                    'title' => $fileMetadata['title'] ?: $parsedMetadata['title'],
                    'artist' => $fileMetadata['artist'] ?: $parsedMetadata['artist'],
                    'artist_credits' => $parsedMetadata['artist_credits'],
                    'artist_slugs' => $parsedMetadata['artist_slugs'],
                    'album' => $fileMetadata['album'],
                    'year' => $fileMetadata['year'],
                    'duration' => $fileMetadata['duration'] > 0 ? $fileMetadata['duration'] : $duration,
                    'file_path' => $relativePath,
                    'source' => 'import',
                    'analysis_status' => 'pending',
                ];

                $track = $skipAnalysis
                    ? Track::withoutEvents(fn () => Track::create($attributes))
                    : Track::create($attributes);

                if (!$skipAnalysis) {
                    // Queue for ML analysis (will extract metadata and features later).
                    TrackAnalysisJob::dispatch($track);
                }

                $imported++;
            } catch (\Exception $e) {
                $errors++;
                $this->newLine();
                $this->error("Error importing {$filename}: " . $e->getMessage());
            }

            $bar->advance();
        }

        $bar->finish();
        $this->newLine();
        $this->info("Done! Imported {$imported} files, {$errors} errors.");
        
        return $errors > 0 ? 1 : 0;
    }

    private function getAudioDuration(string $filePath): float
    {
        // Try ffprobe first
        $result = @shell_exec("ffprobe -v quiet -show_entries format=duration -of csv=p=0 " . escapeshellarg($filePath) . " 2>/dev/null");
        if ($result && is_numeric(trim($result))) {
            return (float) trim($result);
        }

        // Fallback: estimate from file size (rough: ~1MB per minute for 128kbps mp3)
        $fileSize = filesize($filePath);
        return ($fileSize / 1024 / 1024) * 60; // Very rough estimate
    }

    private function extractTitleFromFilename(string $filename): string
    {
        // Remove extension and yt_ prefix
        $name = pathinfo($filename, PATHINFO_FILENAME);
        if (str_starts_with($name, 'yt_')) {
            return 'YouTube Import ' . substr($name, 3);
        }
        return $name;
    }
}
