<?php

namespace App\Services;

use App\Models\QueueTask;
use App\Models\Track;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use RuntimeException;

class StemSourceBuildService
{
    private const DEFAULT_MODEL_NAME = 'hybrid_htdemucs_ft_6s';

    public function canBuildTrack(Track $track): bool
    {
        $relative = trim((string) $track->file_path);
        if ($relative === '') {
            return false;
        }

        return Storage::disk('local')->exists($relative);
    }

    public function enqueue(Track $track): QueueTask
    {
        $audioRelativePath = trim((string) $track->file_path);
        if ($audioRelativePath === '' || !Storage::disk('local')->exists($audioRelativePath)) {
            throw new RuntimeException('This track has no source audio file in local storage, so stems cannot be built yet.');
        }

        $buildId = now()->format('Ymd_His') . '_' . Str::lower(Str::random(6));
        $outputRelativePath = 'stem_builds/track_' . $track->id . '/build_' . $buildId;
        $latestPointerRelativePath = 'stem_builds/track_' . $track->id . '/latest_build.json';
        $playbackManifestRelativePath = $audioRelativePath . '.playback.json';

        $queueTask = QueueTask::create([
            'task_type' => QueueTask::TYPE_STEM_SOURCE_BUILD,
            'payload' => [
                'track_id' => (int) $track->id,
                'build_id' => $buildId,
                'audio_path' => Storage::disk('local')->path($audioRelativePath),
                'output_dir' => Storage::disk('local')->path($outputRelativePath),
                'latest_pointer_path' => Storage::disk('local')->path($latestPointerRelativePath),
                'playback_manifest_path' => Storage::disk('local')->path($playbackManifestRelativePath),
                'model_name' => self::DEFAULT_MODEL_NAME,
            ],
            'status' => QueueTask::STATUS_PENDING,
        ]);

        MLQueueService::pushStemSourceBuild(
            $queueTask->id,
            (int) $track->id,
            $buildId,
            $this->workerStoragePath($audioRelativePath),
            $this->workerStoragePath($outputRelativePath),
            $this->workerStoragePath($playbackManifestRelativePath),
            $this->workerStoragePath($latestPointerRelativePath),
            self::DEFAULT_MODEL_NAME,
        );

        return $queueTask;
    }

    private function workerStoragePath(string $relativePath): string
    {
        $normalized = ltrim(str_replace('\\', '/', $relativePath), '/');

        return '/app/storage/' . $normalized;
    }
}
