<?php

namespace App\Services;

use App\Models\QueueTask;
use App\Models\Track;
use Illuminate\Support\Str;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Redis;
use RuntimeException;

class StemTuningRenderService
{
    public function __construct(
        private readonly TrackPlaybackManifestService $playbackManifestService,
    ) {}

    /**
     * @param array<string, mixed> $preset
     */
    public function enqueue(Track $track, array $preset): QueueTask
    {
        $config = config('stem_tuning_lab.tracks.' . $track->id);
        [$sourceManifest, $rawReferencePath] = $this->resolveRenderInputs($track, is_array($config) ? $config : null);
        $sourceAudioPath = $this->resolveTrackAudioPath($track);
        $defaultWorkerPreset = $this->defaultWorkerPreset(
            $sourceManifest,
            is_array($config['default_preset'] ?? null) ? $config['default_preset'] : null,
        );

        $renderId = now()->format('Ymd_His') . '_' . Str::lower(Str::random(6));
        $outputDirRelative = 'lab_artifacts/tuning_lab/track_' . $track->id . '/render_' . $renderId;
        $latestPointerRelative = 'lab_artifacts/tuning_lab/track_' . $track->id . '/latest_render.json';

        $normalizedPreset = $this->normalizePresetForWorker($preset, $defaultWorkerPreset);

        $queueTask = QueueTask::create([
            'task_type' => QueueTask::TYPE_STEM_TUNING_RENDER,
            'payload' => [
                'track_id' => (int) $track->id,
                'render_id' => $renderId,
                'source_stem_manifest_path' => $sourceManifest,
                'source_audio_path' => $sourceAudioPath,
                'output_dir' => base_path($outputDirRelative),
                'latest_pointer_path' => base_path($latestPointerRelative),
                'frontend_preset' => $preset,
                'default_worker_preset' => $defaultWorkerPreset,
                'preset' => $normalizedPreset,
                'raw_reference_path' => $rawReferencePath,
            ],
            'status' => QueueTask::STATUS_PENDING,
        ]);

        MLQueueService::pushStemTuningRender(
            $queueTask->id,
            (int) $track->id,
            $renderId,
            $this->toWorkerPath($sourceManifest),
            $this->relativeToWorkerPath($outputDirRelative),
            $this->relativeToWorkerPath($latestPointerRelative),
            $normalizedPreset,
            $rawReferencePath !== null ? $this->toWorkerPath($rawReferencePath) : null,
            $sourceAudioPath !== null ? $this->absolutePathToWorkerPath($sourceAudioPath) : null,
        );

        return $queueTask;
    }

    /**
     * Cancel or supersede a stem tuning render task for the given track.
     *
     * @return array{task:QueueTask,queue_cleanup:array{removed:int,remaining:int}|null}
     */
    public function cancel(Track $track, int $taskId, string $reason = 'Superseded by newer tuning preset changes'): array
    {
        $task = QueueTask::query()
            ->whereKey($taskId)
            ->where('task_type', QueueTask::TYPE_STEM_TUNING_RENDER)
            ->first();

        if ($task === null || (int) ($task->payload['track_id'] ?? 0) !== (int) $track->id) {
            throw new RuntimeException('Stem tuning render task not found for this track.');
        }

        if (in_array($task->status, [QueueTask::STATUS_COMPLETED, QueueTask::STATUS_FAILED, QueueTask::STATUS_CANCELLED], true)) {
            return [
                'task' => $task,
                'queue_cleanup' => null,
            ];
        }

        $message = trim($reason) !== '' ? trim($reason) : 'Stem tuning render cancelled';
        if ($task->status === QueueTask::STATUS_PENDING) {
            $queueCleanup = MLQueueService::removeQueuedTaskById($task->id);
            $task->markAsCancelled($message);
            MLQueueService::clearStemTuningRenderCancellation($task->id);
            $this->writeStatusPayload(
                $task->id,
                QueueTask::STATUS_CANCELLED,
                100,
                $message,
                'cancelled',
                [
                    'render_id' => (string) ($task->payload['render_id'] ?? ''),
                ],
            );

            return [
                'task' => $task->fresh(),
                'queue_cleanup' => $queueCleanup,
            ];
        }

        MLQueueService::requestStemTuningRenderCancellation($task->id, $message);
        $this->writeStatusPayload(
            $task->id,
            QueueTask::STATUS_PROCESSING,
            0,
            'Cancellation requested. The worker will stop this render and restart from the newest preset.',
            'cancel_requested',
            [
                'render_id' => (string) ($task->payload['render_id'] ?? ''),
                'cancel_requested' => true,
                'cancel_reason' => $message,
            ],
        );

        return [
            'task' => $task->fresh(),
            'queue_cleanup' => null,
        ];
    }

    /**
     * @param array<string, mixed>|null $config
     * @return array{0:string,1:?string}
     */
    private function resolveRenderInputs(Track $track, ?array $config): array
    {
        if (is_array($config)) {
            $sourceManifest = $this->resolveConfiguredPath($config['source_stem_manifest'] ?? null);
            if ($sourceManifest !== null && is_file($sourceManifest)) {
                $rawReferencePath = $this->resolveConfiguredPath($config['raw_reference_path'] ?? null);
                return [$sourceManifest, $rawReferencePath];
            }
        }

        $generatedManifest = $this->buildGenericSourceStemManifest($track);
        if ($generatedManifest === null) {
            throw new RuntimeException('No stem sources are available for this track yet. Build stems first, then open the tuning lab.');
        }

        return [$generatedManifest, $this->resolveTrackAudioPath($track)];
    }

    private function buildGenericSourceStemManifest(Track $track): ?string
    {
        $publicManifest = $this->playbackManifestService->manifestForTrack($track);
        $publicStems = is_array($publicManifest['stems'] ?? null) ? $publicManifest['stems'] : [];
        if ($publicStems === []) {
            return null;
        }

        $stems = [];
        foreach ($publicStems as $stemPayload) {
            if (!is_array($stemPayload)) {
                continue;
            }

            $stemKey = (string) ($stemPayload['key'] ?? '');
            if ($stemKey === '') {
                continue;
            }

            $resolvedStem = $this->playbackManifestService->resolveStemSource($track, $stemKey);
            if (!is_array($resolvedStem) || !is_string($resolvedStem['path'] ?? null) || !is_file($resolvedStem['path'])) {
                continue;
            }

            $stems[$stemKey] = [
                'path' => $this->absolutePathToWorkerPath($resolvedStem['path']),
                'label' => (string) ($resolvedStem['label'] ?? $stemKey),
                'description' => (string) ($resolvedStem['description'] ?? 'Auto-discovered stem from the playback manifest.'),
                'family' => (string) ($resolvedStem['family'] ?? 'playback_manifest'),
                'kind' => (string) ($resolvedStem['kind'] ?? 'stem'),
                'default_enabled' => (bool) ($resolvedStem['default_enabled'] ?? true),
            ];
        }

        if ($stems === []) {
            return null;
        }

        $manifestRelative = 'storage/app/tuning_lab/generated_source_manifests/track_' . $track->id . '.json';
        $manifestAbsolute = base_path($manifestRelative);
        $manifestDir = dirname($manifestAbsolute);
        if (!is_dir($manifestDir)) {
            mkdir($manifestDir, 0777, true);
        }

        $payload = [
            'track_id' => (int) $track->id,
            'audio_path' => ($audioPath = $this->resolveTrackAudioPath($track)) !== null
                ? $this->absolutePathToWorkerPath($audioPath)
                : null,
            'model_name' => 'generic_playback_manifest',
            'sample_rate' => 44100,
            'stems' => $stems,
        ];

        file_put_contents($manifestAbsolute, json_encode($payload, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE));

        $realPath = realpath($manifestAbsolute);
        return $realPath !== false ? $realPath : $manifestAbsolute;
    }

    private function resolveTrackAudioPath(Track $track): ?string
    {
        $relative = trim((string) $track->file_path);
        if ($relative === '' || !Storage::disk('local')->exists($relative)) {
            return null;
        }

        $absolute = Storage::disk('local')->path($relative);
        $realPath = realpath($absolute);
        return $realPath !== false ? $realPath : $absolute;
    }

    private function resolveConfiguredPath(mixed $candidate): ?string
    {
        if (!is_string($candidate) || trim($candidate) === '') {
            return null;
        }

        if (is_file($candidate)) {
            $realPath = realpath($candidate);
            return $realPath !== false ? $realPath : $candidate;
        }

        $normalized = str_replace(['/', '\\'], DIRECTORY_SEPARATOR, $candidate);
        $resolved = base_path($normalized);
        if (!is_file($resolved) && !is_dir($resolved)) {
            return $resolved;
        }

        $realPath = realpath($resolved);
        return $realPath !== false ? $realPath : $resolved;
    }

    private function relativeToWorkerPath(string $relativePath): string
    {
        $normalized = ltrim(str_replace('\\', '/', $relativePath), '/');

        if (str_starts_with($normalized, 'lab_artifacts/')) {
            return '/app/' . $normalized;
        }

        if (str_starts_with($normalized, 'storage/app/')) {
            return '/app/' . str_replace('storage/app/', 'storage/', $normalized);
        }

        return '/app/' . $normalized;
    }

    private function toWorkerPath(string $absolutePath): string
    {
        $normalized = str_replace('\\', '/', $absolutePath);
        $baseNormalized = str_replace('\\', '/', base_path());

        if (str_starts_with($normalized, '/app/')) {
            return $normalized;
        }

        if (!str_starts_with($normalized, $baseNormalized . '/')) {
            throw new RuntimeException('Configured tuning render path is outside the project mount: ' . $absolutePath);
        }

        $relative = ltrim(substr($normalized, strlen($baseNormalized)), '/');
        return $this->relativeToWorkerPath($relative);
    }

    private function absolutePathToWorkerPath(string $absolutePath): string
    {
        $normalized = str_replace('\\', '/', $absolutePath);
        $storageNormalized = rtrim(str_replace('\\', '/', Storage::disk('local')->path('')), '/');

        if (str_starts_with($normalized, $storageNormalized . '/')) {
            $relative = ltrim(substr($normalized, strlen($storageNormalized)), '/');
            return '/app/storage/' . $relative;
        }

        return $this->toWorkerPath($absolutePath);
    }

    /**
     * @param array<string, mixed> $extra
     */
    private function writeStatusPayload(
        int $taskId,
        string $status,
        int $progress,
        string $message,
        ?string $stage = null,
        array $extra = [],
    ): void {
        $statusKey = 'stem_tuning_render_status:' . $taskId;
        $existingRaw = Redis::connection()->get($statusKey);
        $existing = is_string($existingRaw) ? json_decode($existingRaw, true) : null;
        $logs = is_array($existing['logs'] ?? null) ? $existing['logs'] : [];

        $logs[] = [
            'timestamp' => now()->toIso8601String(),
            'level' => $status === QueueTask::STATUS_FAILED ? 'error' : 'info',
            'progress' => max(0, min(100, $progress)),
            'message' => $message,
            'stage' => $stage,
        ];

        Redis::connection()->setex($statusKey, 3600, json_encode([
            'status' => $status,
            'progress' => max(0, min(100, $progress)),
            'message' => $message,
            'task_id' => $taskId,
            'updated_at' => now()->toIso8601String(),
            'logs' => array_slice($logs, -120),
            'stage' => $stage,
            ...$extra,
        ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES));
    }

    /**
     * @return array<string, mixed>
     */
    private function defaultWorkerPreset(string $sourceManifestPath, ?array $configuredDefaultPreset = null): array
    {
        $decoded = json_decode((string) file_get_contents($sourceManifestPath), true);
        $stems = is_array($decoded['stems'] ?? null) ? $decoded['stems'] : [];
        if ($stems === []) {
            throw new RuntimeException('Source stem manifest has no editable base stems.');
        }

        $allowedStemKeys = array_values(array_filter(
            array_keys($stems),
            static fn (mixed $stemKey): bool => is_string($stemKey) && trim($stemKey) !== '',
        ));
        if ($allowedStemKeys === []) {
            throw new RuntimeException('Source stem manifest has no editable base stems.');
        }

        $configuredSourceKey = $this->normalizeLayerKey($configuredDefaultPreset['sourceKey'] ?? null);
        $manifestDefaultSourceKey = $this->manifestDefaultSourceKey($stems, $allowedStemKeys[0]);
        $defaultSourceKey = in_array($configuredSourceKey, $allowedStemKeys, true)
            ? $configuredSourceKey
            : $manifestDefaultSourceKey;

        $configuredStates = is_array($configuredDefaultPreset['layerStates'] ?? null)
            ? $configuredDefaultPreset['layerStates']
            : [];

        $layerStates = [];
        foreach ($allowedStemKeys as $stemKey) {
            $configuredState = null;
            foreach ($configuredStates as $key => $state) {
                if (!is_string($key) || !is_array($state)) {
                    continue;
                }
                if ($this->normalizeLayerKey($key) === $stemKey) {
                    $configuredState = $state;
                    break;
                }
            }

            $manifestState = is_array($stems[$stemKey] ?? null) ? $stems[$stemKey] : [];
            $layerStates[$stemKey] = [
                'enabled' => (bool) ($configuredState['enabled'] ?? $manifestState['default_enabled'] ?? true),
                'gainDb' => max(-48.0, min(12.0, (float) ($configuredState['gainDb'] ?? 0.0))),
                'transientDb' => max(-18.0, min(18.0, (float) ($configuredState['transientDb'] ?? 0.0))),
                'bodyDb' => max(-18.0, min(18.0, (float) ($configuredState['bodyDb'] ?? 0.0))),
                'harmonicDuckDb' => max(-18.0, min(0.0, (float) ($configuredState['harmonicDuckDb'] ?? 0.0))),
                'bleedDuckDb' => max(-18.0, min(0.0, (float) ($configuredState['bleedDuckDb'] ?? 0.0))),
            ];
        }

        return [
            'version' => 1,
            'sourceKey' => $defaultSourceKey,
            'renderMode' => $this->normalizeRenderMode($configuredDefaultPreset['renderMode'] ?? null, 'post_dsp'),
            'layerStates' => $layerStates,
            'hybridBuildConfig' => $this->defaultHybridBuildConfig(
                is_array($configuredDefaultPreset['hybridBuildConfig'] ?? null)
                    ? $configuredDefaultPreset['hybridBuildConfig']
                    : null,
            ),
            'stemRewriteConfig' => $this->defaultStemRewriteConfig(
                is_array($configuredDefaultPreset['stemRewriteConfig'] ?? null)
                    ? $configuredDefaultPreset['stemRewriteConfig']
                    : null,
            ),
            'vocalsRewriteConfig' => $this->defaultVocalsRewriteConfig(
                is_array($configuredDefaultPreset['vocalsRewriteConfig'] ?? null)
                    ? $configuredDefaultPreset['vocalsRewriteConfig']
                    : null,
            ),
            'unsafeModelInternals' => $this->defaultUnsafeModelInternals(
                is_array($configuredDefaultPreset['unsafeModelInternals'] ?? null)
                    ? $configuredDefaultPreset['unsafeModelInternals']
                    : null,
            ),
            'processorConfig' => $this->defaultProcessorConfig(
                is_array($configuredDefaultPreset['processorConfig'] ?? null)
                    ? $configuredDefaultPreset['processorConfig']
                    : null,
            ),
            'focusWindow' => [
                'enabled' => false,
                'startSec' => 0.0,
                'endSec' => 0.0,
                'xfadeSec' => 0.0,
                'targetLayerKey' => $defaultSourceKey,
                'layerGainDb' => 0.0,
                'otherLayersDuckDb' => 0.0,
            ],
        ];
    }

    /**
     * @param array<string, mixed> $preset
     * @param array<string, mixed> $defaultPreset
     * @return array<string, mixed>
     */
    private function normalizePresetForWorker(array $preset, array $defaultPreset): array
    {
        $layerDefaults = is_array($defaultPreset['layerStates'] ?? null) ? $defaultPreset['layerStates'] : [];
        $allowedStemKeys = array_values(array_filter(
            array_keys($layerDefaults),
            static fn (mixed $stemKey): bool => is_string($stemKey) && trim($stemKey) !== '',
        ));
        if ($allowedStemKeys === []) {
            throw new RuntimeException('Source stem manifest has no editable base stems.');
        }

        $defaultStemKey = (string) ($defaultPreset['sourceKey'] ?? $allowedStemKeys[0]);
        if (!in_array($defaultStemKey, $allowedStemKeys, true)) {
            $defaultStemKey = $allowedStemKeys[0];
        }

        $normalized = $defaultPreset;

        $normalizedSourceKey = $this->normalizeLayerKey($preset['sourceKey'] ?? null);
        $normalized['sourceKey'] = in_array($normalizedSourceKey, $allowedStemKeys, true)
            ? $normalizedSourceKey
            : $defaultStemKey;

        $normalized['renderMode'] = $this->normalizeRenderMode(
            $preset['renderMode'] ?? null,
            (string) ($defaultPreset['renderMode'] ?? 'post_dsp'),
        );

        $normalized['hybridBuildConfig'] = $this->normalizeHybridBuildConfig(
            is_array($preset['hybridBuildConfig'] ?? null) ? $preset['hybridBuildConfig'] : [],
            is_array($defaultPreset['hybridBuildConfig'] ?? null) ? $defaultPreset['hybridBuildConfig'] : $this->defaultHybridBuildConfig(),
        );

        $normalized['stemRewriteConfig'] = $this->normalizeStemRewriteConfig(
            is_array($preset['stemRewriteConfig'] ?? null) ? $preset['stemRewriteConfig'] : [],
            is_array($defaultPreset['stemRewriteConfig'] ?? null) ? $defaultPreset['stemRewriteConfig'] : $this->defaultStemRewriteConfig(),
        );

        $normalized['vocalsRewriteConfig'] = $this->normalizeVocalsRewriteConfig(
            is_array($preset['vocalsRewriteConfig'] ?? null) ? $preset['vocalsRewriteConfig'] : [],
            is_array($defaultPreset['vocalsRewriteConfig'] ?? null) ? $defaultPreset['vocalsRewriteConfig'] : $this->defaultVocalsRewriteConfig(),
        );

        $normalized['unsafeModelInternals'] = $this->normalizeUnsafeModelInternals(
            is_array($preset['unsafeModelInternals'] ?? null) ? $preset['unsafeModelInternals'] : [],
            is_array($defaultPreset['unsafeModelInternals'] ?? null) ? $defaultPreset['unsafeModelInternals'] : $this->defaultUnsafeModelInternals(),
        );

        $normalized['processorConfig'] = $this->normalizeProcessorConfig(
            is_array($preset['processorConfig'] ?? null) ? $preset['processorConfig'] : [],
            is_array($defaultPreset['processorConfig'] ?? null) ? $defaultPreset['processorConfig'] : $this->defaultProcessorConfig(),
        );

        $focusWindow = is_array($preset['focusWindow'] ?? null) ? $preset['focusWindow'] : [];
        $targetLayerKey = $this->normalizeLayerKey($focusWindow['targetLayerKey'] ?? null);
        $normalized['focusWindow'] = [
            'enabled' => (bool) ($focusWindow['enabled'] ?? ($defaultPreset['focusWindow']['enabled'] ?? false)),
            'startSec' => max(0.0, (float) ($focusWindow['startSec'] ?? 0.0)),
            'endSec' => max(0.0, (float) ($focusWindow['endSec'] ?? 0.0)),
            'xfadeSec' => max(0.0, min(5.0, (float) ($focusWindow['xfadeSec'] ?? ($defaultPreset['focusWindow']['xfadeSec'] ?? 0.0)))),
            'targetLayerKey' => in_array($targetLayerKey, $allowedStemKeys, true)
            ? $targetLayerKey
            : $normalized['sourceKey'],
            'layerGainDb' => max(-24.0, min(24.0, (float) ($focusWindow['layerGainDb'] ?? ($defaultPreset['focusWindow']['layerGainDb'] ?? 0.0)))),
            'otherLayersDuckDb' => max(-24.0, min(0.0, (float) ($focusWindow['otherLayersDuckDb'] ?? ($defaultPreset['focusWindow']['otherLayersDuckDb'] ?? 0.0)))),
        ];
        if ($normalized['focusWindow']['endSec'] < $normalized['focusWindow']['startSec']) {
            $normalized['focusWindow']['endSec'] = $normalized['focusWindow']['startSec'];
        }

        $layerStates = is_array($preset['layerStates'] ?? null) ? $preset['layerStates'] : [];
        $normalizedStates = [];
        foreach ($layerStates as $key => $state) {
            if (!is_string($key) || !is_array($state)) {
                continue;
            }
            $normalizedKey = $this->normalizeLayerKey($key);
            if (!in_array($normalizedKey, $allowedStemKeys, true)) {
                continue;
            }
            $defaultState = is_array($layerDefaults[$normalizedKey] ?? null) ? $layerDefaults[$normalizedKey] : [];
            $normalizedStates[$normalizedKey] = [
                'enabled' => (bool) ($state['enabled'] ?? $defaultState['enabled'] ?? false),
                'gainDb' => max(-48.0, min(12.0, (float) ($state['gainDb'] ?? $defaultState['gainDb'] ?? 0.0))),
                'transientDb' => max(-18.0, min(18.0, (float) ($state['transientDb'] ?? $defaultState['transientDb'] ?? 0.0))),
                'bodyDb' => max(-18.0, min(18.0, (float) ($state['bodyDb'] ?? $defaultState['bodyDb'] ?? 0.0))),
                'harmonicDuckDb' => max(-18.0, min(0.0, (float) ($state['harmonicDuckDb'] ?? $defaultState['harmonicDuckDb'] ?? 0.0))),
                'bleedDuckDb' => max(-18.0, min(0.0, (float) ($state['bleedDuckDb'] ?? $defaultState['bleedDuckDb'] ?? 0.0))),
            ];
        }

        foreach ($allowedStemKeys as $stemKey) {
            if (array_key_exists($stemKey, $normalizedStates)) {
                continue;
            }

            $normalizedStates[$stemKey] = $layerDefaults[$stemKey];
        }

        $normalized['layerStates'] = $normalizedStates;

        return $normalized;
    }

    private function normalizeRenderMode(mixed $candidate, string $fallback): string
    {
        return in_array($candidate, ['post_dsp', 'hybrid_build', 'stem_rewrite', 'vocals_rewrite'], true)
            ? (string) $candidate
            : $fallback;
    }

    /**
     * @param array<string, mixed>|null $configured
     * @return array<string, mixed>
     */
    private function defaultHybridBuildConfig(?array $configured = null): array
    {
        return $this->normalizeHybridBuildConfig($configured ?? [], [
            'modelName' => 'hybrid_htdemucs_ft_6s',
            'sharpenBleed' => true,
            'includeMelodyStem' => false,
            'aggressiveMelodyFirst' => false,
        ]);
    }

    /**
     * @param array<string, mixed> $candidate
     * @param array<string, mixed> $fallback
     * @return array<string, mixed>
     */
    private function normalizeHybridBuildConfig(array $candidate, array $fallback): array
    {
        $modelName = (string) ($candidate['modelName'] ?? $fallback['modelName'] ?? 'hybrid_htdemucs_ft_6s');
        if (!in_array($modelName, ['hybrid_htdemucs_ft_6s', 'hybrid_htdemucs_ft_bs_roformer_6s'], true)) {
            $modelName = (string) ($fallback['modelName'] ?? 'hybrid_htdemucs_ft_6s');
        }

        return [
            'modelName' => $modelName,
            'sharpenBleed' => (bool) ($candidate['sharpenBleed'] ?? $fallback['sharpenBleed'] ?? true),
            'includeMelodyStem' => (bool) ($candidate['includeMelodyStem'] ?? $fallback['includeMelodyStem'] ?? false),
            'aggressiveMelodyFirst' => (bool) ($candidate['aggressiveMelodyFirst'] ?? $fallback['aggressiveMelodyFirst'] ?? false),
        ];
    }

    /**
     * @param array<string, mixed>|null $configured
     * @return array<string, mixed>
     */
    private function defaultStemRewriteConfig(?array $configured = null): array
    {
        return $this->normalizeStemRewriteConfig($configured ?? [], [
            'otherMove' => 0.45,
            'guitarMove' => 1.0,
            'pianoMove' => 1.0,
            'drumsMaskedMove' => 0.12,
            'targetMelodyRms' => 0.026,
            'maxMelodyGain' => 1.45,
            'stageDrumHarmonicMove' => 0.42,
            'stageHarmonicStemPercussiveMove' => 0.22,
            'stageCleanupDrumHarmonicMove' => 0.05,
            'stageCleanupHarmonicStemPercussiveMove' => 0.28,
            'harmonicCount' => 3,
            'confidenceFloor' => 0.3,
        ]);
    }

    /**
     * @param array<string, mixed>|null $configured
     * @return array<string, mixed>
     */
    private function defaultVocalsRewriteConfig(?array $configured = null): array
    {
        return $this->normalizeVocalsRewriteConfig($configured ?? [], [
            'taskMode' => 'custom',
            'vocalIsolationIntensity' => 0.0,
            'bleedRemovalIntensity' => 0.0,
            'airRestore' => 0.0,
            'bodyPreserve' => 0.0,
            'artifactSuppression' => 0.0,
            'centerVocalSuppression' => 0.0,
            'candidateMode' => 'hybrid_vocals',
            'lowBsWeight' => 0.75,
            'highBsWeight' => 0.25,
            'inactiveLowBsWeight' => 0.97,
            'inactiveHighBsWeight' => 0.9,
            'crossoverHz' => 3000.0,
            'activityAbsFloorRatio' => 0.08,
            'activityRelFloor' => 0.35,
            'activitySmoothingFrames' => 11,
            'vocalSharpenAlpha' => 0.0,
            'tailGateStrength' => 0.0,
            'tailGateFloorRatio' => 0.08,
            'tailGateSmoothingFrames' => 11,
            'subtractiveStrength' => 0.0,
            'subtractivePower' => 1.0,
            'subtractiveResidualFloor' => 0.12,
            'subtractiveActiveRestore' => 0.0,
            'subtractiveTargetRmsRatio' => 0.0,
            'subtractiveMaxGain' => 1.35,
            'hardMaskThreshold' => 0.0,
            'hardMaskPower' => 3.0,
            'hardMaskResidualFloor' => 0.02,
        ]);
    }

    /**
     * @param array<string, mixed>|null $configured
     * @return array<string, mixed>
     */
    private function defaultUnsafeModelInternals(?array $configured = null): array
    {
        return $this->normalizeUnsafeModelInternals($configured ?? [], [
            'hybrid' => [
                'requestedDevice' => 'auto',
                'stage1Model' => 'htdemucs_ft',
                'stage2Model' => 'htdemucs_6s',
                'stage1RequestedDevice' => 'auto',
                'stage2RequestedDevice' => 'auto',
                'stage1Split' => true,
                'stage1Progress' => true,
                'stage1Shifts' => 1,
                'stage1Overlap' => 0.25,
                'stage1Segment' => 0.0,
                'stage2Split' => true,
                'stage2Progress' => true,
                'stage2Shifts' => 1,
                'stage2Overlap' => 0.25,
                'stage2Segment' => 0.0,
            ],
            'bsRoformer' => [
                'requestedDevice' => 'auto',
                'deviceIds' => ['0'],
                'forceCpu' => false,
                'disableDetailedPbar' => false,
                'repoPath' => '/tmp/msst_zfturbo',
                'configPath' => '/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml',
                'checkpointPath' => '/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt',
                'useTta' => true,
                'extractInstrumental' => true,
                'filenameTemplate' => '{instr}',
                'configOverrideYaml' => '',
                'extraCliArgs' => [],
            ],
        ]);
    }

    /**
     * @param array<string, mixed> $candidate
     * @param array<string, mixed> $fallback
     * @return array<string, mixed>
     */
    private function normalizeStemRewriteConfig(array $candidate, array $fallback): array
    {
        return [
            'otherMove' => max(0.0, min(1.5, (float) ($candidate['otherMove'] ?? $fallback['otherMove'] ?? 0.45))),
            'guitarMove' => max(0.0, min(1.5, (float) ($candidate['guitarMove'] ?? $fallback['guitarMove'] ?? 1.0))),
            'pianoMove' => max(0.0, min(1.5, (float) ($candidate['pianoMove'] ?? $fallback['pianoMove'] ?? 1.0))),
            'drumsMaskedMove' => max(0.0, min(1.0, (float) ($candidate['drumsMaskedMove'] ?? $fallback['drumsMaskedMove'] ?? 0.12))),
            'targetMelodyRms' => max(0.001, min(0.1, (float) ($candidate['targetMelodyRms'] ?? $fallback['targetMelodyRms'] ?? 0.026))),
            'maxMelodyGain' => max(1.0, min(3.0, (float) ($candidate['maxMelodyGain'] ?? $fallback['maxMelodyGain'] ?? 1.45))),
            'stageDrumHarmonicMove' => max(0.0, min(1.0, (float) ($candidate['stageDrumHarmonicMove'] ?? $fallback['stageDrumHarmonicMove'] ?? 0.42))),
            'stageHarmonicStemPercussiveMove' => max(0.0, min(1.0, (float) ($candidate['stageHarmonicStemPercussiveMove'] ?? $fallback['stageHarmonicStemPercussiveMove'] ?? 0.22))),
            'stageCleanupDrumHarmonicMove' => max(0.0, min(1.0, (float) ($candidate['stageCleanupDrumHarmonicMove'] ?? $fallback['stageCleanupDrumHarmonicMove'] ?? 0.05))),
            'stageCleanupHarmonicStemPercussiveMove' => max(0.0, min(1.0, (float) ($candidate['stageCleanupHarmonicStemPercussiveMove'] ?? $fallback['stageCleanupHarmonicStemPercussiveMove'] ?? 0.28))),
            'harmonicCount' => max(1, min(8, (int) ($candidate['harmonicCount'] ?? $fallback['harmonicCount'] ?? 3))),
            'confidenceFloor' => max(0.0, min(1.0, (float) ($candidate['confidenceFloor'] ?? $fallback['confidenceFloor'] ?? 0.3))),
        ];
    }

    /**
     * @param array<string, mixed> $candidate
     * @param array<string, mixed> $fallback
     * @return array<string, mixed>
     */
    private function normalizeVocalsRewriteConfig(array $candidate, array $fallback): array
    {
        $taskMode = (string) ($candidate['taskMode'] ?? $fallback['taskMode'] ?? 'custom');
        if (!in_array($taskMode, ['custom', 'clean_vocals', 'minus_from_vocals', 'artifact_repair', 'aggressive_vocal_suppression', 'detail_preserve'], true)) {
            $taskMode = (string) ($fallback['taskMode'] ?? 'custom');
        }

        $candidateMode = (string) ($candidate['candidateMode'] ?? $fallback['candidateMode'] ?? 'hybrid_vocals');
        if (!in_array($candidateMode, ['ft_vocals', 'bs_vocals', 'hybrid_vocals'], true)) {
            $candidateMode = (string) ($fallback['candidateMode'] ?? 'hybrid_vocals');
        }

        return [
            'taskMode' => $taskMode,
            'vocalIsolationIntensity' => max(0.0, min(1.0, (float) ($candidate['vocalIsolationIntensity'] ?? $fallback['vocalIsolationIntensity'] ?? 0.0))),
            'bleedRemovalIntensity' => max(0.0, min(1.0, (float) ($candidate['bleedRemovalIntensity'] ?? $fallback['bleedRemovalIntensity'] ?? 0.0))),
            'airRestore' => max(0.0, min(1.0, (float) ($candidate['airRestore'] ?? $fallback['airRestore'] ?? 0.0))),
            'bodyPreserve' => max(0.0, min(1.0, (float) ($candidate['bodyPreserve'] ?? $fallback['bodyPreserve'] ?? 0.0))),
            'artifactSuppression' => max(0.0, min(1.0, (float) ($candidate['artifactSuppression'] ?? $fallback['artifactSuppression'] ?? 0.0))),
            'centerVocalSuppression' => max(0.0, min(1.0, (float) ($candidate['centerVocalSuppression'] ?? $fallback['centerVocalSuppression'] ?? 0.0))),
            'candidateMode' => $candidateMode,
            'lowBsWeight' => max(0.0, min(1.0, (float) ($candidate['lowBsWeight'] ?? $fallback['lowBsWeight'] ?? 0.75))),
            'highBsWeight' => max(0.0, min(1.0, (float) ($candidate['highBsWeight'] ?? $fallback['highBsWeight'] ?? 0.25))),
            'inactiveLowBsWeight' => max(0.0, min(1.0, (float) ($candidate['inactiveLowBsWeight'] ?? $fallback['inactiveLowBsWeight'] ?? 0.97))),
            'inactiveHighBsWeight' => max(0.0, min(1.0, (float) ($candidate['inactiveHighBsWeight'] ?? $fallback['inactiveHighBsWeight'] ?? 0.9))),
            'crossoverHz' => max(200.0, min(12000.0, (float) ($candidate['crossoverHz'] ?? $fallback['crossoverHz'] ?? 3000.0))),
            'activityAbsFloorRatio' => max(0.0, min(1.0, (float) ($candidate['activityAbsFloorRatio'] ?? $fallback['activityAbsFloorRatio'] ?? 0.08))),
            'activityRelFloor' => max(0.0, min(1.0, (float) ($candidate['activityRelFloor'] ?? $fallback['activityRelFloor'] ?? 0.35))),
            'activitySmoothingFrames' => max(1, min(101, (int) ($candidate['activitySmoothingFrames'] ?? $fallback['activitySmoothingFrames'] ?? 11))),
            'vocalSharpenAlpha' => max(0.0, min(4.0, (float) ($candidate['vocalSharpenAlpha'] ?? $fallback['vocalSharpenAlpha'] ?? 0.0))),
            'tailGateStrength' => max(0.0, min(1.0, (float) ($candidate['tailGateStrength'] ?? $fallback['tailGateStrength'] ?? 0.0))),
            'tailGateFloorRatio' => max(0.0, min(1.0, (float) ($candidate['tailGateFloorRatio'] ?? $fallback['tailGateFloorRatio'] ?? 0.08))),
            'tailGateSmoothingFrames' => max(1, min(101, (int) ($candidate['tailGateSmoothingFrames'] ?? $fallback['tailGateSmoothingFrames'] ?? 11))),
            'subtractiveStrength' => max(0.0, min(4.0, (float) ($candidate['subtractiveStrength'] ?? $fallback['subtractiveStrength'] ?? 0.0))),
            'subtractivePower' => max(0.1, min(8.0, (float) ($candidate['subtractivePower'] ?? $fallback['subtractivePower'] ?? 1.0))),
            'subtractiveResidualFloor' => max(0.0, min(1.0, (float) ($candidate['subtractiveResidualFloor'] ?? $fallback['subtractiveResidualFloor'] ?? 0.12))),
            'subtractiveActiveRestore' => max(0.0, min(1.0, (float) ($candidate['subtractiveActiveRestore'] ?? $fallback['subtractiveActiveRestore'] ?? 0.0))),
            'subtractiveTargetRmsRatio' => max(0.0, min(4.0, (float) ($candidate['subtractiveTargetRmsRatio'] ?? $fallback['subtractiveTargetRmsRatio'] ?? 0.0))),
            'subtractiveMaxGain' => max(1.0, min(4.0, (float) ($candidate['subtractiveMaxGain'] ?? $fallback['subtractiveMaxGain'] ?? 1.35))),
            'hardMaskThreshold' => max(0.0, min(1.0, (float) ($candidate['hardMaskThreshold'] ?? $fallback['hardMaskThreshold'] ?? 0.0))),
            'hardMaskPower' => max(1.0, min(12.0, (float) ($candidate['hardMaskPower'] ?? $fallback['hardMaskPower'] ?? 3.0))),
            'hardMaskResidualFloor' => max(0.0, min(1.0, (float) ($candidate['hardMaskResidualFloor'] ?? $fallback['hardMaskResidualFloor'] ?? 0.02))),
        ];
    }

    /**
     * @param array<string, mixed> $candidate
     * @param array<string, mixed> $fallback
     * @return array<string, mixed>
     */
    private function normalizeUnsafeModelInternals(array $candidate, array $fallback): array
    {
        $hybridCandidate = is_array($candidate['hybrid'] ?? null) ? $candidate['hybrid'] : [];
        $hybridFallback = is_array($fallback['hybrid'] ?? null) ? $fallback['hybrid'] : [];
        $bsCandidate = is_array($candidate['bsRoformer'] ?? null) ? $candidate['bsRoformer'] : [];
        $bsFallback = is_array($fallback['bsRoformer'] ?? null) ? $fallback['bsRoformer'] : [];

        return [
            'hybrid' => [
                'requestedDevice' => trim((string) ($hybridCandidate['requestedDevice'] ?? $hybridFallback['requestedDevice'] ?? 'auto')) ?: 'auto',
                'stage1Model' => trim((string) ($hybridCandidate['stage1Model'] ?? $hybridFallback['stage1Model'] ?? 'htdemucs_ft')) ?: 'htdemucs_ft',
                'stage2Model' => trim((string) ($hybridCandidate['stage2Model'] ?? $hybridFallback['stage2Model'] ?? 'htdemucs_6s')) ?: 'htdemucs_6s',
                'stage1RequestedDevice' => trim((string) ($hybridCandidate['stage1RequestedDevice'] ?? $hybridFallback['stage1RequestedDevice'] ?? 'auto')) ?: 'auto',
                'stage2RequestedDevice' => trim((string) ($hybridCandidate['stage2RequestedDevice'] ?? $hybridFallback['stage2RequestedDevice'] ?? 'auto')) ?: 'auto',
                'stage1Split' => (bool) ($hybridCandidate['stage1Split'] ?? $hybridFallback['stage1Split'] ?? true),
                'stage1Progress' => (bool) ($hybridCandidate['stage1Progress'] ?? $hybridFallback['stage1Progress'] ?? true),
                'stage1Shifts' => max(1, min(8, (int) ($hybridCandidate['stage1Shifts'] ?? $hybridFallback['stage1Shifts'] ?? 1))),
                'stage1Overlap' => max(0.0, min(0.99, (float) ($hybridCandidate['stage1Overlap'] ?? $hybridFallback['stage1Overlap'] ?? 0.25))),
                'stage1Segment' => max(0.0, min(600.0, (float) ($hybridCandidate['stage1Segment'] ?? $hybridFallback['stage1Segment'] ?? 0.0))),
                'stage2Split' => (bool) ($hybridCandidate['stage2Split'] ?? $hybridFallback['stage2Split'] ?? true),
                'stage2Progress' => (bool) ($hybridCandidate['stage2Progress'] ?? $hybridFallback['stage2Progress'] ?? true),
                'stage2Shifts' => max(1, min(8, (int) ($hybridCandidate['stage2Shifts'] ?? $hybridFallback['stage2Shifts'] ?? 1))),
                'stage2Overlap' => max(0.0, min(0.99, (float) ($hybridCandidate['stage2Overlap'] ?? $hybridFallback['stage2Overlap'] ?? 0.25))),
                'stage2Segment' => max(0.0, min(600.0, (float) ($hybridCandidate['stage2Segment'] ?? $hybridFallback['stage2Segment'] ?? 0.0))),
            ],
            'bsRoformer' => [
                'requestedDevice' => trim((string) ($bsCandidate['requestedDevice'] ?? $bsFallback['requestedDevice'] ?? 'auto')) ?: 'auto',
                'deviceIds' => array_values(array_filter(array_map(
                    static fn (mixed $value): string => trim((string) $value),
                    is_array($bsCandidate['deviceIds'] ?? null)
                        ? $bsCandidate['deviceIds']
                        : (is_array($bsFallback['deviceIds'] ?? null) ? $bsFallback['deviceIds'] : ['0'])
                ), static fn (string $value): bool => $value !== '')),
                'forceCpu' => (bool) ($bsCandidate['forceCpu'] ?? $bsFallback['forceCpu'] ?? false),
                'disableDetailedPbar' => (bool) ($bsCandidate['disableDetailedPbar'] ?? $bsFallback['disableDetailedPbar'] ?? false),
                'repoPath' => trim((string) ($bsCandidate['repoPath'] ?? $bsFallback['repoPath'] ?? '/tmp/msst_zfturbo')) ?: '/tmp/msst_zfturbo',
                'configPath' => trim((string) ($bsCandidate['configPath'] ?? $bsFallback['configPath'] ?? '/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml')) ?: '/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml',
                'checkpointPath' => trim((string) ($bsCandidate['checkpointPath'] ?? $bsFallback['checkpointPath'] ?? '/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt')) ?: '/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt',
                'useTta' => (bool) ($bsCandidate['useTta'] ?? $bsFallback['useTta'] ?? true),
                'extractInstrumental' => (bool) ($bsCandidate['extractInstrumental'] ?? $bsFallback['extractInstrumental'] ?? true),
                'filenameTemplate' => trim((string) ($bsCandidate['filenameTemplate'] ?? $bsFallback['filenameTemplate'] ?? '{instr}')) ?: '{instr}',
                'configOverrideYaml' => (string) ($bsCandidate['configOverrideYaml'] ?? $bsFallback['configOverrideYaml'] ?? ''),
                'extraCliArgs' => array_values(array_filter(array_map(
                    static fn (mixed $value): string => trim((string) $value),
                    is_array($bsCandidate['extraCliArgs'] ?? null)
                        ? $bsCandidate['extraCliArgs']
                        : (is_array($bsFallback['extraCliArgs'] ?? null) ? $bsFallback['extraCliArgs'] : [])
                ), static fn (string $value): bool => $value !== '')),
            ],
        ];
    }

    /**
     * @param array<string, mixed>|null $configured
     * @return array<string, mixed>
     */
    private function defaultProcessorConfig(?array $configured = null): array
    {
        return $this->normalizeProcessorConfig($configured ?? [], [
            'bypass' => false,
            'harmonicMargin' => 1.0,
            'percussiveMargin' => 4.0,
            'transientCutoffHz' => 2500.0,
            'bodyLowHz' => 90.0,
            'bodyHighHz' => 320.0,
            'bleedLowHz' => 700.0,
            'bleedHighHz' => 3500.0,
        ]);
    }

    /**
     * @param array<string, mixed> $candidate
     * @param array<string, mixed> $fallback
     * @return array<string, mixed>
     */
    private function normalizeProcessorConfig(array $candidate, array $fallback): array
    {
        $bodyLow = max(40.0, min(400.0, (float) ($candidate['bodyLowHz'] ?? $fallback['bodyLowHz'] ?? 90.0)));
        $bleedLow = max(120.0, min(2500.0, (float) ($candidate['bleedLowHz'] ?? $fallback['bleedLowHz'] ?? 700.0)));

        return [
            'bypass' => (bool) ($candidate['bypass'] ?? $fallback['bypass'] ?? false),
            'harmonicMargin' => max(0.5, min(4.0, (float) ($candidate['harmonicMargin'] ?? $fallback['harmonicMargin'] ?? 1.0))),
            'percussiveMargin' => max(0.5, min(6.0, (float) ($candidate['percussiveMargin'] ?? $fallback['percussiveMargin'] ?? 4.0))),
            'transientCutoffHz' => max(800.0, min(8000.0, (float) ($candidate['transientCutoffHz'] ?? $fallback['transientCutoffHz'] ?? 2500.0))),
            'bodyLowHz' => $bodyLow,
            'bodyHighHz' => max($bodyLow + 20.0, min(1200.0, (float) ($candidate['bodyHighHz'] ?? $fallback['bodyHighHz'] ?? 320.0))),
            'bleedLowHz' => $bleedLow,
            'bleedHighHz' => max($bleedLow + 100.0, min(9000.0, (float) ($candidate['bleedHighHz'] ?? $fallback['bleedHighHz'] ?? 3500.0))),
        ];
    }

    /**
     * @return array<int, string>
     */
    private function stemKeysFromSourceManifest(string $sourceManifestPath): array
    {
        $decoded = json_decode((string) file_get_contents($sourceManifestPath), true);
        if (!is_array($decoded)) {
            return [];
        }

        $stems = is_array($decoded['stems'] ?? null) ? $decoded['stems'] : [];
        $keys = [];
        foreach ($stems as $stemKey => $payload) {
            if (!is_string($stemKey) || !is_array($payload)) {
                continue;
            }
            $keys[] = $stemKey;
        }

        return array_values($keys);
    }

    /**
     * @param array<string, mixed> $stems
     */
    private function manifestDefaultSourceKey(array $stems, string $fallback): string
    {
        foreach ($stems as $stemKey => $payload) {
            if (!is_string($stemKey) || !is_array($payload)) {
                continue;
            }
            if ((bool) ($payload['default_enabled'] ?? false)) {
                return $stemKey;
            }
        }

        return $fallback;
    }

    private function normalizeLayerKey(mixed $key): string
    {
        if (!is_string($key) || trim($key) === '') {
            return '';
        }

        return preg_replace('/^(stem_|render_)/', '', $key) ?? $key;
    }
}
