<?php

namespace App\Services;

use App\Models\Track;

class StemTuningLabRegistry
{
    /**
     * @var array<int, array<string, mixed>|null>
     */
    private array $trackConfigCache = [];

    /**
     * @var array<int, array<string, array<string, mixed>|null>>
     */
    private array $resolvedSourceCache = [];

    public function __construct(
        private readonly TrackPlaybackManifestService $playbackManifestService,
        private readonly StemTuningSegmentDefinitionService $segmentDefinitions,
    ) {}

    /**
     * @return array<string, mixed>|null
     */
    public function sessionForTrack(Track $track): ?array
    {
        $trackConfig = $this->trackConfig($track);
        if (!is_array($trackConfig)) {
            return null;
        }

        $sources = $this->publicSourcesForTrack($track, $trackConfig);
        if ($sources === []) {
            return null;
        }

        $defaultSourceKey = (string) ($trackConfig['default_source_key'] ?? array_key_first($sources));
        if (!array_key_exists($defaultSourceKey, $sources)) {
            $defaultSourceKey = (string) array_key_first($sources);
        }

        return [
            'title' => (string) ($trackConfig['title'] ?? ('Track ' . $track->id . ' Tuning Lab')),
            'description' => (string) ($trackConfig['description'] ?? ''),
            'stem_key' => (string) ($trackConfig['stem_key'] ?? 'stem'),
            'default_source_key' => $defaultSourceKey,
            'sources' => array_values($sources),
            'focus_windows' => array_values(array_map(
                static fn (array $window): array => [
                    'id' => (string) ($window['id'] ?? ''),
                    'label' => (string) ($window['label'] ?? ''),
                    'start_sec' => (float) ($window['start_sec'] ?? 0),
                    'end_sec' => (float) ($window['end_sec'] ?? 0),
                    'kind' => (string) ($window['kind'] ?? 'window'),
                    'notes' => (string) ($window['notes'] ?? ''),
                ],
                array_values(array_filter(
                    is_array($trackConfig['focus_windows'] ?? null) ? $trackConfig['focus_windows'] : [],
                    static fn (mixed $window): bool => is_array($window),
                )),
            )),
            'detector_segments' => $this->segmentDefinitions->forTrack($track),
            'default_preset' => $trackConfig['default_preset'] ?? [],
        ];
    }

    /**
     * @return array<string, mixed>|null
     */
    public function resolveSource(Track $track, string $sourceKey): ?array
    {
        $cacheKey = (int) $track->id;
        if (array_key_exists($sourceKey, $this->resolvedSourceCache[$cacheKey] ?? [])) {
            return $this->resolvedSourceCache[$cacheKey][$sourceKey];
        }

        $trackConfig = $this->trackConfig($track);
        $resolved = is_array($trackConfig)
            ? $this->resolveSourceFromTrackConfig($track, $trackConfig, $sourceKey)
            : null;

        $this->resolvedSourceCache[$cacheKey][$sourceKey] = $resolved;

        return $resolved;
    }

    private function resolvePath(mixed $candidate): ?string
    {
        if (!is_string($candidate) || trim($candidate) === '') {
            return null;
        }

        if (is_file($candidate)) {
            $realPath = realpath($candidate);
            return $realPath !== false ? $realPath : $candidate;
        }

        if (str_starts_with($candidate, '/app/')) {
            $mapped = base_path(ltrim(substr($candidate, 5), '/'));
            if (is_file($mapped)) {
                $realPath = realpath($mapped);
                return $realPath !== false ? $realPath : $mapped;
            }
            return $mapped;
        }

        if (str_starts_with($candidate, '/var/www/html/')) {
            if (is_file($candidate)) {
                $realPath = realpath($candidate);
                return $realPath !== false ? $realPath : $candidate;
            }
            return $candidate;
        }

        return base_path(str_replace(['/', '\\'], DIRECTORY_SEPARATOR, $candidate));
    }

    /**
     * @return array<string, array<string, mixed>>
     */
    private function publicSourcesForTrack(Track $track, array $trackConfig): array
    {
        $configuredSources = is_array($trackConfig['sources'] ?? null) ? $trackConfig['sources'] : [];

        $sources = [];
        foreach ($configuredSources as $key => $source) {
            if (!is_string($key) || !is_array($source)) {
                continue;
            }

            $resolved = $this->resolveSourceFromTrackConfig($track, $trackConfig, $key);
            if ($resolved === null) {
                continue;
            }

            $sources[$key] = [
                'key' => $resolved['key'],
                'label' => $resolved['label'],
                'description' => $resolved['description'],
                'family' => $resolved['family'],
                'kind' => $resolved['kind'],
                'editable' => (bool) ($resolved['editable'] ?? str_starts_with($key, 'stem_')),
                'recommended' => $resolved['recommended'],
                'file_name' => $resolved['file_name'],
                'file_size_bytes' => $resolved['file_size_bytes'],
                'extension' => $resolved['extension'],
                'stream_url' => '/api/tracks/' . $track->id . '/tuning-lab/sources/' . $key . '/stream?v=' . rawurlencode($this->sourceRevision($resolved)),
                'waveform_url' => '/api/tracks/' . $track->id . '/tuning-lab/sources/' . $key . '/waveform?v=' . rawurlencode($this->sourceRevision($resolved)),
            ];
        }

        return $sources;
    }

    /**
     * @return array<string, mixed>|null
     */
    private function trackConfig(Track $track): ?array
    {
        $cacheKey = (int) $track->id;
        if (array_key_exists($cacheKey, $this->trackConfigCache)) {
            return $this->trackConfigCache[$cacheKey];
        }

        $configured = config('stem_tuning_lab.tracks.' . $track->id);
        if (is_array($configured)) {
            return $this->trackConfigCache[$cacheKey] = $this->expandConfiguredTrackConfig($track, $configured);
        }

        return $this->trackConfigCache[$cacheKey] = $this->buildGenericTrackConfig($track);
    }

    /**
     * @param array<string, mixed> $trackConfig
     * @return array<string, mixed>|null
     */
    private function resolveSourceFromTrackConfig(Track $track, array $trackConfig, string $sourceKey): ?array
    {
        $source = $trackConfig['sources'][$sourceKey] ?? null;
        if (!is_array($source)) {
            return null;
        }

        $path = $this->resolvePath($source['path'] ?? null);
        if ($path === null || !is_file($path)) {
            return null;
        }

        return [
            'key' => $sourceKey,
            'label' => (string) ($source['label'] ?? $sourceKey),
            'description' => (string) ($source['description'] ?? ''),
            'family' => (string) ($source['family'] ?? 'unknown'),
            'kind' => (string) ($source['kind'] ?? 'audio'),
            'editable' => (bool) ($source['editable'] ?? str_starts_with($sourceKey, 'stem_')),
            'recommended' => (bool) ($source['recommended'] ?? false),
            'path' => $path,
            'file_name' => basename($path),
            'file_size_bytes' => filesize($path) ?: 0,
            'extension' => strtolower(pathinfo($path, PATHINFO_EXTENSION)),
        ];
    }

    /**
     * @param array<string, mixed> $configured
     * @return array<string, mixed>
     */
    private function expandConfiguredTrackConfig(Track $track, array $configured): array
    {
        $sources = is_array($configured['sources'] ?? null) ? $configured['sources'] : [];

        $sourceStemManifest = $this->resolvePath($configured['source_stem_manifest'] ?? null);
        if ($sourceStemManifest !== null && is_file($sourceStemManifest)) {
            $sources = [
                ...$this->buildSourcesFromStemManifest(
                    $sourceStemManifest,
                    keyPrefix: 'stem_',
                    labelPrefix: '',
                    familyFallback: 'configured_stem_manifest',
                    kindFallback: 'stem',
                    includeMix: false,
                ),
                ...$sources,
            ];
        }

        $latestRenderPointer = $this->resolvePath($configured['render_latest_pointer'] ?? null);
        if ($latestRenderPointer !== null && is_file($latestRenderPointer)) {
            $pointer = json_decode((string) file_get_contents($latestRenderPointer), true);
            $renderManifestPath = is_array($pointer) ? $this->resolvePath($pointer['manifest_path'] ?? null) : null;
            if ($renderManifestPath !== null && is_file($renderManifestPath)) {
                $sources = [
                    ...$sources,
                    ...$this->buildSourcesFromStemManifest(
                        $renderManifestPath,
                        keyPrefix: 'render_',
                        labelPrefix: 'Rendered ',
                        familyFallback: 'tuning_render',
                        kindFallback: 'rendered_stem',
                        includeMix: true,
                    ),
                ];
            }
        }

        if (is_array($configured['extra_sources'] ?? null)) {
            foreach ($configured['extra_sources'] as $key => $source) {
                if (is_string($key) && is_array($source)) {
                    $sources[$key] = $source;
                }
            }
        }

        $configured['sources'] = $sources;

        return $configured;
    }

    /**
     * Generic fallback so the tuning lab works for any track even when no
     * stem-specific session has been authored yet.
     *
     * @return array<string, mixed>|null
     */
    private function buildGenericTrackConfig(Track $track): ?array
    {
        $sources = $this->buildGenericSources($track);
        if ($sources === []) {
            return null;
        }

        $latestRenderPointer = base_path('lab_artifacts/tuning_lab/track_' . $track->id . '/latest_render.json');
        if (is_file($latestRenderPointer)) {
            $pointer = json_decode((string) file_get_contents($latestRenderPointer), true);
            $renderManifestPath = is_array($pointer) ? $this->resolvePath($pointer['manifest_path'] ?? null) : null;
            if ($renderManifestPath !== null && is_file($renderManifestPath)) {
                $sources = [
                    ...$sources,
                    ...$this->buildSourcesFromStemManifest(
                        $renderManifestPath,
                        keyPrefix: 'render_',
                        labelPrefix: 'Rendered ',
                        familyFallback: 'tuning_render',
                        kindFallback: 'rendered_stem',
                        includeMix: true,
                    ),
                ];
            }
        }

        $defaultSourceKey = $this->genericDefaultSourceKey($sources);

        return [
            'title' => 'Track ' . $track->id . ' Tuning Lab',
            'description' => 'Stem-aware tuning lab built from auto-discovered playback stems for this track. These settings drive the container render loop and produce new stem files.',
            'stem_key' => 'stem_mix',
            'default_source_key' => $defaultSourceKey,
            'sources' => $sources,
            'focus_windows' => [],
            'default_preset' => [
                'version' => 1,
                'sourceKey' => $defaultSourceKey,
                'renderMode' => 'post_dsp',
                'layerStates' => $this->genericDefaultLayerStates($this->editableSources($sources), $defaultSourceKey),
                'hybridBuildConfig' => [
                    'modelName' => 'hybrid_htdemucs_ft_6s',
                    'sharpenBleed' => true,
                    'includeMelodyStem' => false,
                    'aggressiveMelodyFirst' => false,
                ],
                'stemRewriteConfig' => [
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
                ],
                'vocalsRewriteConfig' => [
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
                ],
                'unsafeModelInternals' => [
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
                ],
                'processorConfig' => [
                    'bypass' => false,
                    'harmonicMargin' => 1.0,
                    'percussiveMargin' => 4.0,
                    'transientCutoffHz' => 2500.0,
                    'bodyLowHz' => 90.0,
                    'bodyHighHz' => 320.0,
                    'bleedLowHz' => 700.0,
                    'bleedHighHz' => 3500.0,
                ],
                'focusWindow' => [
                    'enabled' => false,
                    'startSec' => 0,
                    'endSec' => 0,
                    'xfadeSec' => 0,
                    'targetLayerKey' => $defaultSourceKey,
                    'layerGainDb' => 0,
                    'otherLayersDuckDb' => 0,
                ],
            ],
        ];
    }

    /**
     * @return array<string, array<string, mixed>>
     */
    private function buildGenericSources(Track $track): array
    {
        $sources = [];

        $manifest = $this->playbackManifestService->manifestForTrack($track);
        foreach ($manifest['stems'] ?? [] as $stemPayload) {
            $stemKey = is_array($stemPayload) ? (string) ($stemPayload['key'] ?? '') : '';
            if ($stemKey === '') {
                continue;
            }

            $resolvedStem = $this->playbackManifestService->resolveStemSource($track, $stemKey);
            if (!is_array($resolvedStem) || !is_string($resolvedStem['path'] ?? null) || !is_file($resolvedStem['path'])) {
                continue;
            }

            $sources['stem_' . $stemKey] = [
                'label' => (string) ($resolvedStem['label'] ?? $stemKey),
                'description' => (string) ($resolvedStem['description'] ?? 'Auto-discovered stem from the playback manifest.'),
                'family' => (string) ($resolvedStem['family'] ?? 'playback_manifest'),
                'kind' => (string) ($resolvedStem['kind'] ?? 'stem'),
                'path' => (string) $resolvedStem['path'],
                'editable' => true,
                'recommended' => (bool) ($stemPayload['default_enabled'] ?? false),
                'default_enabled' => (bool) ($stemPayload['default_enabled'] ?? true),
            ];
        }

        return $sources;
    }

    /**
     * @return array<string, array<string, mixed>>
     */
    private function buildSourcesFromStemManifest(
        string $manifestPath,
        string $keyPrefix,
        string $labelPrefix,
        string $familyFallback,
        string $kindFallback,
        bool $includeMix,
    ): array {
        $decoded = json_decode((string) file_get_contents($manifestPath), true);
        if (!is_array($decoded)) {
            return [];
        }

        $sources = [];

        if ($includeMix) {
            $mixPath = $this->resolvePath($decoded['mix']['path'] ?? null);
            if ($mixPath !== null && is_file($mixPath)) {
                $sources[$keyPrefix . 'mix'] = [
                    'label' => trim($labelPrefix . 'Mix'),
                    'description' => 'Latest rendered summed mix from the tuning container loop.',
                    'family' => $familyFallback,
                    'kind' => 'rendered_mix',
                    'path' => $mixPath,
                    'editable' => false,
                    'recommended' => true,
                ];
            }
        }

        foreach ($decoded['stems'] ?? [] as $stemKey => $stemPayload) {
            if (!is_string($stemKey) || !is_array($stemPayload)) {
                continue;
            }

            $path = $this->resolvePath($stemPayload['path'] ?? null);
            if ($path === null || !is_file($path)) {
                continue;
            }

            $sources[$keyPrefix . $stemKey] = [
                'label' => trim($labelPrefix . ($stemPayload['label'] ?? $this->humanizeStemKey($stemKey))),
                'description' => (string) ($stemPayload['description'] ?? 'Stem discovered from the tuning manifest.'),
                'family' => (string) ($stemPayload['family'] ?? $familyFallback),
                'kind' => (string) ($stemPayload['kind'] ?? $kindFallback),
                'path' => $path,
                'editable' => str_starts_with($keyPrefix, 'stem_'),
                'recommended' => false,
                'default_enabled' => (bool) ($stemPayload['default_enabled'] ?? true),
            ];
        }

        return $sources;
    }

    private function humanizeStemKey(string $stemKey): string
    {
        return ucwords(str_replace(['_', '-'], ' ', $stemKey));
    }

    /**
     * @param array<string, mixed> $resolved
     */
    private function sourceRevision(array $resolved): string
    {
        $path = (string) ($resolved['path'] ?? '');
        $mtime = $path !== '' && is_file($path) ? (int) (filemtime($path) ?: 0) : 0;
        $size = (int) ($resolved['file_size_bytes'] ?? 0);

        return $mtime . '-' . $size;
    }

    /**
     * @param array<string, array<string, mixed>> $sources
     */
    private function genericDefaultSourceKey(array $sources): string
    {
        foreach ($sources as $key => $source) {
            if (str_starts_with($key, 'stem_') && (bool) ($source['default_enabled'] ?? false)) {
                return $key;
            }
        }

        foreach (array_keys($sources) as $key) {
            if (str_starts_with($key, 'stem_')) {
                return $key;
            }
        }

        return (string) array_key_first($sources);
    }

    /**
     * @param array<string, array<string, mixed>> $sources
     * @return array<string, array<string, mixed>>
     */
    private function editableSources(array $sources): array
    {
        return array_filter(
            $sources,
            static fn (array $source, string $key): bool => (bool) ($source['editable'] ?? str_starts_with($key, 'stem_')),
            ARRAY_FILTER_USE_BOTH,
        );
    }

    /**
     * @param array<string, array<string, mixed>> $sources
     * @return array<string, array{enabled: bool, gainDb: int, transientDb: int, bodyDb: int, harmonicDuckDb: int, bleedDuckDb: int}>
     */
    private function genericDefaultLayerStates(array $sources, string $defaultSourceKey): array
    {
        $hasStems = collect(array_keys($sources))->contains(
            static fn (string $key): bool => str_starts_with($key, 'stem_'),
        );

        $states = [];
        foreach ($sources as $key => $source) {
            $enabled = $hasStems
                ? (str_starts_with($key, 'stem_') && (bool) ($source['default_enabled'] ?? true))
                : $key === $defaultSourceKey;

            $states[$key] = [
                'enabled' => $enabled,
                'gainDb' => 0,
                'transientDb' => 0,
                'bodyDb' => 0,
                'harmonicDuckDb' => 0,
                'bleedDuckDb' => 0,
            ];
        }

        return $states;
    }
}
