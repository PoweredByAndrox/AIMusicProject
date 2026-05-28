<?php

namespace App\Services;

use App\Models\Track;

class StemTuningSegmentDefinitionService
{
    /**
     * @return array<string, mixed>|null
     */
    public function forTrack(Track $track): ?array
    {
        $path = $this->cachePathForTrack((int) $track->id);
        if (!is_file($path)) {
            return null;
        }

        $payload = json_decode((string) file_get_contents($path), true);
        if (!is_array($payload)) {
            return null;
        }

        return [
            'track_id' => (int) ($payload['trackId'] ?? $track->id),
            'generated_at' => (string) ($payload['generatedAt'] ?? ''),
            'source' => [
                'structure_cache_path' => (string) (($payload['source']['structureCachePath'] ?? '')),
                'playback_manifest_path' => (string) (($payload['source']['playbackManifestPath'] ?? '')),
                'audio_path' => (string) (($payload['source']['audioPath'] ?? '')),
                'model_name' => (string) (($payload['source']['modelName'] ?? '')),
            ],
            'analysis' => [
                'duration_sec' => (float) (($payload['analysis']['durationSec'] ?? 0)),
                'frame_sec' => (float) (($payload['analysis']['frameSec'] ?? 0)),
                'window_sec' => (float) (($payload['analysis']['windowSec'] ?? 0)),
                'sample_rate' => (int) (($payload['analysis']['sampleRate'] ?? 0)),
                'vocal_activity_threshold_rms' => (float) (($payload['analysis']['vocalActivityThresholdRms'] ?? 0)),
                'accompaniment_activity_threshold_rms' => (float) (($payload['analysis']['accompanimentActivityThresholdRms'] ?? 0)),
            ],
            'segments' => array_values(array_map(
                static fn (array $segment): array => [
                    'segmentId' => (string) ($segment['segmentId'] ?? ''),
                    'startSec' => (float) ($segment['startSec'] ?? 0),
                    'endSec' => (float) ($segment['endSec'] ?? 0),
                    'durationSec' => (float) ($segment['durationSec'] ?? 0),
                    'structureLabel' => (string) ($segment['structureLabel'] ?? ''),
                    'kinds' => array_values(array_map('strval', is_array($segment['kinds'] ?? null) ? $segment['kinds'] : [])),
                    'scores' => [
                        'segmentPriority' => (float) (($segment['scores']['segmentPriority'] ?? 0)),
                        'vocalActivityFraction' => (float) (($segment['scores']['vocalActivityFraction'] ?? 0)),
                        'vocalMeanPercentile' => (float) (($segment['scores']['vocalMeanPercentile'] ?? 0)),
                        'vocalPeak' => (float) (($segment['scores']['vocalPeak'] ?? 0)),
                        'accompanimentMeanPercentile' => (float) (($segment['scores']['accompanimentMeanPercentile'] ?? 0)),
                        'accompanimentPeakPercentile' => (float) (($segment['scores']['accompanimentPeakPercentile'] ?? 0)),
                        'accompanimentOnsetPercentile' => (float) (($segment['scores']['accompanimentOnsetPercentile'] ?? 0)),
                        'transitionLift' => (float) (($segment['scores']['transitionLift'] ?? 0)),
                        'transitionPercentile' => (float) (($segment['scores']['transitionPercentile'] ?? 0)),
                        'brightnessNormMean' => (float) (($segment['scores']['brightnessNormMean'] ?? 0)),
                    ],
                    'metrics' => [
                        'vocalMeanRms' => (float) (($segment['metrics']['vocalMeanRms'] ?? 0)),
                        'accompanimentMeanRms' => (float) (($segment['metrics']['accompanimentMeanRms'] ?? 0)),
                        'accompanimentOnsetMean' => (float) (($segment['metrics']['accompanimentOnsetMean'] ?? 0)),
                        'brightnessMeanHz' => (float) (($segment['metrics']['brightnessMeanHz'] ?? 0)),
                    ],
                    'window' => [
                        'startSec' => (float) (($segment['window']['startSec'] ?? 0)),
                        'endSec' => (float) (($segment['window']['endSec'] ?? 0)),
                        'score' => (float) (($segment['window']['score'] ?? 0)),
                    ],
                ],
                array_values(array_filter(
                    is_array($payload['segments'] ?? null) ? $payload['segments'] : [],
                    static fn (mixed $segment): bool => is_array($segment),
                )),
            )),
            'selected' => [
                'vocal_drop' => $this->normalizeSelectedWindows($payload['selected']['vocal_drop'] ?? []),
                'drop_like' => $this->normalizeSelectedWindows($payload['selected']['drop_like'] ?? []),
            ],
        ];
    }

    public function cachePathForTrack(int $trackId): string
    {
        return storage_path('app/analysis_cache/segment-definitions/track_' . $trackId . '_vocal_drops.json');
    }

    /**
     * @param mixed $windows
     * @return array<int, array<string, mixed>>
     */
    private function normalizeSelectedWindows(mixed $windows): array
    {
        return array_values(array_map(
            static fn (array $window): array => [
                'segmentId' => (string) ($window['segmentId'] ?? ''),
                'startSec' => (float) ($window['startSec'] ?? 0),
                'endSec' => (float) ($window['endSec'] ?? 0),
                'score' => (float) ($window['score'] ?? 0),
                'segmentStartSec' => (float) ($window['segmentStartSec'] ?? 0),
                'segmentEndSec' => (float) ($window['segmentEndSec'] ?? 0),
            ],
            array_values(array_filter(
                is_array($windows) ? $windows : [],
                static fn (mixed $window): bool => is_array($window),
            )),
        ));
    }
}
