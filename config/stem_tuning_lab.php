<?php

return [
    'tracks' => [
        15 => [
            'title' => 'Track 15 Stem Tuning Lab',
            'description' => 'Container-backed stem tuning lab for Track 15. This session uses the accepted Hybrid6S stem set as the base source and can render new stem files through the ML worker from your UI preset.',
            'stem_key' => 'stem_mix',
            'default_source_key' => 'stem_drums',
            'source_stem_manifest' => 'lab_artifacts/2026-03-29_track_15_hq_stems_hybrid6s_full/stem_manifest.json',
            'render_latest_pointer' => 'lab_artifacts/tuning_lab/track_15/latest_render.json',
            'raw_reference_path' => 'lab_artifacts/2026-03-29_track_15_minus_instrumental_resurrection_full/raw_ref.flac',
            'extra_sources' => [
                'raw_reference' => [
                    'label' => 'Raw Reference',
                    'description' => 'Accepted full-track reference for comparison only.',
                    'family' => 'reference',
                    'kind' => 'full_mix_reference',
                    'path' => 'lab_artifacts/2026-03-29_track_15_minus_instrumental_resurrection_full/raw_ref.flac',
                ],
            ],
            'focus_windows' => [],
            'default_preset' => [
                'version' => 1,
                'sourceKey' => 'stem_drums',
                'layerStates' => [
                    'stem_drums' => [
                        'enabled' => true,
                        'gainDb' => 0,
                        'transientDb' => 0,
                        'bodyDb' => 0,
                        'harmonicDuckDb' => 0,
                        'bleedDuckDb' => 0,
                    ],
                    'stem_bass' => [
                        'enabled' => true,
                        'gainDb' => 0,
                        'transientDb' => 0,
                        'bodyDb' => 0,
                        'harmonicDuckDb' => 0,
                        'bleedDuckDb' => 0,
                    ],
                    'stem_other' => [
                        'enabled' => true,
                        'gainDb' => 0,
                        'transientDb' => 0,
                        'bodyDb' => 0,
                        'harmonicDuckDb' => 0,
                        'bleedDuckDb' => 0,
                    ],
                    'stem_vocals' => [
                        'enabled' => true,
                        'gainDb' => 0,
                        'transientDb' => 0,
                        'bodyDb' => 0,
                        'harmonicDuckDb' => 0,
                        'bleedDuckDb' => 0,
                    ],
                    'stem_guitar' => [
                        'enabled' => true,
                        'gainDb' => 0,
                        'transientDb' => 0,
                        'bodyDb' => 0,
                        'harmonicDuckDb' => 0,
                        'bleedDuckDb' => 0,
                    ],
                    'stem_piano' => [
                        'enabled' => true,
                        'gainDb' => 0,
                        'transientDb' => 0,
                        'bodyDb' => 0,
                        'harmonicDuckDb' => 0,
                        'bleedDuckDb' => 0,
                    ],
                ],
                'focusWindow' => [
                    'enabled' => false,
                    'startSec' => 0,
                    'endSec' => 0,
                    'xfadeSec' => 0,
                    'targetLayerKey' => 'stem_drums',
                    'layerGainDb' => 0,
                    'otherLayersDuckDb' => 0,
                ],
            ],
        ],
    ],
];
