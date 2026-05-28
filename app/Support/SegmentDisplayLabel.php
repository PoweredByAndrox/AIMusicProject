<?php

namespace App\Support;

final class SegmentDisplayLabel
{
    /**
     * @var array<string, string>
     */
    private const LABELS = [
        'intro' => 'INTRO',
        'melodic' => 'MELODIC',
        'outro' => 'OUTRO',
        'breakdown' => 'BREAKDOWN',
        'break' => 'BREAK',
        'buildup' => 'BUILD-UP',
        'build_up' => 'BUILD-UP',
        'drop' => 'DROP',
        'drop_bass' => 'DROP BASS',
        'verse' => 'VERSE',
        'chorus' => 'CHORUS',
        'bridge' => 'BRIDGE',
        'vocal' => 'VOCAL',
        'instrumental' => 'INSTRUMENTAL',
        'hook' => 'HOOK',
        'climax' => 'CLIMAX',
        'groove' => 'GROOVE',
        'ambient' => 'AMBIENT',
        'bass' => 'BASS',
        'lead' => 'LEAD',
        'theme' => 'THEME',
        'segment' => 'SEGMENT',
        'unknown' => 'SEGMENT',
    ];

    public static function extractTypeFromPatternName(?string $patternName): string
    {
        if (preg_match('/\[auto:([a-z0-9_]+)\]/i', $patternName ?? '', $matches)) {
            return strtolower((string) $matches[1]);
        }

        return 'unknown';
    }

    public static function fromType(?string $type): string
    {
        $normalized = strtolower(trim((string) ($type ?? 'segment')));
        if ($normalized === '') {
            return 'SEGMENT';
        }

        if (array_key_exists($normalized, self::LABELS)) {
            return self::LABELS[$normalized];
        }

        return strtoupper(str_replace('_', ' ', $normalized));
    }
}
