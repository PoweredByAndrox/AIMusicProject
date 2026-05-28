<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * PatternMatch Model
 * 
 * Represents a found match of a pattern in a track.
 */
class PatternMatch extends Model
{
    protected $fillable = [
        'pattern_id',
        'track_id',
        'start_time',
        'end_time',
        'score',
        'confirmed',
    ];

    protected function casts(): array
    {
        return [
            'start_time' => 'float',
            'end_time' => 'float',
            'score' => 'float',
            'confirmed' => 'boolean',
        ];
    }

    public function pattern(): BelongsTo
    {
        return $this->belongsTo(Pattern::class);
    }

    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }
}
