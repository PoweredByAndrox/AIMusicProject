<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * PatternExample Model
 * 
 * Represents a positive or negative example of a pattern.
 */
class PatternExample extends Model
{
    protected $fillable = [
        'pattern_id',
        'track_id',
        'start_time',
        'end_time',
        'is_positive',
        'rating',
        'notes',
        'representation_path',
        'embedding_path',
    ];

    protected function casts(): array
    {
        return [
            'start_time' => 'float',
            'end_time' => 'float',
            'is_positive' => 'boolean',
            'rating' => 'float',
            'embedding_path' => 'string',
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
