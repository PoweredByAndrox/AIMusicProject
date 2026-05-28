<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;

/**
 * Pattern Model
 * 
 * Represents a melodic pattern that can be searched across tracks.
 */
class Pattern extends Model
{
    protected $fillable = [
        'name',
        'description',
        'representation_type',
        'transpose_invariant',
        'tempo_invariant',
        'match_threshold',
        'importance',
        'prototype_path',
    ];

    protected function casts(): array
    {
        return [
            'transpose_invariant' => 'boolean',
            'tempo_invariant' => 'boolean',
            'match_threshold' => 'float',
            'importance' => 'integer',
        ];
    }

    public function examples(): HasMany
    {
        return $this->hasMany(PatternExample::class);
    }

    public function matches(): HasMany
    {
        return $this->hasMany(PatternMatch::class);
    }

    public function positiveExamples(): HasMany
    {
        return $this->hasMany(PatternExample::class)->where('is_positive', true);
    }

    public function negativeExamples(): HasMany
    {
        return $this->hasMany(PatternExample::class)->where('is_positive', false);
    }
}
