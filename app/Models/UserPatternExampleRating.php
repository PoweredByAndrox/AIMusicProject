<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Builder;

class UserPatternExampleRating extends Model
{
    protected $fillable = [
        'user_id',
        'pattern_example_id',
        'rating',
        'is_positive',
    ];

    protected function casts(): array
    {
        return [
            'user_id' => 'integer',
            'pattern_example_id' => 'integer',
            'rating' => 'float',
            'is_positive' => 'boolean',
            'created_at' => 'datetime',
            'updated_at' => 'datetime',
        ];
    }

    public function scopeForUser(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereRaw('1 = 0')
            : $query->where('user_id', $userId);
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function patternExample(): BelongsTo
    {
        return $this->belongsTo(PatternExample::class);
    }
}
