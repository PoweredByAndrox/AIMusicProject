<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class UserTrackScore extends Model
{
    protected $fillable = [
        'user_id',
        'track_id',
        'preference_score',
        'score_calculated_at',
    ];

    protected function casts(): array
    {
        return [
            'user_id' => 'integer',
            'track_id' => 'integer',
            'preference_score' => 'float',
            'score_calculated_at' => 'datetime',
            'created_at' => 'datetime',
            'updated_at' => 'datetime',
        ];
    }

    /**
     * @param Builder<UserTrackScore> $query
     * @return Builder<UserTrackScore>
     */
    public function scopeForUser(Builder $query, int $userId): Builder
    {
        return $query->where('user_id', $userId);
    }

    /**
     * @return BelongsTo<User, UserTrackScore>
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * @return BelongsTo<Track, UserTrackScore>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }
}
