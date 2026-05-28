<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * UserTrackPreference Model
 * 
 * Represents user-specific preferences for a track (volume, last position).
 * Requirements: 4.4 - Store preferences in database with track_id binding
 * 
 * @property int $id
 * @property int|null $user_id
 * @property int $track_id
 * @property float $volume
 * @property float $last_position
 * @property \Carbon\Carbon $updated_at
 */
class UserTrackPreference extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'user_track_preferences';

    /**
     * Indicates if the model should be timestamped.
     * Note: This table only has updated_at, no created_at
     *
     * @var bool
     */
    public $timestamps = false;

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'user_id',
        'track_id',
        'volume',
        'last_position',
        'updated_at',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'user_id' => 'integer',
            'track_id' => 'integer',
            'volume' => 'float',
            'last_position' => 'float',
            'updated_at' => 'datetime',
        ];
    }

    /**
     * Get the track that owns the preference.
     *
     * @return BelongsTo<Track, UserTrackPreference>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }

    /**
     * @return BelongsTo<User, UserTrackPreference>
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Scope query to a specific user.
     *
     * @param Builder<UserTrackPreference> $query
     * @return Builder<UserTrackPreference>
     */
    public function scopeForUser(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereNull('user_id')
            : $query->where('user_id', $userId);
    }
}
