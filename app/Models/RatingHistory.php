<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * RatingHistory Model
 * 
 * Represents a historical record of rating changes for a track.
 * Requirements: 14.1 - Record rating changes with old_rating, new_rating, timestamp
 * 
 * @property int $id
 * @property int|null $user_id
 * @property int|null $track_id
 * @property int|null $old_rating
 * @property int|null $new_rating
 * @property \Carbon\Carbon $changed_at
 */
class RatingHistory extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'rating_history';

    /**
     * Indicates if the model should be timestamped.
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
        'old_rating',
        'new_rating',
        'changed_at',
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
            'old_rating' => 'integer',
            'new_rating' => 'integer',
            'changed_at' => 'datetime',
        ];
    }

    /**
     * Get the track that owns the rating history entry.
     *
     * @return BelongsTo<Track, RatingHistory>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }

    /**
     * @return BelongsTo<User, RatingHistory>
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Scope query to a specific user.
     *
     * @param Builder<RatingHistory> $query
     * @return Builder<RatingHistory>
     */
    public function scopeForUser(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereNull('user_id')
            : $query->where('user_id', $userId);
    }
}
