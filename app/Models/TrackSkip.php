<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Builder;

/**
 * TrackSkip Model
 * 
 * Represents a skip event for a track.
 * Requirements: 18.2 - Store track_id, timestamp, position_at_skip, source
 * 
 * @property int $id
 * @property int|null $user_id
 * @property int $track_id
 * @property float|null $position_at_skip
 * @property string|null $source
 * @property \Carbon\Carbon $skipped_at
 */
class TrackSkip extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'track_skips';

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
        'position_at_skip',
        'source',
        'skipped_at',
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
            'position_at_skip' => 'float',
            'skipped_at' => 'datetime',
        ];
    }

    /**
     * Get the track that owns the skip event.
     *
     * @return BelongsTo<Track, TrackSkip>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }

    /**
     * Scope a query to only include early skips (within first 30 seconds).
     *
     * @param Builder<TrackSkip> $query
     * @return Builder<TrackSkip>
     */
    public function scopeEarlySkips(Builder $query): Builder
    {
        return $query->where('position_at_skip', '<', 30);
    }

    /**
     * Scope a query to filter by source.
     *
     * @param Builder<TrackSkip> $query
     * @param string $source
     * @return Builder<TrackSkip>
     */
    public function scopeBySource(Builder $query, string $source): Builder
    {
        return $query->where('source', $source);
    }
}
