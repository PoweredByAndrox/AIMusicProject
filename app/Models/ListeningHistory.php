<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Builder;

/**
 * ListeningHistory Model
 * 
 * Represents a playback history record for a track.
 * Requirements: 16.3 - Record track_id, timestamp, duration_played, completed
 * 
 * @property int $id
 * @property int $track_id
 * @property \Carbon\Carbon $started_at
 * @property float|null $duration_played
 * @property bool $completed
 * @property string|null $source
 */
class ListeningHistory extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'listening_history';

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
        'started_at',
        'duration_played',
        'completed',
        'source',
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
            'started_at' => 'datetime',
            'duration_played' => 'float',
            'completed' => 'boolean',
        ];
    }

    /**
     * Get the track that owns the listening history entry.
     *
     * @return BelongsTo<Track, ListeningHistory>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function scopeForUser(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereNull('user_id')
            : $query->where('user_id', $userId);
    }

    /**
     * Scope a query to only include completed plays.
     *
     * @param Builder<ListeningHistory> $query
     * @return Builder<ListeningHistory>
     */
    public function scopeCompleted(Builder $query): Builder
    {
        return $query->where('completed', true);
    }

    /**
     * Scope a query to filter by source.
     *
     * @param Builder<ListeningHistory> $query
     * @param string $source
     * @return Builder<ListeningHistory>
     */
    public function scopeBySource(Builder $query, string $source): Builder
    {
        return $query->where('source', $source);
    }

    /**
     * Scope a query to filter by recent plays within hours.
     *
     * @param Builder<ListeningHistory> $query
     * @param int $hours
     * @return Builder<ListeningHistory>
     */
    public function scopeRecentWithinHours(Builder $query, int $hours): Builder
    {
        return $query->where('started_at', '>=', now()->subHours($hours));
    }
}
