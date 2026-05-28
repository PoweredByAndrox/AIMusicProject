<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Builder;

/**
 * Fragment Model
 * 
 * Represents a marked audio segment within a track with time boundaries.
 * Supports rating (like/dislike) and comments for user feedback.
 * Requirements: 4.3, 6.1, 6.2, 7.1, 8.1
 * 
 * @property int $id
 * @property int $track_id
 * @property float $start_time
 * @property float $end_time
 * @property bool|null $is_liked null = not rated, true = liked, false = disliked
 * @property string|null $comment User comment explaining the rating (max 500 chars)
 * @property \Carbon\Carbon|null $rated_at Timestamp when fragment was rated
 * @property string|null $embedding_path
 * @property string $analysis_status
 * @property \Carbon\Carbon $created_at
 */
class Fragment extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'fragments';

    /**
     * Indicates if the model should be timestamped.
     * Note: fragments table only has created_at, no updated_at
     *
     * @var bool
     */
    public $timestamps = true;

    /**
     * The name of the "updated at" column.
     * Set to null since fragments table doesn't have updated_at
     *
     * @var string|null
     */
    const UPDATED_AT = null;

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'track_id',
        'user_id',
        'start_time',
        'end_time',
        'is_liked',
        'comment',
        'rated_at',
        'embedding_path',
        'analysis_status',
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
            'start_time' => 'float',
            'end_time' => 'float',
            'is_liked' => 'boolean', // nullable boolean: null = not rated
            'comment' => 'string',
            'rated_at' => 'datetime',
            'created_at' => 'datetime',
        ];
    }

    /**
     * Get the track that owns the fragment.
     *
     * @return BelongsTo<Track, Fragment>
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
     * Scope a query to only include liked fragments.
     *
     * @param Builder<Fragment> $query
     * @return Builder<Fragment>
     */
    public function scopeLiked(Builder $query): Builder
    {
        return $query->where('is_liked', true);
    }

    /**
     * Scope a query to only include disliked fragments.
     *
     * @param Builder<Fragment> $query
     * @return Builder<Fragment>
     */
    public function scopeDisliked(Builder $query): Builder
    {
        return $query->where('is_liked', false);
    }

    /**
     * Scope a query to only include rated fragments (liked or disliked).
     *
     * @param Builder<Fragment> $query
     * @return Builder<Fragment>
     */
    public function scopeRated(Builder $query): Builder
    {
        return $query->whereNotNull('is_liked');
    }

    /**
     * Scope a query to only include unrated fragments.
     *
     * @param Builder<Fragment> $query
     * @return Builder<Fragment>
     */
    public function scopeUnrated(Builder $query): Builder
    {
        return $query->whereNull('is_liked');
    }

    /**
     * Scope a query to only include fragments with comments.
     * Requirements: 8.6
     *
     * @param Builder<Fragment> $query
     * @return Builder<Fragment>
     */
    public function scopeWithComment(Builder $query): Builder
    {
        return $query->whereNotNull('comment');
    }

    /**
     * Scope a query to only include fragments without comments.
     * Requirements: 8.6
     *
     * @param Builder<Fragment> $query
     * @return Builder<Fragment>
     */
    public function scopeWithoutComment(Builder $query): Builder
    {
        return $query->whereNull('comment');
    }

    /**
     * Scope a query to only include analyzed fragments.
     *
     * @param Builder<Fragment> $query
     * @return Builder<Fragment>
     */
    public function scopeAnalyzed(Builder $query): Builder
    {
        return $query->where('analysis_status', 'completed');
    }

    /**
     * Get the duration of the fragment in seconds.
     *
     * @return float
     */
    public function getDurationAttribute(): float
    {
        return $this->end_time - $this->start_time;
    }

    /**
     * Check if the fragment has been rated.
     *
     * @return bool
     */
    public function isRated(): bool
    {
        return $this->is_liked !== null;
    }

    /**
     * Check if the fragment has a comment.
     *
     * @return bool
     */
    public function hasComment(): bool
    {
        return $this->comment !== null && $this->comment !== '';
    }
}
