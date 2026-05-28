<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Builder;

/**
 * RadioFeedback Model
 * 
 * Represents Smart Radio "more/less" feedback for a track.
 * Requirements: 17.4 - Save feedback for Preference_Model training
 * 
 * @property int $id
 * @property int|null $user_id
 * @property int $track_id
 * @property string $feedback_type
 * @property \Carbon\Carbon $created_at
 */
class RadioFeedback extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'radio_feedback';

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
        'feedback_type',
        'created_at',
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
            'created_at' => 'datetime',
        ];
    }

    /**
     * Get the track that owns the feedback.
     *
     * @return BelongsTo<Track, RadioFeedback>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }

    /**
     * Get the user that owns the feedback.
     *
     * @return BelongsTo<User, RadioFeedback>
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Scope a query to only include "more" feedback.
     *
     * @param Builder<RadioFeedback> $query
     * @return Builder<RadioFeedback>
     */
    public function scopeMore(Builder $query): Builder
    {
        return $query->where('feedback_type', 'more');
    }

    /**
     * Scope a query to only include "less" feedback.
     *
     * @param Builder<RadioFeedback> $query
     * @return Builder<RadioFeedback>
     */
    public function scopeLess(Builder $query): Builder
    {
        return $query->where('feedback_type', 'less');
    }

    /**
     * Scope a query to a specific user.
     *
     * @param Builder<RadioFeedback> $query
     * @param int|null $userId
     * @return Builder<RadioFeedback>
     */
    public function scopeForUser(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereRaw('1 = 0')
            : $query->where('user_id', $userId);
    }
}
