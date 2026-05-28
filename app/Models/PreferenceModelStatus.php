<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * PreferenceModelStatus Model
 * 
 * Tracks the state of the personal preference model for recommendations.
 * Requirements: 8.1, 8.2, 8.3
 * 
 * @property int $id
 * @property int|null $user_id
 * @property string $status
 * @property string|null $model_path
 * @property \Carbon\Carbon|null $trained_at
 * @property int $likes_count
 * @property \Carbon\Carbon $created_at
 * @property \Carbon\Carbon $updated_at
 */
class PreferenceModelStatus extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'preference_model_status';

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'user_id',
        'status',
        'model_path',
        'trained_at',
        'likes_count',
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
            'trained_at' => 'datetime',
            'likes_count' => 'integer',
            'created_at' => 'datetime',
            'updated_at' => 'datetime',
        ];
    }

    /**
     * Status constants
     */
    public const STATUS_NOT_TRAINED = 'not_trained';
    public const STATUS_TRAINING = 'training';
    public const STATUS_TRAINED = 'trained';

    /**
     * Check if the model is trained.
     *
     * @return bool
     */
    public function isTrained(): bool
    {
        return $this->status === self::STATUS_TRAINED;
    }

    /**
     * Check if the model is currently training.
     *
     * @return bool
     */
    public function isTraining(): bool
    {
        return $this->status === self::STATUS_TRAINING;
    }

    /**
     * Check if there are enough likes to train the model.
     * Minimum threshold is 10 likes.
     *
     * @return bool
     */
    public function hasEnoughLikes(): bool
    {
        return $this->likes_count >= 10;
    }

    /**
     * @param Builder<PreferenceModelStatus> $query
     * @return Builder<PreferenceModelStatus>
     */
    public function scopeForUser(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereNull('user_id')
            : $query->where('user_id', $userId);
    }

    /**
     * @return BelongsTo<User, PreferenceModelStatus>
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }
}
