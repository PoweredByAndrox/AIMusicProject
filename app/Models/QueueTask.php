<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Builder;

/**
 * QueueTask Model
 * 
 * Monitors ML worker task processing for analysis, training, and recommendations.
 * Requirements: 13.2, 13.3
 * 
 * @property int $id
 * @property string $task_type
 * @property array $payload
 * @property string $status
 * @property string|null $error_message
 * @property \Carbon\Carbon $created_at
 * @property \Carbon\Carbon|null $completed_at
 */
class QueueTask extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'queue_tasks';

    /**
     * Indicates if the model should be timestamped.
     * Note: queue_tasks table only has created_at, no updated_at
     *
     * @var bool
     */
    public $timestamps = true;

    /**
     * The name of the "updated at" column.
     * Set to null since queue_tasks table doesn't have updated_at
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
        'task_type',
        'payload',
        'status',
        'error_message',
        'completed_at',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'payload' => 'array',
            'created_at' => 'datetime',
            'completed_at' => 'datetime',
        ];
    }

    /**
     * Task type constants
     */
    public const TYPE_ANALYSIS = 'analysis';
    public const TYPE_FRAGMENT_ANALYSIS = 'fragment_analysis';
    public const TYPE_TRAINING = 'training';
    public const TYPE_RECOMMENDATION = 'recommendation';
    public const TYPE_STEM_TUNING_RENDER = 'stem_tuning_render';
    public const TYPE_STEM_SOURCE_BUILD = 'stem_source_build';
    public const TYPE_QUERYSEP_LONG_TAIL_BUILD = 'querysep_long_tail_build';

    /**
     * Status constants
     */
    public const STATUS_PENDING = 'pending';
    public const STATUS_PROCESSING = 'processing';
    public const STATUS_COMPLETED = 'completed';
    public const STATUS_CANCELLED = 'cancelled';
    public const STATUS_FAILED = 'failed';

    /**
     * Scope a query to filter by task type.
     *
     * @param Builder<QueueTask> $query
     * @param string $type
     * @return Builder<QueueTask>
     */
    public function scopeOfType(Builder $query, string $type): Builder
    {
        return $query->where('task_type', $type);
    }

    /**
     * Scope a query to filter by status.
     *
     * @param Builder<QueueTask> $query
     * @param string $status
     * @return Builder<QueueTask>
     */
    public function scopeWithStatus(Builder $query, string $status): Builder
    {
        return $query->where('status', $status);
    }

    /**
     * Scope a query to only include pending tasks.
     *
     * @param Builder<QueueTask> $query
     * @return Builder<QueueTask>
     */
    public function scopePending(Builder $query): Builder
    {
        return $query->where('status', self::STATUS_PENDING);
    }

    /**
     * Scope a query to only include failed tasks.
     *
     * @param Builder<QueueTask> $query
     * @return Builder<QueueTask>
     */
    public function scopeFailed(Builder $query): Builder
    {
        return $query->where('status', self::STATUS_FAILED);
    }

    /**
     * Check if the task is completed.
     *
     * @return bool
     */
    public function isCompleted(): bool
    {
        return $this->status === self::STATUS_COMPLETED;
    }

    /**
     * Check if the task has failed.
     *
     * @return bool
     */
    public function isFailed(): bool
    {
        return $this->status === self::STATUS_FAILED;
    }

    /**
     * Check if the task was cancelled.
     *
     * @return bool
     */
    public function isCancelled(): bool
    {
        return $this->status === self::STATUS_CANCELLED;
    }

    /**
     * Mark the task as completed.
     *
     * @return bool
     */
    public function markAsCompleted(): bool
    {
        return $this->update([
            'status' => self::STATUS_COMPLETED,
            'completed_at' => now(),
        ]);
    }

    /**
     * Mark the task as failed with an error message.
     *
     * @param string $errorMessage
     * @return bool
     */
    public function markAsFailed(string $errorMessage): bool
    {
        return $this->update([
            'status' => self::STATUS_FAILED,
            'error_message' => $errorMessage,
            'completed_at' => now(),
        ]);
    }

    /**
     * Mark the task as cancelled with an optional message.
     *
     * @param string|null $message
     * @return bool
     */
    public function markAsCancelled(?string $message = null): bool
    {
        return $this->update([
            'status' => self::STATUS_CANCELLED,
            'error_message' => $message,
            'completed_at' => now(),
        ]);
    }
}
