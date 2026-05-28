<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

class DirectMessage extends Model
{
    public const TYPE_MESSAGE = 'message';
    public const TYPE_TRACK_SHARE = 'track_share';
    public const TYPE_SEGMENT_SHARE = 'segment_share';

    protected $fillable = [
        'sender_id',
        'recipient_id',
        'type',
        'body',
        'payload',
        'read_at',
    ];

    protected function casts(): array
    {
        return [
            'sender_id' => 'integer',
            'recipient_id' => 'integer',
            'payload' => 'array',
            'read_at' => 'datetime',
            'created_at' => 'datetime',
            'updated_at' => 'datetime',
        ];
    }

    /**
     * @return BelongsTo<User, DirectMessage>
     */
    public function sender(): BelongsTo
    {
        return $this->belongsTo(User::class, 'sender_id');
    }

    /**
     * @return BelongsTo<User, DirectMessage>
     */
    public function recipient(): BelongsTo
    {
        return $this->belongsTo(User::class, 'recipient_id');
    }
}
