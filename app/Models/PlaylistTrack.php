<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;

/**
 * PlaylistTrack Model
 * 
 * Represents the pivot relationship between playlists and tracks with position ordering.
 * Requirements: 5.2 - Adding tracks to playlists with position ordering
 * 
 * @property int $id
 * @property int $playlist_id
 * @property int $track_id
 * @property int $position
 * @property \Carbon\Carbon $added_at
 */
class PlaylistTrack extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'playlist_tracks';

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
        'playlist_id',
        'track_id',
        'position',
        'added_at',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'playlist_id' => 'integer',
            'track_id' => 'integer',
            'position' => 'integer',
            'added_at' => 'datetime',
        ];
    }

    /**
     * Get the playlist that owns the playlist track.
     *
     * @return BelongsTo<Playlist, PlaylistTrack>
     */
    public function playlist(): BelongsTo
    {
        return $this->belongsTo(Playlist::class);
    }

    /**
     * Get the track that belongs to the playlist track.
     *
     * @return BelongsTo<Track, PlaylistTrack>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }
}
