<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;
use Illuminate\Database\Eloquent\Builder;

/**
 * Playlist Model
 * 
 * Represents a user or AI-generated playlist containing tracks.
 * Requirements: 5.1 - Playlist creation with name and description
 * 
 * @property int $id
 * @property int|null $user_id
 * @property string $name
 * @property string|null $description
 * @property bool $is_ai_generated
 * @property int $track_count
 * @property int $total_duration
 * @property \Carbon\Carbon $created_at
 * @property \Carbon\Carbon $updated_at
 */
class Playlist extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'playlists';

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'user_id',
        'name',
        'description',
        'is_ai_generated',
        'track_count',
        'total_duration',
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
            'is_ai_generated' => 'boolean',
            'track_count' => 'integer',
            'total_duration' => 'integer',
            'created_at' => 'datetime',
            'updated_at' => 'datetime',
        ];
    }

    /**
     * @return BelongsTo<User, Playlist>
     */
    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    /**
     * Get the playlist tracks for the playlist.
     *
     * @return HasMany<PlaylistTrack>
     */
    public function playlistTracks(): HasMany
    {
        return $this->hasMany(PlaylistTrack::class)->orderBy('position');
    }

    /**
     * Get the tracks in the playlist through the pivot table.
     *
     * @return BelongsToMany<Track>
     */
    public function tracks(): BelongsToMany
    {
        return $this->belongsToMany(Track::class, 'playlist_tracks')
            ->withPivot(['position', 'added_at'])
            ->orderByPivot('position');
    }

    /**
     * Scope a query to only include AI-generated playlists.
     *
     * @param Builder<Playlist> $query
     * @return Builder<Playlist>
     */
    public function scopeAiGenerated(Builder $query): Builder
    {
        return $query->where('is_ai_generated', true);
    }

    /**
     * Scope a query to only include user-created playlists.
     *
     * @param Builder<Playlist> $query
     * @return Builder<Playlist>
     */
    public function scopeUserCreated(Builder $query): Builder
    {
        return $query->where('is_ai_generated', false);
    }

    /**
     * Scope a query to playlists owned by the given user.
     *
     * @param Builder<Playlist> $query
     * @return Builder<Playlist>
     */
    public function scopeOwnedBy(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereNull('user_id')
            : $query->where('user_id', $userId);
    }
}
