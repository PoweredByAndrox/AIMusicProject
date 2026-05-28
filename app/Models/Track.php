<?php

namespace App\Models;

use App\Events\TrackWithFilePathCreated;
use App\Services\ArtistCreditParser;
use App\Services\TrackSidecarWriter;
use Illuminate\Database\Eloquent\Casts\Attribute;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\HasMany;
use Illuminate\Database\Eloquent\Relations\HasOne;
use Illuminate\Database\Eloquent\Relations\BelongsToMany;
use Illuminate\Database\Eloquent\Builder;
use Illuminate\Support\Facades\Auth;

/**
 * Track Model
 * 
 * Represents an audio track with metadata and ML characteristics.
 * Requirements: 4.1, 4.2, 4.3
 * 
 * @property int $id
 * @property string $title
 * @property string|null $artist
 * @property array<int, array{name: string, slug: string, roles: array<int, string>}>|null $artist_credits
 * @property array<int, string>|null $artist_slugs
 * @property string|null $album
 * @property int|null $year
 * @property float $duration
 * @property string $file_path
 * @property string $source
 * @property string|null $youtube_url
 * @property string|null $genre
 * @property float|null $bpm
 * @property string|null $key
 * @property string|null $mode
 * @property string|null $mood
 * @property bool|null $has_vocals
 * @property bool|null $is_acoustic
 * @property float|null $loudness_integrated_lufs
 * @property float|null $loudness_true_peak_db
 * @property float|null $loudness_gain_db
 * @property float|null $loudness_target_lufs
 * @property string $analysis_status
 * @property bool $is_liked
 * @property array $manually_edited
 * @property string|null $embedding_path
 * @property \Carbon\Carbon $created_at
 * @property \Carbon\Carbon $updated_at
 */
class Track extends Model
{
    use HasFactory;

    protected static function booted(): void
    {
        static::created(function (self $track): void {
            static::dispatchWaveformPrewarmEvent($track);
            app(TrackSidecarWriter::class)->writeForTrack($track);
        });

        static::updated(function (self $track): void {
            if ($track->wasChanged('file_path')) {
                $previousFilePath = $track->getOriginal('file_path');
                if (is_string($previousFilePath) && trim($previousFilePath) !== '') {
                    app(TrackSidecarWriter::class)->deleteForRelativePath($previousFilePath);
                }
            }

            app(TrackSidecarWriter::class)->writeForTrack($track);

            if (!$track->wasChanged('file_path')) {
                return;
            }

            $previousFilePath = $track->getOriginal('file_path');
            if (is_string($previousFilePath) && trim($previousFilePath) !== '') {
                return;
            }

            static::dispatchWaveformPrewarmEvent($track);
        });
    }

    private static function dispatchWaveformPrewarmEvent(self $track): void
    {
        if (!is_string($track->file_path) || trim($track->file_path) === '') {
            return;
        }

        event(new TrackWithFilePathCreated($track));
    }

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'tracks';

    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'title',
        'artist',
        'artist_credits',
        'artist_slugs',
        'album',
        'year',
        'duration',
        'file_path',
        'source',
        'youtube_url',
        'youtube_video_id',
        'youtube_channel',
        'genre',
        'genre_tags',
        'bpm',
        'key',
        'mode',
        'mood',
        'mood_tags',
        'has_vocals',
        'is_acoustic',
        'loudness_integrated_lufs',
        'loudness_true_peak_db',
        'loudness_gain_db',
        'loudness_target_lufs',
        'analysis_status',
        'is_liked',
        'manually_edited',
        'embedding_path',
        'chroma_path',
        'pitch_contour_path',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'duration' => 'float',
            'year' => 'integer',
            'bpm' => 'float',
            'has_vocals' => 'boolean',
            'is_acoustic' => 'boolean',
            'loudness_integrated_lufs' => 'float',
            'loudness_true_peak_db' => 'float',
            'loudness_gain_db' => 'float',
            'loudness_target_lufs' => 'float',
            'is_liked' => 'boolean',
            'manually_edited' => 'array',
            'artist_credits' => 'array',
            'artist_slugs' => 'array',
            'genre_tags' => 'array',
            'mood_tags' => 'array',
            'created_at' => 'datetime',
            'updated_at' => 'datetime',
        ];
    }

    protected function artist(): Attribute
    {
        return Attribute::make(
            get: fn (mixed $value, array $attributes): ?string => app(ArtistCreditParser::class)
                ->resolveFromAttributes($attributes + ['artist' => $value])['artist'],
        );
    }

    protected function artistCredits(): Attribute
    {
        return Attribute::make(
            get: fn (mixed $value, array $attributes): array => app(ArtistCreditParser::class)
                ->resolveFromAttributes($attributes + ['artist_credits' => $value])['artist_credits'],
        );
    }

    protected function artistSlugs(): Attribute
    {
        return Attribute::make(
            get: fn (mixed $value, array $attributes): array => app(ArtistCreditParser::class)
                ->resolveFromAttributes($attributes + ['artist_slugs' => $value])['artist_slugs'],
        );
    }

    /**
     * Get the fragments for the track.
     *
     * @return HasMany<Fragment>
     */
    public function fragments(): HasMany
    {
        return $this->hasMany(Fragment::class);
    }

    /**
     * Get all per-user likes for the track.
     *
     * @return HasMany<UserTrackLike>
     */
    public function trackLikes(): HasMany
    {
        return $this->hasMany(UserTrackLike::class);
    }

    /**
     * Get current user's like state for the track.
     *
     * @return HasOne<UserTrackLike>
     */
    public function trackLike(): HasOne
    {
        $relation = $this->hasOne(UserTrackLike::class);
        $userId = Auth::id();

        return $userId === null
            ? $relation->whereRaw('1 = 0')
            : $relation->where('user_id', $userId);
    }

    /**
     * Get the rating for the track.
     *
     * @return HasOne<TrackRating>
     */
    public function rating(): HasOne
    {
        $relation = $this->hasOne(TrackRating::class);
        $userId = Auth::id();

        return $userId === null
            ? $relation->whereNull('user_id')
            : $relation->where('user_id', $userId);
    }

    /**
     * Get all ratings for the track.
     *
     * @return HasMany<TrackRating>
     */
    public function ratings(): HasMany
    {
        return $this->hasMany(TrackRating::class);
    }

    /**
     * Get the rating history for the track.
     *
     * @return HasMany<RatingHistory>
     */
    public function ratingHistory(): HasMany
    {
        return $this->hasMany(RatingHistory::class);
    }

    /**
     * Get the user preferences for the track.
     *
     * @return HasOne<UserTrackPreference>
     */
    public function userPreference(): HasOne
    {
        $relation = $this->hasOne(UserTrackPreference::class);
        $userId = Auth::id();

        return $userId === null
            ? $relation->whereNull('user_id')
            : $relation->where('user_id', $userId);
    }

    /**
     * Get all per-user preferences for the track.
     *
     * @return HasMany<UserTrackPreference>
     */
    public function userPreferences(): HasMany
    {
        return $this->hasMany(UserTrackPreference::class);
    }

    /**
     * Get the listening history for the track.
     *
     * @return HasMany<ListeningHistory>
     */
    public function listeningHistory(): HasMany
    {
        return $this->hasMany(ListeningHistory::class);
    }

    /**
     * Get the skip events for the track.
     *
     * @return HasMany<TrackSkip>
     */
    public function skips(): HasMany
    {
        return $this->hasMany(TrackSkip::class);
    }

    /**
     * Get the radio feedback for the track.
     *
     * @return HasMany<RadioFeedback>
     */
    public function radioFeedback(): HasMany
    {
        return $this->hasMany(RadioFeedback::class);
    }

    /**
     * Get the DJ analysis for the track.
     *
     * @return HasOne<TrackDjAnalysis>
     */
    public function djAnalysis(): HasOne
    {
        return $this->hasOne(TrackDjAnalysis::class);
    }

    /**
     * Get the pattern examples (segments) for the track.
     *
     * @return HasMany<PatternExample>
     */
    public function patternExamples(): HasMany
    {
        return $this->hasMany(PatternExample::class);
    }

    /**
     * Get the playlists that contain this track.
     *
     * @return BelongsToMany<Playlist>
     */
    public function playlists(): BelongsToMany
    {
        return $this->belongsToMany(Playlist::class, 'playlist_tracks')
            ->withPivot(['position', 'added_at']);
    }

    /**
     * Scope a query to only include liked tracks.
     *
     * @param Builder<Track> $query
     * @return Builder<Track>
     */
    public function scopeLiked(Builder $query): Builder
    {
        return $query->where('is_liked', true);
    }

    /**
     * Scope a query to current user's liked tracks.
     *
     * @param Builder<Track> $query
     * @param int|null $userId
     * @return Builder<Track>
     */
    public function scopeLikedByUser(Builder $query, ?int $userId): Builder
    {
        return $userId === null
            ? $query->whereRaw('1 = 0')
            : $query->whereHas('trackLikes', function (Builder $likeQuery) use ($userId) {
                $likeQuery->where('user_id', $userId);
            });
    }

    /**
     * Scope a query to only include analyzed tracks.
     *
     * @param Builder<Track> $query
     * @return Builder<Track>
     */
    public function scopeAnalyzed(Builder $query): Builder
    {
        return $query->where('analysis_status', 'completed');
    }

    /**
     * Scope a query to filter by genre.
     *
     * @param Builder<Track> $query
     * @param string $genre
     * @return Builder<Track>
     */
    public function scopeByGenre(Builder $query, string $genre): Builder
    {
        return $query->where('genre', $genre);
    }

    /**
     * Scope a query to filter by mood.
     *
     * @param Builder<Track> $query
     * @param string $mood
     * @return Builder<Track>
     */
    public function scopeByMood(Builder $query, string $mood): Builder
    {
        return $query->where('mood', $mood);
    }

    /**
     * Scope a query to filter by BPM range.
     *
     * @param Builder<Track> $query
     * @param float $min
     * @param float $max
     * @return Builder<Track>
     */
    public function scopeBpmRange(Builder $query, float $min, float $max): Builder
    {
        return $query->whereBetween('bpm', [$min, $max]);
    }

    public function getIsLikedAttribute($value): bool
    {
        $userId = Auth::id();

        if ($userId === null) {
            return (bool) $value;
        }

        if ($this->relationLoaded('trackLike')) {
            return $this->getRelation('trackLike') !== null;
        }

        if (!$this->exists) {
            return false;
        }

        return $this->trackLikes()
            ->where('user_id', $userId)
            ->exists();
    }
}
