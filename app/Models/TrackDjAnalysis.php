<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Builder;

/**
 * TrackDjAnalysis Model
 * 
 * Represents AI DJ analysis results for a track (Camelot key, BPM, energy, structure points).
 * Requirements: 27.5 - Save analysis results to database for quick access
 * 
 * @property int $id
 * @property int $track_id
 * @property string|null $camelot_key
 * @property float|null $bpm
 * @property int|null $energy_level
 * @property float|null $intro_point
 * @property float|null $outro_point
 * @property array $drop_points
 * @property array $breakdown_points
 * @property \Carbon\Carbon $analyzed_at
 */
class TrackDjAnalysis extends Model
{
    use HasFactory;

    /**
     * The table associated with the model.
     *
     * @var string
     */
    protected $table = 'track_dj_analysis';

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
        'track_id',
        'camelot_key',
        'bpm',
        'energy_level',
        'intro_point',
        'outro_point',
        'drop_points',
        'breakdown_points',
        'analyzed_at',
    ];

    /**
     * Get the attributes that should be cast.
     *
     * @return array<string, string>
     */
    protected function casts(): array
    {
        return [
            'track_id' => 'integer',
            'bpm' => 'float',
            'energy_level' => 'integer',
            'intro_point' => 'float',
            'outro_point' => 'float',
            'drop_points' => 'array',
            'breakdown_points' => 'array',
            'analyzed_at' => 'datetime',
        ];
    }

    /**
     * Get the track that owns the DJ analysis.
     *
     * @return BelongsTo<Track, TrackDjAnalysis>
     */
    public function track(): BelongsTo
    {
        return $this->belongsTo(Track::class);
    }

    /**
     * Scope a query to filter by Camelot key.
     *
     * @param Builder<TrackDjAnalysis> $query
     * @param string $key
     * @return Builder<TrackDjAnalysis>
     */
    public function scopeByCamelotKey(Builder $query, string $key): Builder
    {
        return $query->where('camelot_key', $key);
    }

    /**
     * Scope a query to filter by BPM range.
     *
     * @param Builder<TrackDjAnalysis> $query
     * @param float $min
     * @param float $max
     * @return Builder<TrackDjAnalysis>
     */
    public function scopeBpmRange(Builder $query, float $min, float $max): Builder
    {
        return $query->whereBetween('bpm', [$min, $max]);
    }

    /**
     * Scope a query to filter by energy level range.
     *
     * @param Builder<TrackDjAnalysis> $query
     * @param int $min
     * @param int $max
     * @return Builder<TrackDjAnalysis>
     */
    public function scopeEnergyRange(Builder $query, int $min, int $max): Builder
    {
        return $query->whereBetween('energy_level', [$min, $max]);
    }
}
