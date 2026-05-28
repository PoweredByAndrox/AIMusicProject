<?php

namespace Database\Factories;

use App\Models\Track;
use Illuminate\Database\Eloquent\Factories\Factory;

/**
 * @extends Factory<Track>
 */
class TrackFactory extends Factory
{
    protected $model = Track::class;

    public function definition(): array
    {
        return [
            'title' => $this->faker->sentence(3),
            'artist' => $this->faker->name(),
            'album' => $this->faker->sentence(2),
            'year' => $this->faker->numberBetween(1990, 2025),
            'duration' => $this->faker->numberBetween(120, 360),
            'file_path' => 'audio/' . $this->faker->uuid() . '.mp3',
            'source' => $this->faker->randomElement(['upload', 'youtube']),
            'genre' => $this->faker->randomElement(['rock', 'pop', 'electronic', 'jazz', 'classical']),
            'bpm' => $this->faker->numberBetween(80, 180),
            'key' => $this->faker->randomElement(['C', 'D', 'E', 'F', 'G', 'A', 'B']),
            'mode' => $this->faker->randomElement(['major', 'minor']),
            'mood' => $this->faker->randomElement(['energetic', 'calm', 'happy', 'sad', 'aggressive']),
            'has_vocals' => $this->faker->boolean(),
            'is_acoustic' => $this->faker->boolean(),
            'analysis_status' => 'pending',
            'is_liked' => false,
            'manually_edited' => [],
        ];
    }

    public function completed(): static
    {
        return $this->state(fn (array $attributes) => [
            'analysis_status' => 'completed',
        ]);
    }

    public function liked(): static
    {
        return $this->state(fn (array $attributes) => [
            'is_liked' => true,
        ]);
    }
}
