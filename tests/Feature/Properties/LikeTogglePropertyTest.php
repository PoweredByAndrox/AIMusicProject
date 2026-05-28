<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use App\Models\Fragment;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: Like Toggle Idempotence
 * 
 * @tag Feature: ai-music-recommendations, Property 6: Like Toggle Idempotence
 * **Validates: Requirements 7.1, 7.2, 7.3**
 * 
 * For any track or fragment, double application of like operation
 * (like → unlike → like) should return the object to is_liked=true state,
 * and double unlike (unlike → like → unlike) should return to is_liked=false state.
 */
class LikeTogglePropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    /**
     * @test
     */
    public function track_like_unlike_like_returns_to_liked(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\string())
            ->then(function ($title) {
                $title = empty($title) ? 'Test Track' : substr($title, 0, 255);
                
                $track = Track::create([
                    'title' => $title,
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'is_liked' => false,
                ]);

                // like → unlike → like
                $track->is_liked = true;
                $track->save();
                
                $track->is_liked = false;
                $track->save();
                
                $track->is_liked = true;
                $track->save();

                $retrieved = Track::find($track->id);
                $this->assertTrue($retrieved->is_liked);
            });
    }

    /**
     * @test
     */
    public function track_unlike_like_unlike_returns_to_unliked(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\string())
            ->then(function ($title) {
                $title = empty($title) ? 'Test Track' : substr($title, 0, 255);
                
                $track = Track::create([
                    'title' => $title,
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'is_liked' => true,
                ]);

                // unlike → like → unlike
                $track->is_liked = false;
                $track->save();
                
                $track->is_liked = true;
                $track->save();
                
                $track->is_liked = false;
                $track->save();

                $retrieved = Track::find($track->id);
                $this->assertFalse($retrieved->is_liked);
            });
    }

    /**
     * @test
     */
    public function fragment_like_toggle_is_idempotent(): void
    {
        $track = Track::create([
            'title' => 'Test Track',
            'duration' => 300.0,
            'file_path' => '/storage/test.mp3',
            'source' => 'upload',
        ]);

        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(0, 100),  // start_time
                Generator\choose(10, 50)   // duration
            )
            ->then(function ($startTime, $duration) use ($track) {
                $endTime = $startTime + $duration;
                
                $fragment = Fragment::create([
                    'track_id' => $track->id,
                    'start_time' => (float) $startTime,
                    'end_time' => (float) $endTime,
                    'is_liked' => false,
                ]);

                // like → unlike → like
                $fragment->is_liked = true;
                $fragment->save();
                
                $fragment->is_liked = false;
                $fragment->save();
                
                $fragment->is_liked = true;
                $fragment->save();

                $retrieved = Fragment::find($fragment->id);
                $this->assertTrue($retrieved->is_liked);
            });
    }

    /**
     * @test
     */
    public function multiple_like_operations_are_idempotent(): void
    {
        $this
            ->limitTo(100)
            ->forAll(
                Generator\choose(1, 10)  // number of like operations
            )
            ->then(function ($likeCount) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'is_liked' => false,
                ]);

                // Apply like operation multiple times
                for ($i = 0; $i < $likeCount; $i++) {
                    $track->is_liked = true;
                    $track->save();
                }

                $retrieved = Track::find($track->id);
                $this->assertTrue($retrieved->is_liked);
            });
    }

    /**
     * @test
     */
    public function like_state_persists_after_retrieval(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\bool())
            ->then(function ($initialLiked) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 180.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                    'is_liked' => $initialLiked,
                ]);

                // Retrieve and verify state persists
                $retrieved = Track::find($track->id);
                $this->assertEquals($initialLiked, $retrieved->is_liked);
            });
    }
}
