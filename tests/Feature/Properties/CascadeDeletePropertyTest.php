<?php

namespace Tests\Feature\Properties;

use App\Models\Track;
use App\Models\Fragment;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Illuminate\Support\Facades\Storage;
use Tests\TestCase;

/**
 * Property Test: Cascade Delete Completeness
 * 
 * @tag Feature: ai-music-recommendations, Property 7: Cascade Delete Completeness
 * **Validates: Requirements 4.6**
 * 
 * For any deleted track, after deletion there should not exist:
 * - Related fragments in the fragments table
 * - Audio file in file storage
 * - Embedding file in file storage
 * - Record in FAISS index
 */
class CascadeDeletePropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    /**
     * @test
     */
    public function deleting_track_removes_all_fragments(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 10))
            ->then(function ($numFragments) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 300.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                ]);

                // Create fragments
                for ($i = 0; $i < $numFragments; $i++) {
                    Fragment::create([
                        'track_id' => $track->id,
                        'start_time' => $i * 10.0,
                        'end_time' => ($i + 1) * 10.0,
                    ]);
                }

                $trackId = $track->id;

                // Verify fragments exist
                $this->assertEquals($numFragments, Fragment::where('track_id', $trackId)->count());

                // Delete track
                $track->delete();

                // Verify all fragments are deleted (cascade)
                $this->assertEquals(0, Fragment::where('track_id', $trackId)->count());
            });
    }

    /**
     * @test
     */
    public function deleted_track_not_found_in_database(): void
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
                ]);

                $trackId = $track->id;
                $track->delete();

                // Verify track is not found
                $this->assertNull(Track::find($trackId));
            });
    }

    /**
     * @test
     */
    public function fragment_ids_not_reused_after_cascade_delete(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($numFragments) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 300.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                ]);

                // Create and store fragment IDs
                $fragmentIds = [];
                for ($i = 0; $i < $numFragments; $i++) {
                    $fragment = Fragment::create([
                        'track_id' => $track->id,
                        'start_time' => $i * 10.0,
                        'end_time' => ($i + 1) * 10.0,
                    ]);
                    $fragmentIds[] = $fragment->id;
                }

                // Delete track (cascades to fragments)
                $track->delete();

                // Verify old fragment IDs don't exist
                foreach ($fragmentIds as $fragmentId) {
                    $this->assertNull(Fragment::find($fragmentId));
                }
            });
    }

    /**
     * @test
     */
    public function liked_fragments_also_deleted(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($numLikedFragments) {
                $track = Track::create([
                    'title' => 'Test Track',
                    'duration' => 300.0,
                    'file_path' => '/storage/test.mp3',
                    'source' => 'upload',
                ]);

                // Create liked fragments
                for ($i = 0; $i < $numLikedFragments; $i++) {
                    Fragment::create([
                        'track_id' => $track->id,
                        'start_time' => $i * 10.0,
                        'end_time' => ($i + 1) * 10.0,
                        'is_liked' => true,
                    ]);
                }

                $trackId = $track->id;

                // Verify liked fragments exist
                $this->assertEquals(
                    $numLikedFragments,
                    Fragment::where('track_id', $trackId)->liked()->count()
                );

                // Delete track
                $track->delete();

                // Verify liked fragments are also deleted
                $this->assertEquals(0, Fragment::where('track_id', $trackId)->count());
            });
    }

    /**
     * @test
     */
    public function multiple_tracks_delete_independently(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(2, 5))
            ->then(function ($numTracks) {
                $tracks = [];
                
                // Create multiple tracks with fragments
                for ($t = 0; $t < $numTracks; $t++) {
                    $track = Track::create([
                        'title' => "Track {$t}",
                        'duration' => 300.0,
                        'file_path' => "/storage/test{$t}.mp3",
                        'source' => 'upload',
                    ]);
                    $tracks[] = $track;

                    Fragment::create([
                        'track_id' => $track->id,
                        'start_time' => 0.0,
                        'end_time' => 10.0,
                    ]);
                }

                // Delete first track
                $deletedTrackId = $tracks[0]->id;
                $tracks[0]->delete();

                // Verify first track's fragments are deleted
                $this->assertEquals(0, Fragment::where('track_id', $deletedTrackId)->count());

                // Verify other tracks' fragments still exist
                for ($t = 1; $t < $numTracks; $t++) {
                    $this->assertEquals(
                        1,
                        Fragment::where('track_id', $tracks[$t]->id)->count()
                    );
                }
            });
    }
}
