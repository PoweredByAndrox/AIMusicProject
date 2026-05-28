<?php

namespace Tests\Feature\Properties;

use App\Models\Playlist;
use App\Models\PlaylistTrack;
use App\Models\Track;
use App\Services\PlaylistService;
use Eris\Generator;
use Eris\TestTrait;
use Illuminate\Foundation\Testing\DatabaseTransactions;
use Tests\TestCase;

/**
 * Property Test: Playlist Integrity
 * 
 * @tag Feature: advanced-music-player, Property 6: Playlist Integrity
 * **Validates: Requirements 5.2, 5.3, 5.4, 5.6**
 * 
 * For any playlist and set of tracks:
 * - Adding a track increases track_count by 1
 * - Removing a track from playlist does NOT delete the track from library
 * - Deleting a playlist does NOT delete tracks from library
 * - Track order is preserved after reorder operation
 */
class PlaylistIntegrityPropertyTest extends TestCase
{
    use TestTrait;
    use DatabaseTransactions;

    private PlaylistService $playlistService;

    protected function setUp(): void
    {
        parent::setUp();
        $this->playlistService = new PlaylistService();
    }

    /**
     * Helper to create a track with random data.
     */
    private function createTrack(string $title = 'Test Track'): Track
    {
        return Track::create([
            'title' => $title,
            'duration' => rand(60, 600),
            'file_path' => '/storage/test_' . uniqid() . '.mp3',
            'source' => 'upload',
        ]);
    }

    /**
     * @test
     * 
     * Property: Adding a track increases track_count by exactly 1
     * Validates: Requirement 5.2
     */
    public function adding_track_increases_track_count_by_one(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 10))
            ->then(function ($numTracksToAdd) {
                // Create a playlist
                $playlist = $this->playlistService->create('Test Playlist', 'Test Description');
                
                $this->assertEquals(0, $playlist->track_count);

                // Add tracks one by one and verify count increases by 1 each time
                for ($i = 0; $i < $numTracksToAdd; $i++) {
                    $track = $this->createTrack("Track {$i}");
                    $previousCount = $playlist->fresh()->track_count;
                    
                    $this->playlistService->addTrack($playlist->id, $track->id);
                    
                    $playlist->refresh();
                    $this->assertEquals(
                        $previousCount + 1,
                        $playlist->track_count,
                        "Adding track should increase track_count by exactly 1"
                    );
                }

                // Final verification
                $this->assertEquals($numTracksToAdd, $playlist->track_count);
            });
    }

    /**
     * @test
     * 
     * Property: Removing track from playlist leaves track in library (Track::find still works)
     * Validates: Requirement 5.3
     */
    public function removing_track_from_playlist_leaves_track_in_library(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($numTracks) {
                // Create playlist and tracks
                $playlist = $this->playlistService->create('Test Playlist');
                $tracks = [];
                
                for ($i = 0; $i < $numTracks; $i++) {
                    $track = $this->createTrack("Track {$i}");
                    $tracks[] = $track;
                    $this->playlistService->addTrack($playlist->id, $track->id);
                }

                // Remove each track from playlist and verify it still exists in library
                foreach ($tracks as $track) {
                    $trackId = $track->id;
                    
                    // Verify track is in playlist
                    $this->assertTrue(
                        PlaylistTrack::where('playlist_id', $playlist->id)
                            ->where('track_id', $trackId)
                            ->exists()
                    );
                    
                    // Remove track from playlist
                    $this->playlistService->removeTrack($playlist->id, $trackId);
                    
                    // Verify track is NOT in playlist anymore
                    $this->assertFalse(
                        PlaylistTrack::where('playlist_id', $playlist->id)
                            ->where('track_id', $trackId)
                            ->exists()
                    );
                    
                    // Verify track STILL EXISTS in library (Track::find works)
                    $foundTrack = Track::find($trackId);
                    $this->assertNotNull(
                        $foundTrack,
                        "Track should still exist in library after removal from playlist"
                    );
                    $this->assertEquals($trackId, $foundTrack->id);
                }
            });
    }

    /**
     * @test
     * 
     * Property: Deleting playlist leaves all tracks in library
     * Validates: Requirement 5.4
     */
    public function deleting_playlist_leaves_all_tracks_in_library(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 10))
            ->then(function ($numTracks) {
                // Create playlist and tracks
                $playlist = $this->playlistService->create('Test Playlist');
                $trackIds = [];
                
                for ($i = 0; $i < $numTracks; $i++) {
                    $track = $this->createTrack("Track {$i}");
                    $trackIds[] = $track->id;
                    $this->playlistService->addTrack($playlist->id, $track->id);
                }

                $playlistId = $playlist->id;

                // Verify all tracks are in the playlist
                $this->assertEquals(
                    $numTracks,
                    PlaylistTrack::where('playlist_id', $playlistId)->count()
                );

                // Delete the playlist
                $this->playlistService->delete($playlistId);

                // Verify playlist is deleted
                $this->assertNull(Playlist::find($playlistId));

                // Verify playlist_tracks are deleted (cascade)
                $this->assertEquals(
                    0,
                    PlaylistTrack::where('playlist_id', $playlistId)->count()
                );

                // Verify ALL tracks still exist in library
                foreach ($trackIds as $trackId) {
                    $foundTrack = Track::find($trackId);
                    $this->assertNotNull(
                        $foundTrack,
                        "Track {$trackId} should still exist in library after playlist deletion"
                    );
                }
            });
    }

    /**
     * @test
     * 
     * Property: Reorder operation preserves all tracks (same count, same IDs)
     * Validates: Requirement 5.6
     */
    public function reorder_operation_preserves_all_tracks(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(2, 8))
            ->then(function ($numTracks) {
                // Create playlist and tracks
                $playlist = $this->playlistService->create('Test Playlist');
                $trackIds = [];
                
                for ($i = 0; $i < $numTracks; $i++) {
                    $track = $this->createTrack("Track {$i}");
                    $trackIds[] = $track->id;
                    $this->playlistService->addTrack($playlist->id, $track->id);
                }

                // Get original track IDs in order
                $originalTrackIds = PlaylistTrack::where('playlist_id', $playlist->id)
                    ->orderBy('position')
                    ->pluck('track_id')
                    ->toArray();

                // Create a shuffled order (reverse for deterministic test)
                $newOrder = array_reverse($trackIds);

                // Perform reorder
                $this->playlistService->reorderTracks($playlist->id, $newOrder);

                // Get track IDs after reorder
                $reorderedTrackIds = PlaylistTrack::where('playlist_id', $playlist->id)
                    ->orderBy('position')
                    ->pluck('track_id')
                    ->toArray();

                // Verify same count
                $this->assertCount(
                    count($originalTrackIds),
                    $reorderedTrackIds,
                    "Reorder should preserve track count"
                );

                // Verify same IDs (just in different order)
                $sortedOriginal = $originalTrackIds;
                $sortedReordered = $reorderedTrackIds;
                sort($sortedOriginal);
                sort($sortedReordered);
                
                $this->assertEquals(
                    $sortedOriginal,
                    $sortedReordered,
                    "Reorder should preserve all track IDs"
                );

                // Verify the new order matches what we requested
                $this->assertEquals(
                    $newOrder,
                    $reorderedTrackIds,
                    "Reorder should apply the requested order"
                );
            });
    }

    /**
     * @test
     * 
     * Property: Track count decreases by 1 when removing a track
     * Validates: Requirement 5.3 (inverse of 5.2)
     */
    public function removing_track_decreases_track_count_by_one(): void
    {
        $this
            ->limitTo(100)
            ->forAll(Generator\choose(1, 5))
            ->then(function ($numTracks) {
                // Create playlist with tracks
                $playlist = $this->playlistService->create('Test Playlist');
                $trackIds = [];
                
                for ($i = 0; $i < $numTracks; $i++) {
                    $track = $this->createTrack("Track {$i}");
                    $trackIds[] = $track->id;
                    $this->playlistService->addTrack($playlist->id, $track->id);
                }

                $playlist->refresh();
                $this->assertEquals($numTracks, $playlist->track_count);

                // Remove tracks one by one and verify count decreases by 1 each time
                foreach ($trackIds as $trackId) {
                    $previousCount = $playlist->fresh()->track_count;
                    
                    $this->playlistService->removeTrack($playlist->id, $trackId);
                    
                    $playlist->refresh();
                    $this->assertEquals(
                        $previousCount - 1,
                        $playlist->track_count,
                        "Removing track should decrease track_count by exactly 1"
                    );
                }

                // Final verification
                $this->assertEquals(0, $playlist->track_count);
            });
    }

    /**
     * @test
     * 
     * Property: Multiple playlists can contain the same track independently
     * Validates: Requirements 5.2, 5.3, 5.4
     */
    public function multiple_playlists_share_tracks_independently(): void
    {
        $this
            ->limitTo(50)
            ->forAll(Generator\choose(2, 4))
            ->then(function ($numPlaylists) {
                // Create a shared track
                $sharedTrack = $this->createTrack('Shared Track');
                $trackId = $sharedTrack->id;
                
                // Create multiple playlists and add the same track to each
                $playlistIds = [];
                for ($i = 0; $i < $numPlaylists; $i++) {
                    $playlist = $this->playlistService->create("Playlist {$i}");
                    $playlistIds[] = $playlist->id;
                    $this->playlistService->addTrack($playlist->id, $trackId);
                }

                // Verify track is in all playlists
                foreach ($playlistIds as $playlistId) {
                    $this->assertTrue(
                        PlaylistTrack::where('playlist_id', $playlistId)
                            ->where('track_id', $trackId)
                            ->exists()
                    );
                }

                // Delete first playlist
                $this->playlistService->delete($playlistIds[0]);

                // Verify track still exists in library
                $this->assertNotNull(Track::find($trackId));

                // Verify track is still in other playlists
                for ($i = 1; $i < $numPlaylists; $i++) {
                    $this->assertTrue(
                        PlaylistTrack::where('playlist_id', $playlistIds[$i])
                            ->where('track_id', $trackId)
                            ->exists(),
                        "Track should still be in playlist {$i} after deleting playlist 0"
                    );
                }
            });
    }
}
