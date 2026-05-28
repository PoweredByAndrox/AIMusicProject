<?php

namespace App\Http\Controllers;

use App\Models\Fragment;
use App\Models\Track;
use App\Services\PreferenceTrainingService;
use App\Services\TrackFeedbackSnapshotService;
use App\Services\TrackRatingService;
use Illuminate\Http\JsonResponse;

/**
 * LikeController
 *
 * Handles liking/unliking tracks and fragments.
 * Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
 */
class LikeController extends Controller
{
    public function __construct(
        private readonly TrackRatingService $trackRatingService,
        private readonly TrackFeedbackSnapshotService $trackFeedbackSnapshots,
    ) {}

    /**
     * Like a track.
     *
     * Requirements: 7.1
     *
     * @param int $trackId
     * @return JsonResponse
     */
    public function likeTrack(int $trackId): JsonResponse
    {
        $userId = (int) request()->user()?->id;
        $track = Track::find($trackId);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $this->trackRatingService->setRating($userId, $track->id, 5);
        $feedback = $this->trackFeedbackSnapshots->buildForUser($userId, $track->fresh());

        return response()->json([
            'message' => 'Track liked',
            'track_id' => $trackId,
            'is_liked' => true,
            'effective_rating' => $feedback['track_rating'] ?? 5,
            'explicit_track_rating' => $feedback['explicit_track_rating'] ?? 5,
            'computed_track_rating' => $feedback['computed_track_rating'] ?? null,
            'feedback' => $feedback,
            'stats' => PreferenceTrainingService::getStats($userId),
        ]);
    }

    /**
     * Unlike a track.
     *
     * Requirements: 7.3
     *
     * @param int $trackId
     * @return JsonResponse
     */
    public function unlikeTrack(int $trackId): JsonResponse
    {
        $userId = (int) request()->user()?->id;
        $track = Track::find($trackId);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $this->trackRatingService->removeRating($userId, $track->id);
        $feedback = $this->trackFeedbackSnapshots->buildForUser($userId, $track->fresh());

        return response()->json([
            'message' => 'Track unliked',
            'track_id' => $trackId,
            'is_liked' => false,
            'effective_rating' => $feedback['track_rating'] ?? null,
            'explicit_track_rating' => $feedback['explicit_track_rating'] ?? null,
            'computed_track_rating' => $feedback['computed_track_rating'] ?? null,
            'feedback' => $feedback,
        ]);
    }

    /**
     * Like a fragment.
     *
     * Requirements: 7.2
     *
     * @param int $fragmentId
     * @return JsonResponse
     */
    public function likeFragment(int $fragmentId): JsonResponse
    {
        $userId = request()->user()?->id;
        $fragment = Fragment::forUser($userId)->find($fragmentId);

        if (!$fragment) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        }

        $fragment->update([
            'is_liked' => true,
            'rated_at' => now(),
        ]);

        return response()->json([
            'message' => 'Fragment liked',
            'fragment_id' => $fragmentId,
            'is_liked' => true,
            'stats' => PreferenceTrainingService::getStats($userId),
        ]);
    }

    /**
     * Unlike a fragment.
     *
     * Requirements: 7.3
     *
     * @param int $fragmentId
     * @return JsonResponse
     */
    public function unlikeFragment(int $fragmentId): JsonResponse
    {
        $userId = request()->user()?->id;
        $fragment = Fragment::forUser($userId)->find($fragmentId);

        if (!$fragment) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        }

        $fragment->update([
            'is_liked' => false,
            'rated_at' => now(),
        ]);

        return response()->json([
            'message' => 'Fragment unliked',
            'fragment_id' => $fragmentId,
            'is_liked' => false,
        ]);
    }

    /**
     * Get all liked tracks and fragments.
     *
     * Requirements: 7.4
     *
     * @return JsonResponse
     */
    public function index(): JsonResponse
    {
        $userId = request()->user()?->id;
        $likedTracks = Track::likedByUser($userId)->get();
        $likedFragments = Fragment::forUser($userId)->liked()->with('track')->get();

        return response()->json([
            'tracks' => $likedTracks,
            'fragments' => $likedFragments,
            'total_likes' => $likedTracks->count() + $likedFragments->count(),
            'stats' => PreferenceTrainingService::getStats($userId),
        ]);
    }
}
