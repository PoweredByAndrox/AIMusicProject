<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Services\TrackFeedbackSnapshotService;
use App\Services\TrackRatingService;
use App\Services\TrackRatingPropagationService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use InvalidArgumentException;

/**
 * TrackRatingController
 * 
 * Handles REST API endpoints for 5-star track rating management.
 * Requirements: 11.1, 11.3, 14.3
 */
class TrackRatingController extends Controller
{
    public function __construct(
        private readonly TrackRatingService $trackRatingService,
        private readonly TrackRatingPropagationService $ratingPropagation,
        private readonly TrackFeedbackSnapshotService $trackFeedbackSnapshots,
    ) {}

    /**
     * Set or update the rating for a track.
     * 
     * POST /api/tracks/{id}/rating
     * Requirement 11.1: WHEN пользователь устанавливает рейтинг трека THEN THE 
     * Track_Rating SHALL сохранить значение от 1 до 5 звёзд
     *
     * @param Request $request
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function store(Request $request, int $id): JsonResponse
    {
        $rating = $request->input('rating');
        $sentiment = $request->input('sentiment');

        if (($rating === null || !is_numeric($rating)) && is_string($sentiment)) {
            $rating = match (strtolower(trim($sentiment))) {
                'dislike' => 1,
                'like', 'love', 'super_like' => 5,
                default => null,
            };
        }

        if ($rating === null || !is_numeric($rating)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Rating is required and must be a number or supported sentiment',
            ], 400);
        }

        $rating = (int) $rating;

        try {
            $userId = (int) $request->user()->id;
            $trackRating = $this->trackRatingService->setRating($userId, $id, $rating);
            $feedback = $this->trackFeedbackSnapshots->buildForUser($userId, Track::findOrFail($id));

            return response()->json([
                'message' => 'Rating set successfully',
                'rating' => $trackRating,
                'effective_rating' => $feedback['track_rating'] ?? $this->trackRatingService->getRating($userId, $id),
                'explicit_track_rating' => $feedback['explicit_track_rating'] ?? null,
                'computed_track_rating' => $feedback['computed_track_rating'] ?? null,
                'sentiment' => $feedback['sentiment'] ?? $this->ratingPropagation->sentimentForRating((float) $trackRating->rating),
                'feedback' => $feedback,
            ], 200);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to set rating: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get the current rating for a track.
     * 
     * GET /api/tracks/{id}/rating
     *
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function show(Request $request, int $id): JsonResponse
    {
        try {
            $rating = $this->trackRatingService->getRating((int) $request->user()->id, $id);
            $explicitRating = $this->trackRatingService->getExplicitRating((int) $request->user()->id, $id);
            $computedSegmentRating = $this->ratingPropagation->computeWeightedSegmentRating((int) $request->user()->id, $id);

            return response()->json([
                'track_id' => $id,
                'rating' => $rating,
                'explicit_rating' => $explicitRating,
                'computed_segment_rating' => $computedSegmentRating,
                'sentiment' => $this->ratingPropagation->sentimentForRating(
                    $rating !== null ? (float) $rating : $computedSegmentRating
                ),
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to get rating: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Remove the rating for a track.
     * 
     * DELETE /api/tracks/{id}/rating
     * Requirement 11.3: WHEN пользователь удаляет рейтинг THEN THE Track_Rating 
     * SHALL установить значение в NULL
     *
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function destroy(Request $request, int $id): JsonResponse
    {
        try {
            $userId = (int) $request->user()->id;
            $this->trackRatingService->removeRating($userId, $id);
            $feedback = $this->trackFeedbackSnapshots->buildForUser($userId, Track::findOrFail($id));

            return response()->json([
                'message' => 'Rating removed successfully',
                'effective_rating' => $feedback['track_rating'] ?? null,
                'explicit_track_rating' => $feedback['explicit_track_rating'] ?? null,
                'computed_track_rating' => $feedback['computed_track_rating'] ?? null,
                'sentiment' => $feedback['sentiment'] ?? null,
                'feedback' => $feedback,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to remove rating: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get the rating history for a track.
     * 
     * GET /api/tracks/{id}/rating/history
     * Requirement 14.3: THE System SHALL предоставлять страницу просмотра 
     * Rating_History с фильтрацией по дате и треку
     *
     * @param Request $request
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function history(Request $request, int $id): JsonResponse
    {
        try {
            $history = $this->trackRatingService->getHistory((int) $request->user()->id, $id);

            // Optional date filtering
            if ($request->has('from_date')) {
                $fromDate = $request->input('from_date');
                $history = $history->filter(function ($entry) use ($fromDate) {
                    return $entry->changed_at >= $fromDate;
                });
            }

            if ($request->has('to_date')) {
                $toDate = $request->input('to_date');
                $history = $history->filter(function ($entry) use ($toDate) {
                    return $entry->changed_at <= $toDate;
                });
            }

            return response()->json([
                'track_id' => $id,
                'history' => $history->values(),
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to get rating history: ' . $e->getMessage(),
            ], 500);
        }
    }
}
