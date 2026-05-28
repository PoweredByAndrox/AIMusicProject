<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Models\Fragment;
use App\Jobs\FragmentAnalysisJob;
use App\Services\FragmentService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Database\Eloquent\ModelNotFoundException;
use InvalidArgumentException;

/**
 * FragmentController
 * 
 * Handles fragment creation, listing, deletion, rating, and comments.
 * Requirements: 6.1, 6.2, 6.5, 6.6, 7.1, 7.2, 7.3, 7.4, 8.1, 8.3, 8.4
 */
class FragmentController extends Controller
{
    public function __construct(
        private readonly FragmentService $fragmentService
    ) {}
    /**
     * Store a new fragment.
     * 
     * Requirements: 6.1, 6.2, 6.6
     *
     * @param int $trackId
     * @param Request $request
     * @return JsonResponse
     */
    public function store(int $trackId, Request $request): JsonResponse
    {
        $track = Track::find($trackId);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        // Validate time range
        $startTime = (float) $request->input('start_time');
        $endTime = (float) $request->input('end_time');

        if ($startTime < 0) {
            return response()->json([
                'error' => 'Invalid start_time',
                'message' => 'start_time must be >= 0',
            ], 400);
        }

        if ($endTime <= $startTime) {
            return response()->json([
                'error' => 'Invalid time range',
                'message' => 'end_time must be greater than start_time',
            ], 400);
        }

        if ($endTime > $track->duration) {
            return response()->json([
                'error' => 'Invalid end_time',
                'message' => 'end_time exceeds track duration',
            ], 400);
        }

        $isLiked = $request->input('is_liked');
        $comment = $request->input('comment');

        try {
            if ($isLiked !== null) {
                $fragment = $this->fragmentService->rateFragment(
                    $trackId,
                    $startTime,
                    $endTime,
                    filter_var($isLiked, FILTER_VALIDATE_BOOLEAN),
                    is_string($comment) && $comment !== '' ? $comment : null
                );

                return response()->json([
                    'message' => 'Fragment created',
                    'fragment' => $fragment,
                ], 201);
            }

            $fragment = Fragment::create([
                'track_id' => $trackId,
                'user_id' => Auth::id(),
                'start_time' => $startTime,
                'end_time' => $endTime,
                'analysis_status' => 'pending',
            ]);

            // Queue for embedding extraction (Requirement 6.6)
            FragmentAnalysisJob::dispatch($fragment);

            return response()->json([
                'message' => 'Fragment created',
                'fragment' => $fragment,
            ], 201);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Creation failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * List fragments for a track.
     * 
     * Requirements: 6.3
     *
     * @param int|string $trackId
     * @return JsonResponse
     */
    public function index(int|string $trackId): JsonResponse
    {
        $trackId = (int) $trackId;
        $track = Track::find($trackId);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $fragments = Fragment::forUser(request()->user()?->id)
            ->where('track_id', $trackId)
            ->orderBy('start_time')
            ->get();

        return response()->json([
            'fragments' => $fragments,
            'total' => $fragments->count(),
        ]);
    }

    /**
     * Delete a fragment.
     * 
     * Requirements: 6.5
     *
     * @param int $trackId
     * @param int $fragmentId
     * @return JsonResponse
     */
    public function destroy(int $trackId, int $fragmentId): JsonResponse
    {
        $fragment = Fragment::forUser(request()->user()?->id)
            ->where('track_id', $trackId)
            ->where('id', $fragmentId)
            ->first();

        if (!$fragment) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        }

        $fragment->delete();

        return response()->json([
            'message' => 'Fragment deleted',
        ]);
    }

    /**
     * Create a new fragment with a rating (like/dislike) and optional comment.
     * 
     * POST /api/tracks/{id}/fragments/rate
     * 
     * Requirement 7.1: WHEN пользователь выделяет область на waveform и ставит лайк 
     * THEN THE Fragment_Rating SHALL создать фрагмент с is_liked=true
     * 
     * Requirement 7.2: WHEN пользователь выделяет область на waveform и ставит дизлайк 
     * THEN THE Fragment_Rating SHALL создать фрагмент с is_liked=false
     *
     * @param Request $request
     * @param int $id Track ID
     * @return JsonResponse
     */
    public function rate(Request $request, int $id): JsonResponse
    {
        // Validate required fields
        $startTime = $request->input('start_time');
        $endTime = $request->input('end_time');
        $isLiked = $request->input('is_liked');
        $comment = $request->input('comment');

        if ($startTime === null || !is_numeric($startTime)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'start_time is required and must be a number',
            ], 400);
        }

        if ($endTime === null || !is_numeric($endTime)) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'end_time is required and must be a number',
            ], 400);
        }

        if ($isLiked === null) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'is_liked is required (true for like, false for dislike)',
            ], 400);
        }

        try {
            $fragment = $this->fragmentService->rateFragment(
                $id,
                (float) $startTime,
                (float) $endTime,
                filter_var($isLiked, FILTER_VALIDATE_BOOLEAN),
                $comment ? (string) $comment : null
            );

            return response()->json([
                'message' => 'Fragment rated successfully',
                'fragment' => $fragment,
            ], 201);
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
                'message' => 'Failed to rate fragment: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Update the rating of an existing fragment.
     * 
     * PUT /api/fragments/{id}/rating
     * 
     * Requirement 7.3: WHEN пользователь изменяет оценку фрагмента 
     * THEN THE Fragment_Rating SHALL обновить значение is_liked
     *
     * @param Request $request
     * @param int $id Fragment ID
     * @return JsonResponse
     */
    public function updateRating(Request $request, int $id): JsonResponse
    {
        $isLiked = $request->input('is_liked');

        if ($isLiked === null) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'is_liked is required (true for like, false for dislike)',
            ], 400);
        }

        try {
            $fragment = $this->fragmentService->updateRating(
                $id,
                filter_var($isLiked, FILTER_VALIDATE_BOOLEAN)
            );

            return response()->json([
                'message' => 'Fragment rating updated successfully',
                'fragment' => $fragment,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to update fragment rating: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Remove the rating from a fragment (deletes the fragment).
     * 
     * DELETE /api/fragments/{id}/rating
     * 
     * Requirement 7.4: WHEN пользователь удаляет оценку фрагмента 
     * THEN THE Fragment_Rating SHALL удалить запись из базы данных
     *
     * @param int $id Fragment ID
     * @return JsonResponse
     */
    public function destroyRating(int $id): JsonResponse
    {
        try {
            $this->fragmentService->removeRating($id);

            return response()->json([
                'message' => 'Fragment rating removed successfully',
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to remove fragment rating: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Add a comment to an existing fragment.
     * 
     * POST /api/fragments/{id}/comment
     * 
     * Requirement 8.1: WHEN пользователь добавляет комментарий к фрагменту 
     * THEN THE Fragment_Comment SHALL сохранить текст с привязкой к fragment_id
     *
     * @param Request $request
     * @param int $id Fragment ID
     * @return JsonResponse
     */
    public function addComment(Request $request, int $id): JsonResponse
    {
        $comment = $request->input('comment');

        if ($comment === null || !is_string($comment) || trim($comment) === '') {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'comment is required and must be a non-empty string',
            ], 400);
        }

        try {
            $fragment = $this->fragmentService->addComment($id, $comment);

            return response()->json([
                'message' => 'Comment added successfully',
                'fragment' => $fragment,
            ], 201);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to add comment: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Update the comment of an existing fragment.
     * 
     * PUT /api/fragments/{id}/comment
     * 
     * Requirement 8.3: WHEN пользователь редактирует комментарий 
     * THEN THE Fragment_Comment SHALL обновить текст
     *
     * @param Request $request
     * @param int $id Fragment ID
     * @return JsonResponse
     */
    public function updateComment(Request $request, int $id): JsonResponse
    {
        $comment = $request->input('comment');

        if ($comment === null || !is_string($comment) || trim($comment) === '') {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'comment is required and must be a non-empty string',
            ], 400);
        }

        try {
            $fragment = $this->fragmentService->updateComment($id, $comment);

            return response()->json([
                'message' => 'Comment updated successfully',
                'fragment' => $fragment,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        } catch (InvalidArgumentException $e) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => $e->getMessage(),
            ], 400);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to update comment: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Remove the comment from a fragment.
     * 
     * DELETE /api/fragments/{id}/comment
     * 
     * Requirement 8.4: WHEN пользователь удаляет комментарий 
     * THEN THE Fragment_Comment SHALL удалить запись
     *
     * @param int $id Fragment ID
     * @return JsonResponse
     */
    public function destroyComment(int $id): JsonResponse
    {
        try {
            $fragment = $this->fragmentService->removeComment($id);

            return response()->json([
                'message' => 'Comment removed successfully',
                'fragment' => $fragment,
            ]);
        } catch (ModelNotFoundException $e) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Fragment not found',
            ], 404);
        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Operation failed',
                'message' => 'Failed to remove comment: ' . $e->getMessage(),
            ], 500);
        }
    }
}
