<?php

namespace App\Services;

use App\Jobs\FragmentAnalysisJob;
use App\Models\Fragment;
use App\Models\Track;
use Illuminate\Database\Eloquent\Collection;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Auth;
use InvalidArgumentException;

/**
 * FragmentService
 * 
 * Handles fragment rating (like/dislike) and comment operations.
 * Implements FragmentRatingServiceInterface from design document.
 * Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.2, 8.3, 8.4
 */
class FragmentService
{
    /**
     * Maximum allowed comment length.
     * Requirement 8.2: THE Fragment_Comment SHALL поддерживать текст до 500 символов
     */
    private const MAX_COMMENT_LENGTH = 500;

    /**
     * Create a new fragment with a rating (like/dislike) and optional comment.
     * 
     * Requirement 7.1: WHEN пользователь выделяет область на waveform и ставит лайк 
     * THEN THE Fragment_Rating SHALL создать фрагмент с is_liked=true
     * 
     * Requirement 7.2: WHEN пользователь выделяет область на waveform и ставит дизлайк 
     * THEN THE Fragment_Rating SHALL создать фрагмент с is_liked=false
     * 
     * Requirement 7.7: WHEN фрагмент оценён THEN THE System SHALL поставить его 
     * в очередь на ML-анализ эмбеддинга
     *
     * @param int $trackId The ID of the track
     * @param float $startTime Start time of the fragment in seconds
     * @param float $endTime End time of the fragment in seconds
     * @param bool $isLiked True for like, false for dislike
     * @param string|null $comment Optional comment explaining the rating
     * @return Fragment The created fragment
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     * @throws InvalidArgumentException If time range is invalid or comment too long
     */
    public function rateFragment(int $trackId, float $startTime, float $endTime, bool $isLiked, ?string $comment = null): Fragment
    {
        // Verify track exists
        $track = Track::findOrFail($trackId);

        // Validate time range
        $this->validateTimeRange($startTime, $endTime, $track->duration);

        // Validate comment if provided
        if ($comment !== null && $comment !== '') {
            $this->validateCommentLength($comment);
        }

        return DB::transaction(function () use ($track, $startTime, $endTime, $isLiked, $comment) {
            // Create the fragment with rating and optional comment
            $fragment = Fragment::create([
                'track_id' => $track->id,
                'user_id' => Auth::id(),
                'start_time' => $startTime,
                'end_time' => $endTime,
                'is_liked' => $isLiked,
                'comment' => $comment ?: null,
                'rated_at' => now(),
                'analysis_status' => 'pending',
            ]);

            // Queue for ML analysis (Requirement 7.7)
            FragmentAnalysisJob::dispatch($fragment);

            return $fragment;
        });
    }

    /**
     * Update the rating of an existing fragment.
     * 
     * Requirement 7.3: WHEN пользователь изменяет оценку фрагмента 
     * THEN THE Fragment_Rating SHALL обновить значение is_liked
     *
     * @param int $fragmentId The ID of the fragment
     * @param bool $isLiked True for like, false for dislike
     * @return Fragment The updated fragment
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If fragment not found
     */
    public function updateRating(int $fragmentId, bool $isLiked): Fragment
    {
        $fragment = Fragment::forUser(Auth::id())->findOrFail($fragmentId);

        $fragment->update([
            'is_liked' => $isLiked,
            'rated_at' => now(),
        ]);

        return $fragment->fresh();
    }

    /**
     * Remove the rating from a fragment (deletes the fragment).
     * 
     * Requirement 7.4: WHEN пользователь удаляет оценку фрагмента 
     * THEN THE Fragment_Rating SHALL удалить запись из базы данных
     *
     * @param int $fragmentId The ID of the fragment
     * @return void
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If fragment not found
     */
    public function removeRating(int $fragmentId): void
    {
        $fragment = Fragment::forUser(Auth::id())->findOrFail($fragmentId);
        $fragment->delete();
    }

    /**
     * Add a comment to an existing fragment.
     * 
     * Requirement 8.1: WHEN пользователь добавляет комментарий к фрагменту 
     * THEN THE Fragment_Comment SHALL сохранить текст с привязкой к fragment_id
     * 
     * Requirement 8.2: THE Fragment_Comment SHALL поддерживать текст до 500 символов
     *
     * @param int $fragmentId The ID of the fragment
     * @param string $comment The comment text (max 500 characters)
     * @return Fragment The updated fragment
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If fragment not found
     * @throws InvalidArgumentException If comment exceeds 500 characters
     */
    public function addComment(int $fragmentId, string $comment): Fragment
    {
        $fragment = Fragment::forUser(Auth::id())->findOrFail($fragmentId);

        // Validate comment length
        $this->validateCommentLength($comment);

        $fragment->update([
            'comment' => $comment,
        ]);

        return $fragment->fresh();
    }

    /**
     * Update the comment of an existing fragment.
     * 
     * Requirement 8.3: WHEN пользователь редактирует комментарий 
     * THEN THE Fragment_Comment SHALL обновить текст
     *
     * @param int $fragmentId The ID of the fragment
     * @param string $comment The new comment text (max 500 characters)
     * @return Fragment The updated fragment
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If fragment not found
     * @throws InvalidArgumentException If comment exceeds 500 characters
     */
    public function updateComment(int $fragmentId, string $comment): Fragment
    {
        $fragment = Fragment::forUser(Auth::id())->findOrFail($fragmentId);

        // Validate comment length
        $this->validateCommentLength($comment);

        $fragment->update([
            'comment' => $comment,
        ]);

        return $fragment->fresh();
    }

    /**
     * Remove the comment from a fragment (sets to null).
     * 
     * Requirement 8.4: WHEN пользователь удаляет комментарий 
     * THEN THE Fragment_Comment SHALL удалить запись
     * Note: We set comment to null rather than deleting the fragment
     *
     * @param int $fragmentId The ID of the fragment
     * @return Fragment The updated fragment
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If fragment not found
     */
    public function removeComment(int $fragmentId): Fragment
    {
        $fragment = Fragment::forUser(Auth::id())->findOrFail($fragmentId);

        $fragment->update([
            'comment' => null,
        ]);

        return $fragment->fresh();
    }

    /**
     * Get all fragments for a specific track.
     *
     * @param int $trackId The ID of the track
     * @return Collection<int, Fragment> Collection of fragments
     * @throws \Illuminate\Database\Eloquent\ModelNotFoundException If track not found
     */
    public function getFragmentsForTrack(int $trackId): Collection
    {
        // Verify track exists
        Track::findOrFail($trackId);

        return Fragment::forUser(Auth::id())
            ->where('track_id', $trackId)
            ->orderBy('start_time')
            ->get();
    }

    /**
     * Get all liked fragments.
     *
     * @return Collection<int, Fragment> Collection of liked fragments
     */
    public function getLikedFragments(): Collection
    {
        return Fragment::forUser(Auth::id())
            ->liked()
            ->with('track')
            ->orderBy('created_at', 'desc')
            ->get();
    }

    /**
     * Get all disliked fragments.
     *
     * @return Collection<int, Fragment> Collection of disliked fragments
     */
    public function getDislikedFragments(): Collection
    {
        return Fragment::forUser(Auth::id())
            ->disliked()
            ->with('track')
            ->orderBy('created_at', 'desc')
            ->get();
    }

    /**
     * Get fragments filtered by comment presence.
     * 
     * Requirement 8.6: THE System SHALL позволять фильтровать фрагменты 
     * по наличию комментария
     *
     * @param bool $hasComment True to get fragments with comments, false for without
     * @return Collection<int, Fragment> Collection of fragments
     */
    public function getFragmentsByCommentPresence(bool $hasComment): Collection
    {
        $query = Fragment::forUser(Auth::id())->with('track');

        if ($hasComment) {
            $query->withComment();
        } else {
            $query->withoutComment();
        }

        return $query->orderBy('created_at', 'desc')->get();
    }

    /**
     * Validate that the time range is valid.
     *
     * @param float $startTime Start time in seconds
     * @param float $endTime End time in seconds
     * @param float|null $trackDuration Track duration in seconds (optional)
     * @return void
     * @throws InvalidArgumentException If time range is invalid
     */
    private function validateTimeRange(float $startTime, float $endTime, ?float $trackDuration = null): void
    {
        if ($startTime < 0) {
            throw new InvalidArgumentException('Start time cannot be negative');
        }

        if ($endTime <= $startTime) {
            throw new InvalidArgumentException('End time must be greater than start time');
        }

        if ($trackDuration !== null && $endTime > $trackDuration) {
            throw new InvalidArgumentException(
                sprintf('End time (%.2f) cannot exceed track duration (%.2f)', $endTime, $trackDuration)
            );
        }
    }

    /**
     * Validate that the comment length is within limits.
     *
     * @param string $comment The comment text
     * @return void
     * @throws InvalidArgumentException If comment exceeds maximum length
     */
    private function validateCommentLength(string $comment): void
    {
        if (mb_strlen($comment) > self::MAX_COMMENT_LENGTH) {
            throw new InvalidArgumentException(
                sprintf('Comment cannot exceed %d characters', self::MAX_COMMENT_LENGTH)
            );
        }
    }
}
