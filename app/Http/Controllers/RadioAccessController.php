<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Services\PersonalRadioAccessService;
use App\Services\PublicRadioStationService;
use App\Services\RadioAccessTokenService;
use App\Services\TrackFeedbackSnapshotService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Validation\Rule;

class RadioAccessController extends Controller
{
    public function __construct(
        private readonly RadioAccessTokenService $radioAccessTokens,
        private readonly PersonalRadioAccessService $personalRadioAccess,
        private readonly PublicRadioStationService $stations,
        private readonly TrackFeedbackSnapshotService $trackFeedbackSnapshots,
    ) {}

    public function issueAccessToken(Request $request): JsonResponse
    {
        return response()->json($this->radioAccessTokens->issue($request->user()));
    }

    public function personalSettings(Request $request): JsonResponse
    {
        return response()->json($this->personalRadioAccess->getSettingsForOwner($request->user()));
    }

    public function updatePersonalSettings(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'enabled' => ['required', 'boolean'],
            'access_mode' => ['nullable', Rule::in(['all', 'selected_users'])],
            'allowed_user_ids' => ['nullable', 'array'],
            'allowed_user_ids.*' => ['integer', 'exists:users,id'],
            'preferred_genres' => ['nullable', 'array'],
            'preferred_genres.*' => ['string'],
            'allowed_emails' => ['nullable'],
        ]);

        $allowedUserIds = $validated['allowed_user_ids'] ?? [];
        if (!is_array($allowedUserIds)) {
            $allowedUserIds = [];
        }

        $accessMode = is_string($validated['access_mode'] ?? null)
            ? (string) $validated['access_mode']
            : 'selected_users';

        if ($allowedUserIds === []) {
            $allowedEmails = $validated['allowed_emails'] ?? [];
            if (is_string($allowedEmails)) {
                $allowedEmails = preg_split('/[\r\n,;]+/', $allowedEmails) ?: [];
            }

            if (is_array($allowedEmails) && $allowedEmails !== []) {
                $allowedUserIds = \App\Models\User::query()
                    ->whereIn(\Illuminate\Support\Facades\DB::raw('LOWER(email)'), array_map(
                        static fn (string $email): string => mb_strtolower(trim($email)),
                        array_values(array_filter(array_map('strval', $allowedEmails))),
                    ))
                    ->pluck('id')
                    ->map(fn (mixed $userId): int => (int) $userId)
                    ->all();
            }
        }

        return response()->json(
            $this->personalRadioAccess->updateSettingsForOwner(
                $request->user(),
                (bool) $validated['enabled'],
                $accessMode,
                array_values(array_map('intval', $allowedUserIds)),
                is_array($validated['preferred_genres'] ?? null) ? $validated['preferred_genres'] : null,
            )
        );
    }

    public function updatePersonalStationGenres(Request $request, string $slug): JsonResponse
    {
        $owner = $this->personalRadioAccess->ownerForSlug($slug);
        if (!$owner) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Personal radio station not found',
            ], 404);
        }

        $viewer = $this->radioAccessTokens->resolveFromRequest($request);
        if (!$this->personalRadioAccess->canManageStation($owner, $viewer)) {
            return response()->json([
                'error' => 'Forbidden',
                'message' => 'Only the station owner can update personal radio genres',
            ], 403);
        }

        $validated = $request->validate([
            'preferred_genres' => ['nullable', 'array'],
            'preferred_genres.*' => ['string'],
        ]);

        $this->personalRadioAccess->updatePreferredGenresForOwner(
            $owner,
            is_array($validated['preferred_genres'] ?? null) ? $validated['preferred_genres'] : [],
        );
        $this->stations->refreshUpcomingQueue($slug);

        return response()->json($this->stations->snapshot($slug, $viewer));
    }

    public function privateOverlay(Request $request, Track $track): JsonResponse
    {
        $viewer = $this->radioAccessTokens->resolveFromRequest($request);
        if (!$viewer) {
            return response()->json([
                'error' => 'Unauthorized',
                'message' => 'Valid radio access token is required',
            ], 401);
        }

        return response()->json($this->trackFeedbackSnapshots->buildForUser((int) $viewer->id, $track));
    }
}
