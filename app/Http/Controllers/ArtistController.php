<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Services\ArtistCreditParser;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class ArtistController extends Controller
{
    public function __construct(
        private readonly ArtistCreditParser $artistCreditParser,
    ) {}

    public function show(Request $request, string $slug): JsonResponse
    {
        $query = Track::query()
            ->whereJsonContains('artist_slugs', $slug)
            ->orderBy('created_at', 'desc');
        $userId = $request->user()?->id;

        if ($userId !== null) {
            $query->with('trackLike');
        }

        $primaryTracks = [];
        $featuredTracks = [];
        $remixTracks = [];
        $artistName = null;

        foreach ($query->get() as $track) {
            $artistCredits = is_array($track->artist_credits) ? $track->artist_credits : [];
            $roles = $this->artistCreditParser->rolesForSlug($artistCredits, $slug);

            if ($roles === []) {
                continue;
            }

            $artistName ??= $this->artistCreditParser->findNameForSlug($artistCredits, $slug);

            if (in_array('remixer', $roles, true)) {
                $remixTracks[] = $track;
                continue;
            }

            if (in_array('primary', $roles, true)) {
                $primaryTracks[] = $track;
                continue;
            }

            $featuredTracks[] = $track;
        }

        if ($artistName === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Artist not found',
            ], 404);
        }

        return response()->json([
            'artist' => [
                'name' => $artistName,
                'slug' => $slug,
            ],
            'primary_tracks' => array_values($primaryTracks),
            'featured_tracks' => array_values($featuredTracks),
            'remix_tracks' => array_values($remixTracks),
        ]);
    }

    public function update(Request $request, string $slug): JsonResponse
    {
        if ($forbidden = $this->authorizeAdmin($request)) {
            return $forbidden;
        }

        $payload = $request->validate([
            'name' => ['required', 'string', 'max:255'],
        ]);

        $newName = trim((string) $payload['name']);
        if ($newName === '') {
            return response()->json([
                'message' => 'Artist name is required.',
            ], 422);
        }

        $newSlug = $this->artistCreditParser->slugifyArtist($newName);
        $tracks = Track::query()
            ->whereJsonContains('artist_slugs', $slug)
            ->get();

        if ($tracks->isEmpty()) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Artist not found',
            ], 404);
        }

        $updatedTracks = 0;

        DB::transaction(function () use ($tracks, $slug, $newName, $newSlug, &$updatedTracks): void {
            foreach ($tracks as $track) {
                $updatedCredits = $this->renameArtistCredits(
                    is_array($track->artist_credits) ? $track->artist_credits : [],
                    $slug,
                    $newName,
                    $newSlug,
                );

                if ($updatedCredits === null) {
                    continue;
                }

                $track->forceFill([
                    'artist_credits' => $updatedCredits,
                    'artist_slugs' => array_values(array_map(
                        static fn (array $credit): string => $credit['slug'],
                        $updatedCredits,
                    )),
                    'artist' => $this->artistCreditParser->buildLegacyArtistString($updatedCredits),
                ])->save();

                $updatedTracks++;
            }
        });

        if ($updatedTracks === 0) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Artist not found',
            ], 404);
        }

        return response()->json([
            'message' => 'Artist renamed successfully.',
            'artist' => [
                'name' => $newName,
                'slug' => $newSlug,
            ],
            'updated_tracks' => $updatedTracks,
        ]);
    }

    /**
     * @param array<int, array{name?: string, slug?: string, roles?: array<int, string>}> $artistCredits
     * @return array<int, array{name: string, slug: string, roles: array<int, string>}>|null
     */
    private function renameArtistCredits(array $artistCredits, string $oldSlug, string $newName, string $newSlug): ?array
    {
        $updatedCredits = [];
        $matched = false;

        foreach ($artistCredits as $credit) {
            $creditName = is_string($credit['name'] ?? null) ? trim($credit['name']) : null;
            if ($creditName === null || $creditName === '') {
                continue;
            }

            $creditSlug = is_string($credit['slug'] ?? null) && trim((string) $credit['slug']) !== ''
                ? trim((string) $credit['slug'])
                : $this->artistCreditParser->slugifyArtist($creditName);

            $roles = $this->normalizeRoles(is_array($credit['roles'] ?? null) ? $credit['roles'] : []);

            if ($creditSlug === $oldSlug) {
                $creditName = $newName;
                $creditSlug = $newSlug;
                $matched = true;
            }

            if (isset($updatedCredits[$creditSlug])) {
                $updatedCredits[$creditSlug]['roles'] = $this->sortRoles(array_values(array_unique(array_merge(
                    $updatedCredits[$creditSlug]['roles'],
                    $roles,
                ))));

                if ($creditSlug === $newSlug) {
                    $updatedCredits[$creditSlug]['name'] = $newName;
                }

                continue;
            }

            $updatedCredits[$creditSlug] = [
                'name' => $creditSlug === $newSlug ? $newName : $creditName,
                'slug' => $creditSlug,
                'roles' => $roles === [] ? ['primary'] : $this->sortRoles($roles),
            ];
        }

        return $matched ? array_values($updatedCredits) : null;
    }

    /**
     * @param array<int, mixed> $roles
     * @return array<int, string>
     */
    private function normalizeRoles(array $roles): array
    {
        return $this->sortRoles(array_values(array_unique(array_filter(array_map(
            static fn (mixed $role): ?string => is_string($role) ? trim($role) : null,
            $roles,
        )))));
    }

    /**
     * @param array<int, string> $roles
     * @return array<int, string>
     */
    private function sortRoles(array $roles): array
    {
        $order = [
            'primary' => 0,
            'featured' => 1,
            'remixer' => 2,
        ];

        usort($roles, static fn (string $left, string $right): int => ($order[$left] ?? 99) <=> ($order[$right] ?? 99));

        return $roles;
    }

    private function authorizeAdmin(Request $request): ?JsonResponse
    {
        $user = $request->user();

        if (!method_exists($user, 'isAdmin') || !$user->isAdmin()) {
            return response()->json(['message' => 'Forbidden. Admin access required.'], 403);
        }

        return null;
    }
}
