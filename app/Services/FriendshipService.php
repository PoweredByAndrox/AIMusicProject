<?php

namespace App\Services;

use App\Models\Friendship;
use App\Models\User;
use InvalidArgumentException;

class FriendshipService
{
    public function __construct(
        private readonly DirectMessageService $messages,
    ) {}

    /**
     * @return array<string, mixed>
     */
    public function buildState(User $viewer): array
    {
        $relationships = Friendship::query()
            ->with(['requester', 'addressee'])
            ->where(function ($query) use ($viewer): void {
                $query->where('requester_id', $viewer->id)
                    ->orWhere('addressee_id', $viewer->id);
            })
            ->whereIn('status', [Friendship::STATUS_PENDING, Friendship::STATUS_ACCEPTED])
            ->orderByDesc('updated_at')
            ->get();

        $accepted = $relationships
            ->filter(fn (Friendship $friendship): bool => $friendship->status === Friendship::STATUS_ACCEPTED)
            ->values();

        $friendIds = $accepted
            ->map(fn (Friendship $friendship): int => $this->otherParty($friendship, $viewer)->id)
            ->unique()
            ->values()
            ->all();

        $conversationSummaries = $this->messages->conversationSummariesForUser($viewer, $friendIds);

        $friends = $accepted
            ->map(function (Friendship $friendship) use ($viewer, $conversationSummaries): array {
                $other = $this->otherParty($friendship, $viewer);
                $summary = $conversationSummaries[$other->id] ?? [
                    'latest_message' => null,
                    'last_message_at' => null,
                    'unread_count' => 0,
                ];

                return [
                    'friendship_id' => (int) $friendship->id,
                    'user' => $this->serializeUser($other),
                    'accepted_at' => $friendship->accepted_at?->toIso8601String(),
                    'latest_message' => $summary['latest_message'],
                    'last_message_at' => $summary['last_message_at'],
                    'unread_count' => (int) ($summary['unread_count'] ?? 0),
                ];
            })
            ->sortByDesc(static fn (array $friend): string => (string) ($friend['last_message_at'] ?? $friend['accepted_at'] ?? ''))
            ->values()
            ->all();

        $incomingRequests = $relationships
            ->filter(fn (Friendship $friendship): bool => (
                $friendship->status === Friendship::STATUS_PENDING
                && (int) $friendship->addressee_id === (int) $viewer->id
            ))
            ->map(fn (Friendship $friendship): array => $this->serializeRequest($friendship, $viewer))
            ->values()
            ->all();

        $outgoingRequests = $relationships
            ->filter(fn (Friendship $friendship): bool => (
                $friendship->status === Friendship::STATUS_PENDING
                && (int) $friendship->requester_id === (int) $viewer->id
            ))
            ->map(fn (Friendship $friendship): array => $this->serializeRequest($friendship, $viewer))
            ->values()
            ->all();

        $unreadMessages = array_sum(array_map(
            static fn (array $friend): int => (int) ($friend['unread_count'] ?? 0),
            $friends,
        ));

        return [
            'viewer' => $this->serializeUser($viewer),
            'friends' => $friends,
            'incoming_requests' => $incomingRequests,
            'outgoing_requests' => $outgoingRequests,
            'notifications' => [
                'unread_messages' => $unreadMessages,
                'pending_requests' => count($incomingRequests),
                'total' => $unreadMessages + count($incomingRequests),
            ],
        ];
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    public function searchUsers(User $viewer, string $query): array
    {
        $normalizedQuery = mb_strtolower(trim($query));
        $relationships = $this->relationshipMap($viewer);

        $candidates = User::query()
            ->where('id', '!=', $viewer->id)
            ->when($normalizedQuery !== '', function ($builder) use ($normalizedQuery): void {
                $like = '%' . str_replace(['\\', '%', '_'], ['\\\\', '\\%', '\\_'], $normalizedQuery) . '%';
                $builder->where(function ($query) use ($like): void {
                    $query->whereRaw('LOWER(name) LIKE ?', [$like])
                        ->orWhereRaw('LOWER(email) LIKE ?', [$like]);
                });
            })
            ->orderBy('name')
            ->limit(12)
            ->get();

        return $candidates->map(function (User $candidate) use ($relationships): array {
            $relationship = $relationships[$candidate->id] ?? null;

            return [
                'user' => $this->serializeUser($candidate),
                'relationship' => $relationship ? [
                    'friendship_id' => (int) $relationship->id,
                    'status' => $relationship->status,
                    'direction' => (int) $relationship->requester_id === (int) $candidate->id ? 'incoming' : 'outgoing',
                ] : null,
            ];
        })->all();
    }

    public function sendRequest(User $viewer, int $recipientId): Friendship
    {
        $recipient = User::query()->find($recipientId);
        if (!$recipient) {
            throw new InvalidArgumentException('Selected user was not found');
        }

        if ((int) $recipient->id === (int) $viewer->id) {
            throw new InvalidArgumentException('You cannot add yourself as a friend');
        }

        $existing = $this->relationshipBetween($viewer, $recipient);
        if ($existing && $existing->status === Friendship::STATUS_ACCEPTED) {
            return $existing;
        }

        if ($existing && $existing->status === Friendship::STATUS_PENDING) {
            if ((int) $existing->requester_id === (int) $viewer->id) {
                return $existing;
            }

            $existing->forceFill([
                'status' => Friendship::STATUS_ACCEPTED,
                'accepted_at' => now(),
                'responded_at' => now(),
            ])->save();

            return $existing->fresh(['requester', 'addressee']);
        }

        if ($existing) {
            $existing->delete();
        }

        return Friendship::create([
            'requester_id' => $viewer->id,
            'addressee_id' => $recipient->id,
            'status' => Friendship::STATUS_PENDING,
        ])->load(['requester', 'addressee']);
    }

    public function respondToRequest(User $viewer, Friendship $friendship, string $action): Friendship
    {
        if ((int) $friendship->addressee_id !== (int) $viewer->id) {
            throw new InvalidArgumentException('Only the requested user can respond to this friend request');
        }

        if ($friendship->status !== Friendship::STATUS_PENDING) {
            throw new InvalidArgumentException('This friend request is no longer pending');
        }

        $normalizedAction = strtolower(trim($action));
        if ($normalizedAction === 'accept') {
            $friendship->forceFill([
                'status' => Friendship::STATUS_ACCEPTED,
                'accepted_at' => now(),
                'responded_at' => now(),
            ])->save();

            return $friendship->fresh(['requester', 'addressee']);
        }

        if ($normalizedAction === 'decline') {
            $friendship->forceFill([
                'status' => Friendship::STATUS_DECLINED,
                'accepted_at' => null,
                'responded_at' => now(),
            ])->save();

            return $friendship->fresh(['requester', 'addressee']);
        }

        throw new InvalidArgumentException('Unsupported friend request action');
    }

    public function assertAcceptedFriend(User $viewer, int $otherUserId): User
    {
        $otherUser = User::query()->find($otherUserId);
        if (!$otherUser) {
            throw new InvalidArgumentException('Friend was not found');
        }

        $relationship = $this->relationshipBetween($viewer, $otherUser);
        if (!$relationship || $relationship->status !== Friendship::STATUS_ACCEPTED) {
            throw new InvalidArgumentException('Only accepted friends can open this conversation');
        }

        return $otherUser;
    }

    /**
     * @return array<int, int>
     */
    public function acceptedFriendIds(User $viewer): array
    {
        return Friendship::query()
            ->where('status', Friendship::STATUS_ACCEPTED)
            ->where(function ($query) use ($viewer): void {
                $query->where('requester_id', $viewer->id)
                    ->orWhere('addressee_id', $viewer->id);
            })
            ->get()
            ->map(function (Friendship $friendship) use ($viewer): int {
                return (int) ((int) $friendship->requester_id === (int) $viewer->id
                    ? $friendship->addressee_id
                    : $friendship->requester_id);
            })
            ->unique()
            ->values()
            ->all();
    }

    /**
     * @return array<int, Friendship>
     */
    private function relationshipMap(User $viewer): array
    {
        $relationships = Friendship::query()
            ->with(['requester', 'addressee'])
            ->where(function ($query) use ($viewer): void {
                $query->where('requester_id', $viewer->id)
                    ->orWhere('addressee_id', $viewer->id);
            })
            ->whereIn('status', [Friendship::STATUS_PENDING, Friendship::STATUS_ACCEPTED])
            ->get();

        $map = [];
        foreach ($relationships as $relationship) {
            $otherUser = $this->otherParty($relationship, $viewer);
            $map[(int) $otherUser->id] = $relationship;
        }

        return $map;
    }

    private function relationshipBetween(User $left, User $right): ?Friendship
    {
        return Friendship::query()
            ->with(['requester', 'addressee'])
            ->where(function ($query) use ($left, $right): void {
                $query
                    ->where(function ($inner) use ($left, $right): void {
                        $inner->where('requester_id', $left->id)
                            ->where('addressee_id', $right->id);
                    })
                    ->orWhere(function ($inner) use ($left, $right): void {
                        $inner->where('requester_id', $right->id)
                            ->where('addressee_id', $left->id);
                    });
            })
            ->latest('updated_at')
            ->first();
    }

    private function otherParty(Friendship $friendship, User $viewer): User
    {
        return (int) $friendship->requester_id === (int) $viewer->id
            ? $friendship->addressee
            : $friendship->requester;
    }

    /**
     * @return array<string, mixed>
     */
    private function serializeRequest(Friendship $friendship, User $viewer): array
    {
        $other = $this->otherParty($friendship, $viewer);

        return [
            'friendship_id' => (int) $friendship->id,
            'status' => $friendship->status,
            'direction' => (int) $friendship->requester_id === (int) $viewer->id ? 'outgoing' : 'incoming',
            'user' => $this->serializeUser($other),
            'created_at' => $friendship->created_at?->toIso8601String(),
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function serializeUser(User $user): array
    {
        return [
            'id' => (int) $user->id,
            'name' => $user->name,
            'email' => $user->email,
        ];
    }
}
