<?php

namespace App\Http\Controllers;

use App\Models\Friendship;
use App\Models\User;
use App\Services\DirectMessageService;
use App\Services\FriendshipService;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use InvalidArgumentException;

class FriendController extends Controller
{
    public function __construct(
        private readonly FriendshipService $friendships,
        private readonly DirectMessageService $messages,
    ) {}

    public function state(Request $request): JsonResponse
    {
        return response()->json($this->friendships->buildState($request->user()));
    }

    public function search(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'query' => ['nullable', 'string', 'max:100'],
        ]);

        return response()->json([
            'users' => $this->friendships->searchUsers(
                $request->user(),
                (string) ($validated['query'] ?? ''),
            ),
        ]);
    }

    public function sendRequest(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'user_id' => ['required', 'integer'],
        ]);

        try {
            $friendship = $this->friendships->sendRequest($request->user(), (int) $validated['user_id']);
        } catch (InvalidArgumentException $exception) {
            return response()->json([
                'message' => $exception->getMessage(),
            ], 422);
        }

        return response()->json([
            'message' => $friendship->status === Friendship::STATUS_ACCEPTED
                ? 'Friend request accepted'
                : 'Friend request sent',
            'friendship_id' => (int) $friendship->id,
            'requester_id' => (int) $friendship->requester_id,
            'addressee_id' => (int) $friendship->addressee_id,
            'status' => $friendship->status,
        ]);
    }

    public function respond(Request $request, Friendship $friendship): JsonResponse
    {
        $validated = $request->validate([
            'action' => ['required', 'string', 'in:accept,decline'],
        ]);

        try {
            $updated = $this->friendships->respondToRequest(
                $request->user(),
                $friendship,
                (string) $validated['action'],
            );
        } catch (InvalidArgumentException $exception) {
            return response()->json([
                'message' => $exception->getMessage(),
            ], 422);
        }

        return response()->json([
            'message' => $updated->status === Friendship::STATUS_ACCEPTED
                ? 'Friend request accepted'
                : 'Friend request declined',
            'friendship_id' => (int) $updated->id,
            'requester_id' => (int) $updated->requester_id,
            'addressee_id' => (int) $updated->addressee_id,
            'status' => $updated->status,
        ]);
    }

    public function conversation(Request $request, User $user): JsonResponse
    {
        try {
            $otherUser = $this->friendships->assertAcceptedFriend($request->user(), (int) $user->id);
        } catch (InvalidArgumentException $exception) {
            return response()->json([
                'message' => $exception->getMessage(),
            ], 422);
        }

        return response()->json([
            'conversation_with' => [
                'id' => (int) $otherUser->id,
                'name' => $otherUser->name,
                'email' => $otherUser->email,
            ],
            'messages' => $this->messages->conversation($request->user(), $otherUser),
        ]);
    }

    public function sendMessage(Request $request): JsonResponse
    {
        $validated = $request->validate([
            'recipient_id' => ['required', 'integer'],
            'text' => ['nullable', 'string', 'max:1000'],
            'share' => ['nullable', 'array'],
            'share.type' => ['nullable', 'string', 'in:track,segment'],
            'share.track_id' => ['nullable', 'integer'],
            'share.segment_id' => ['nullable', 'integer'],
        ]);

        try {
            $recipient = $this->friendships->assertAcceptedFriend($request->user(), (int) $validated['recipient_id']);
            $message = $this->messages->sendMessage(
                $request->user(),
                $recipient,
                isset($validated['text']) ? (string) $validated['text'] : null,
                isset($validated['share']) && is_array($validated['share']) ? $validated['share'] : null,
            );
        } catch (InvalidArgumentException $exception) {
            return response()->json([
                'message' => $exception->getMessage(),
            ], 422);
        }

        return response()->json([
            'message' => 'Direct message sent',
            'direct_message' => $this->messages->serializeMessage($message),
        ], 201);
    }

    public function markConversationRead(Request $request, User $user): JsonResponse
    {
        try {
            $otherUser = $this->friendships->assertAcceptedFriend($request->user(), (int) $user->id);
        } catch (InvalidArgumentException $exception) {
            return response()->json([
                'message' => $exception->getMessage(),
            ], 422);
        }

        $updatedCount = $this->messages->markConversationRead($request->user(), $otherUser);

        return response()->json([
            'message' => 'Conversation marked as read',
            'updated' => $updatedCount,
        ]);
    }
}
