<?php

namespace App\Services;

use App\Models\DirectMessage;
use App\Models\PatternExample;
use App\Models\Track;
use App\Models\User;
use App\Support\SegmentDisplayLabel;
use InvalidArgumentException;

class DirectMessageService
{
    public function __construct(
        private readonly PublicRadioWaveformService $waveforms,
    ) {}

    /**
     * @param array<int, int> $friendIds
     * @return array<int, array<string, mixed>>
     */
    public function conversationSummariesForUser(User $viewer, array $friendIds): array
    {
        $friendIds = array_values(array_unique(array_filter(
            array_map(static fn ($id): int => (int) $id, $friendIds),
            static fn (int $id): bool => $id > 0,
        )));

        if ($friendIds === []) {
            return [];
        }

        $messages = DirectMessage::query()
            ->with(['sender', 'recipient'])
            ->where(function ($query) use ($viewer, $friendIds): void {
                $query
                    ->where(function ($inner) use ($viewer, $friendIds): void {
                        $inner->where('sender_id', $viewer->id)
                            ->whereIn('recipient_id', $friendIds);
                    })
                    ->orWhere(function ($inner) use ($viewer, $friendIds): void {
                        $inner->where('recipient_id', $viewer->id)
                            ->whereIn('sender_id', $friendIds);
                    });
            })
            ->orderByDesc('created_at')
            ->get();

        $latestByFriendId = [];
        foreach ($messages as $message) {
            $otherUserId = $message->sender_id === $viewer->id
                ? (int) $message->recipient_id
                : (int) $message->sender_id;

            if (!isset($latestByFriendId[$otherUserId])) {
                $latestByFriendId[$otherUserId] = $this->serializeMessage($message);
            }
        }

        $unreadCounts = DirectMessage::query()
            ->selectRaw('sender_id, COUNT(*) as aggregate')
            ->where('recipient_id', $viewer->id)
            ->whereNull('read_at')
            ->whereIn('sender_id', $friendIds)
            ->groupBy('sender_id')
            ->pluck('aggregate', 'sender_id');

        $summaries = [];
        foreach ($friendIds as $friendId) {
            $summaries[$friendId] = [
                'latest_message' => $latestByFriendId[$friendId] ?? null,
                'last_message_at' => $latestByFriendId[$friendId]['created_at'] ?? null,
                'unread_count' => (int) ($unreadCounts[$friendId] ?? 0),
            ];
        }

        return $summaries;
    }

    /**
     * @return array<int, array<string, mixed>>
     */
    public function conversation(User $viewer, User $otherUser, int $limit = 80): array
    {
        $messages = DirectMessage::query()
            ->with(['sender', 'recipient'])
            ->where(function ($query) use ($viewer, $otherUser): void {
                $query
                    ->where(function ($inner) use ($viewer, $otherUser): void {
                        $inner->where('sender_id', $viewer->id)
                            ->where('recipient_id', $otherUser->id);
                    })
                    ->orWhere(function ($inner) use ($viewer, $otherUser): void {
                        $inner->where('sender_id', $otherUser->id)
                            ->where('recipient_id', $viewer->id);
                    });
            })
            ->latest('created_at')
            ->limit(max(1, min(200, $limit)))
            ->get()
            ->reverse()
            ->values();

        return $messages->map(fn (DirectMessage $message): array => $this->serializeMessage($message))->all();
    }

    public function markConversationRead(User $viewer, User $otherUser): int
    {
        return DirectMessage::query()
            ->where('sender_id', $otherUser->id)
            ->where('recipient_id', $viewer->id)
            ->whereNull('read_at')
            ->update([
                'read_at' => now(),
                'updated_at' => now(),
            ]);
    }

    public function unreadCountForUser(User $viewer): int
    {
        return DirectMessage::query()
            ->where('recipient_id', $viewer->id)
            ->whereNull('read_at')
            ->count();
    }

    /**
     * @param array<string, mixed>|null $share
     */
    public function sendMessage(User $sender, User $recipient, ?string $text, ?array $share = null): DirectMessage
    {
        $normalizedText = $this->normalizeText($text);
        $normalizedShare = $this->normalizeSharePayload($share);

        if ($normalizedText === null && $normalizedShare === null) {
            throw new InvalidArgumentException('Message text or share payload is required');
        }

        $message = DirectMessage::create([
            'sender_id' => $sender->id,
            'recipient_id' => $recipient->id,
            'type' => $normalizedShare['type'] ?? DirectMessage::TYPE_MESSAGE,
            'body' => $normalizedText,
            'payload' => $normalizedShare ? $normalizedShare['payload'] : null,
        ]);

        return $message->load(['sender', 'recipient']);
    }

    public function serializeMessage(DirectMessage $message): array
    {
        $message->loadMissing(['sender', 'recipient']);

        return [
            'id' => (int) $message->id,
            'type' => $message->type,
            'text' => $message->body,
            'payload' => $message->payload,
            'sender' => $this->serializeUser($message->sender),
            'recipient' => $this->serializeUser($message->recipient),
            'created_at' => $message->created_at?->toIso8601String(),
            'read_at' => $message->read_at?->toIso8601String(),
        ];
    }

    private function normalizeText(?string $text): ?string
    {
        if ($text === null) {
            return null;
        }

        $normalized = trim(preg_replace('/\s+/u', ' ', $text) ?? '');
        if ($normalized === '') {
            return null;
        }

        return mb_substr($normalized, 0, 1000);
    }

    /**
     * @param array<string, mixed>|null $share
     * @return array{type:string,payload:array<string, mixed>}|null
     */
    private function normalizeSharePayload(?array $share): ?array
    {
        if (!is_array($share)) {
            return null;
        }

        $shareType = strtolower(trim((string) ($share['type'] ?? '')));
        if ($shareType === 'track') {
            $trackId = (int) ($share['track_id'] ?? 0);
            $track = Track::query()->find($trackId);
            if (!$track) {
                throw new InvalidArgumentException('Track share target not found');
            }

            return [
                'type' => DirectMessage::TYPE_TRACK_SHARE,
                'payload' => $this->buildTrackSharePayload($track),
            ];
        }

        if ($shareType === 'segment') {
            $segmentId = (int) ($share['segment_id'] ?? 0);
            $segment = PatternExample::query()
                ->with(['track', 'pattern'])
                ->find($segmentId);

            if (!$segment || !$segment->track) {
                throw new InvalidArgumentException('Segment share target not found');
            }

            return [
                'type' => DirectMessage::TYPE_SEGMENT_SHARE,
                'payload' => $this->buildSegmentSharePayload($segment),
            ];
        }

        throw new InvalidArgumentException('Unsupported share payload');
    }

    /**
     * @return array<string, mixed>
     */
    private function buildTrackSharePayload(Track $track): array
    {
        return [
            'type' => 'track',
            'track_id' => (int) $track->id,
            'title' => $track->title,
            'artist' => $track->artist,
            'duration' => (float) $track->duration,
            'genre' => $track->genre,
            'route' => '/track/' . (int) $track->id,
        ];
    }

    /**
     * @return array<string, mixed>
     */
    private function buildSegmentSharePayload(PatternExample $segment): array
    {
        $track = $segment->track;
        if (!$track) {
            throw new InvalidArgumentException('Segment track not found');
        }

        $waveform = $this->waveforms->getWaveformPayload($track);
        $trackDuration = max(0.1, (float) ($waveform['duration'] ?? $track->duration));
        $peaks = is_array($waveform['peaks'] ?? null) ? $waveform['peaks'] : [];
        $segmentDuration = max(0.1, (float) $segment->end_time - (float) $segment->start_time);
        $type = SegmentDisplayLabel::extractTypeFromPatternName($segment->pattern->name ?? null);

        return [
            'type' => 'segment',
            'track_id' => (int) $track->id,
            'track_title' => $track->title,
            'artist' => $track->artist,
            'track_route' => '/track/' . (int) $track->id,
            'segment_id' => (int) $segment->id,
            'segment_route' => '/track/' . (int) $track->id . '/segments',
            'label' => SegmentDisplayLabel::fromType($type),
            'start' => round((float) $segment->start_time, 3),
            'end' => round((float) $segment->end_time, 3),
            'waveform' => [
                'duration' => round($segmentDuration, 3),
                'peaks' => $this->sliceWaveformPeaks(
                    $peaks,
                    $trackDuration,
                    (float) $segment->start_time,
                    (float) $segment->end_time,
                ),
            ],
        ];
    }

    /**
     * @param array<int, mixed> $peaks
     * @return array<int, float>
     */
    private function sliceWaveformPeaks(array $peaks, float $trackDuration, float $start, float $end): array
    {
        if ($peaks === []) {
            return [];
        }

        $peakCount = count($peaks);
        $startRatio = max(0.0, min(1.0, $start / max(0.001, $trackDuration)));
        $endRatio = max($startRatio, min(1.0, $end / max(0.001, $trackDuration)));
        $startIndex = max(0, min($peakCount - 1, (int) floor($startRatio * $peakCount)));
        $endIndex = max($startIndex + 1, min($peakCount, (int) ceil($endRatio * $peakCount)));
        $slice = array_slice($peaks, $startIndex, $endIndex - $startIndex);

        return $this->downsamplePeaks($slice, 96);
    }

    /**
     * @param array<int, mixed> $peaks
     * @return array<int, float>
     */
    private function downsamplePeaks(array $peaks, int $targetCount): array
    {
        $normalized = array_values(array_map(static fn ($peak): float => round((float) $peak, 4), $peaks));
        $count = count($normalized);
        if ($count <= $targetCount) {
            return $normalized;
        }

        $bucketSize = $count / $targetCount;
        $output = [];
        for ($bucket = 0; $bucket < $targetCount; $bucket++) {
            $bucketStart = (int) floor($bucket * $bucketSize);
            $bucketEnd = (int) floor(($bucket + 1) * $bucketSize);
            $bucketSlice = array_slice($normalized, $bucketStart, max(1, $bucketEnd - $bucketStart));
            $output[] = round(max($bucketSlice), 4);
        }

        return $output;
    }

    private function serializeUser(?User $user): ?array
    {
        if (!$user) {
            return null;
        }

        return [
            'id' => (int) $user->id,
            'name' => $user->name,
            'email' => $user->email,
        ];
    }
}
