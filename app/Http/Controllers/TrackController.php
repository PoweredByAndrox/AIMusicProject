<?php

namespace App\Http\Controllers;

use App\Models\Track;
use App\Models\QueueTask;
use App\Models\UserTrackLike;
use App\Services\ArtistCreditParser;
use App\Services\FriendshipService;
use App\Services\MetadataExtractor;
use App\Services\PublicRadioHlsService;
use App\Services\TrackPlaybackManifestService;
use App\Services\PublicRadioWaveformService;
use App\Services\YouTubeImporter;
use App\Jobs\TrackAnalysisJob;
use App\Jobs\YouTubeDownloadJob;
use Illuminate\Http\JsonResponse;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Facades\Validator;
use Illuminate\Validation\ValidationException;
use RuntimeException;

/**
 * TrackController
 * 
 * Handles track upload, import, listing, and management.
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 4.1, 11.1
 */
class TrackController extends Controller
{
    /**
     * Maximum file size in bytes (100MB)
     */
    private const MAX_FILE_SIZE = 104857600;

    /**
     * Allowed audio file extensions
     */
    private const ALLOWED_EXTENSIONS = ['mp3', 'wav'];

    /**
     * Allowed MIME types
     */
    private const ALLOWED_MIMES = [
        'audio/mpeg',
        'audio/mp3',
        'audio/wav',
        'audio/wave',
        'audio/x-wav',
    ];

    /**
     * @var array<int, string>
     */
    private const RADIO_BITRATE_LADDER = ['32k', '48k', '64k', '96k', '128k', '160k', '192k', '256k', '320k'];
    private const HLS_PACKAGE_VERSION = 'v4';
    private const HLS_SEGMENT_SECONDS = 15;

    public function __construct(
        private readonly MetadataExtractor $metadataExtractor,
        private readonly ArtistCreditParser $artistCreditParser,
        private readonly FriendshipService $friendships,
        private readonly YouTubeImporter $youtubeImporter,
        private readonly PublicRadioWaveformService $publicRadioWaveformService,
        private readonly PublicRadioHlsService $publicRadioHlsService,
        private readonly TrackPlaybackManifestService $trackPlaybackManifestService,
    ) {}

    public function store(Request $request): JsonResponse
    {
        return $this->upload($request);
    }

    /**
     * Upload a local audio file.
     * 
     * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function upload(Request $request): JsonResponse
    {
        // Validate file presence
        if (!$request->hasFile('file')) {
            return response()->json([
                'error' => 'No file provided',
                'message' => 'Please upload an audio file',
            ], 400);
        }

        $file = $request->file('file');

        // Validate file format (Requirement 1.3)
        $extension = strtolower($file->getClientOriginalExtension());
        if (!in_array($extension, self::ALLOWED_EXTENSIONS)) {
            return response()->json([
                'error' => 'Unsupported format',
                'message' => 'Поддерживаются только MP3 и WAV файлы',
            ], 400);
        }

        // Validate MIME type
        $mimeType = $file->getMimeType();
        if (!in_array($mimeType, self::ALLOWED_MIMES)) {
            return response()->json([
                'error' => 'Invalid file type',
                'message' => 'Поддерживаются только MP3 и WAV файлы',
            ], 400);
        }

        // Validate file size (Requirement 1.5)
        if ($file->getSize() > self::MAX_FILE_SIZE) {
            return response()->json([
                'error' => 'File too large',
                'message' => 'Максимальный размер файла: 100MB',
            ], 400);
        }

        try {
            // Store file (Requirement 1.1)
            $filename = uniqid('track_') . '.' . $extension;
            $path = $file->storeAs('audio', $filename, 'local');

            // Extract metadata (Requirement 1.2)
            $fullPath = Storage::disk('local')->path($path);
            $metadata = $this->metadataExtractor->extract($fullPath);
            $parsedMetadata = $this->artistCreditParser->parseUploadMetadata(
                $metadata['artist'] ?? null,
                $metadata['title'] ?? null,
                $file->getClientOriginalName(),
            );

            // Create track record
            $track = Track::create([
                'title' => $parsedMetadata['title'],
                'artist' => $parsedMetadata['artist'],
                'artist_credits' => $parsedMetadata['artist_credits'],
                'artist_slugs' => $parsedMetadata['artist_slugs'],
                'album' => $metadata['album'] ?? null,
                'year' => $metadata['year'] ?? null,
                'duration' => $metadata['duration'] ?? 0,
                'file_path' => $path,
                'source' => 'upload',
                'analysis_status' => 'pending',
            ]);

            // Queue for ML analysis (Requirement 1.4)
            TrackAnalysisJob::dispatch($track);

            return response()->json([
                'message' => 'Track uploaded successfully',
                'track' => $track,
            ], 201);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Upload failed',
                'message' => 'Не удалось загрузить файл: ' . $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Import music from YouTube.
     * 
     * Requirements: 2.1, 2.2, 2.3, 2.5, 2.6
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function importYoutube(Request $request): JsonResponse
    {
        $url = $request->input('url');
        $maxVideos = (int) $request->input('max_videos', 200);
        $likeAll = $request->boolean('like_all', false);
        $userId = $request->user()?->id;

        if (empty($url)) {
            return response()->json([
                'error' => 'No URL provided',
                'message' => 'Please provide a YouTube URL',
            ], 400);
        }

        // Validate URL (Requirement 2.5)
        $validation = $this->youtubeImporter->validateUrl($url);

        if (!$validation['valid']) {
            return response()->json([
                'error' => 'Invalid URL',
                'message' => 'Некорректный URL YouTube',
            ], 400);
        }

        // Dispatch download job (Requirement 2.6)
        YouTubeDownloadJob::dispatch($url, $validation['type'], $validation['id'], $maxVideos, $userId, $likeAll);

        $typeMessages = [
            YouTubeImporter::TYPE_VIDEO => 'Video import started',
            YouTubeImporter::TYPE_PLAYLIST => 'Playlist import started',
            YouTubeImporter::TYPE_CHANNEL => 'Channel import started',
        ];

        return response()->json([
            'message' => $typeMessages[$validation['type']] ?? 'Import started',
            'type' => $validation['type'],
            'id' => $validation['id'],
            'status' => 'processing',
        ], 202);
    }

    /**
     * Batch import multiple YouTube URLs.
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function importYoutubeBatch(Request $request): JsonResponse
    {
        $urls = $request->input('urls', []);
        $likeAll = $request->boolean('like_all', false);
        $userId = $request->user()?->id;
        
        // Support both array and newline-separated string
        if (is_string($urls)) {
            $urls = array_filter(array_map('trim', explode("\n", $urls)));
        }

        if (empty($urls)) {
            return response()->json([
                'error' => 'No URLs provided',
                'message' => 'Укажите список URL для импорта',
            ], 400);
        }

        $results = [
            'queued' => 0,
            'invalid' => 0,
            'errors' => [],
        ];

        $delay = 0;
        foreach ($urls as $url) {
            $url = trim($url);
            if (empty($url)) continue;

            $validation = $this->youtubeImporter->validateUrl($url);

            if (!$validation['valid']) {
                $results['invalid']++;
                $results['errors'][] = "Invalid URL: {$url}";
                continue;
            }

            // Dispatch with staggered delay to avoid rate limiting
            YouTubeDownloadJob::dispatch($url, $validation['type'], $validation['id'], 200, $userId, $likeAll)
                ->delay(now()->addSeconds($delay));
            
            $results['queued']++;
            $delay += 3; // 3 seconds between each
        }

        return response()->json([
            'message' => "Batch import started: {$results['queued']} URLs queued",
            'queued' => $results['queued'],
            'invalid' => $results['invalid'],
            'errors' => $results['errors'],
            'status' => 'processing',
        ], 202);
    }

    /**
     * Display a listing of tracks.
     * 
     * Requirements: 11.1, 11.2, 11.3, 11.4
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function index(Request $request): JsonResponse
    {
        $query = Track::query();
        $userId = $request->user()?->id;

        if ($userId !== null) {
            $query->with('trackLike');
        }

        // Filter by genre (Requirement 11.3)
        if ($request->has('genre')) {
            $query->byGenre($request->input('genre'));
        }

        // Filter by mood (Requirement 11.3)
        if ($request->has('mood')) {
            $query->byMood($request->input('mood'));
        }

        // Filter by BPM range (Requirement 11.3)
        if ($request->has('bpm_min') && $request->has('bpm_max')) {
            $query->bpmRange(
                (float) $request->input('bpm_min'),
                (float) $request->input('bpm_max')
            );
        }

        // Filter by liked status (Requirement 11.3)
        if ($request->has('liked') && $request->input('liked') === 'true') {
            if ($userId !== null) {
                $query->likedByUser($userId);
            } else {
                $query->liked();
            }
        }

        // Filter by analysis status
        if ($request->has('status')) {
            $status = $request->input('status');
            $allowedStatuses = ['pending', 'processing', 'analyzing', 'completed', 'failed'];
            if (in_array($status, $allowedStatuses)) {
                $query->where('analysis_status', $status);
            }
        }

        // Search by title or artist (Requirement 11.4)
        // Supports: "title", "artist", "artist - title", "title - artist"
        if ($request->has('search')) {
            $search = trim($request->input('search'));
            
            // Check if search contains " - " separator
            if (str_contains($search, ' - ')) {
                $parts = array_map('trim', explode(' - ', $search, 2));
                $query->where(function ($q) use ($parts, $search) {
                    // Try "artist - title" format
                    $q->where(function ($q2) use ($parts) {
                        $q2->where('artist', 'ilike', "%{$parts[0]}%")
                           ->where('title', 'ilike', "%{$parts[1]}%");
                    })
                    // Or "title - artist" format
                    ->orWhere(function ($q2) use ($parts) {
                        $q2->where('title', 'ilike', "%{$parts[0]}%")
                           ->where('artist', 'ilike', "%{$parts[1]}%");
                    })
                    // Or just search both fields
                    ->orWhere('title', 'ilike', "%{$search}%")
                    ->orWhere('artist', 'ilike', "%{$search}%");
                });
            } else {
                $query->where(function ($q) use ($search) {
                    $q->where('title', 'ilike', "%{$search}%")
                      ->orWhere('artist', 'ilike', "%{$search}%");
                });
            }
        }

        // Sorting (Requirement 11.2)
        $sortBy = $request->input('sort_by', 'created_at');
        $sortOrder = $request->input('sort_order', 'desc');
        $allowedSorts = ['title', 'artist', 'created_at', 'bpm', 'duration'];

        // Keep placeholder recovery rows out of the default "latest first" view so
        // restored tracks are visible without changing filters.
        if ($sortBy === 'created_at' && $sortOrder !== 'asc') {
            $query->orderByRaw("CASE WHEN title LIKE 'YouTube Import %' THEN 1 ELSE 0 END ASC");
        }

        if (in_array($sortBy, $allowedSorts)) {
            $query->orderBy($sortBy, $sortOrder === 'asc' ? 'asc' : 'desc');
        }

        // Pagination
        $perPage = min((int) $request->input('per_page', 20), 100);
        $tracks = $query->paginate($perPage);

        return response()->json($tracks);
    }

    /**
     * Display the specified track.
     *
     * @param int $id
     * @return JsonResponse
     */
    public function show(int $id): JsonResponse
    {
        $viewer = request()->user();
        $userId = $viewer?->id;
        $track = Track::with([
            'trackLike',
            'fragments' => function ($query) use ($userId) {
                $query->forUser($userId);
            },
        ])->find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $payload = $track->toArray();
        $payload['friend_likes'] = $viewer
            ? $this->buildFriendLikesPayload($track->id, $viewer)
            : [];

        return response()->json($payload);
    }

    /**
     * Update the specified track.
     * 
     * Requirements: 14.1, 14.2, 14.3
     *
     * @param Request $request
     * @param int $id
     * @return JsonResponse
     */
    public function update(Request $request, int $id): JsonResponse
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $editableFields = ['genre', 'bpm', 'key', 'mode', 'mood'];
        $manuallyEdited = $track->manually_edited ?? [];

        foreach ($editableFields as $field) {
            if ($request->has($field)) {
                $track->$field = $request->input($field);
                $manuallyEdited[$field] = true;
            }
        }

        $track->manually_edited = $manuallyEdited;
        $track->save();

        return response()->json([
            'message' => 'Track updated successfully',
            'track' => $track,
        ]);
    }

    /**
     * Remove the specified track.
     * 
     * Requirements: 4.6
     *
     * @param Request $request
     * @param int $id
     * @return JsonResponse
     */
    public function destroy(Request $request, int $id): JsonResponse
    {
        if ($forbidden = $this->authorizeAdmin($request)) {
            return $forbidden;
        }

        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $this->deleteTrackAndArtifacts($track);

        return response()->json([
            'message' => 'Track deleted successfully',
        ]);
    }

    /**
     * Remove multiple tracks in a single request.
     *
     * @param Request $request
     * @return JsonResponse
     */
    public function destroyBatch(Request $request): JsonResponse
    {
        if ($forbidden = $this->authorizeAdmin($request)) {
            return $forbidden;
        }

        $validated = $request->validate([
            'ids' => ['required', 'array', 'min:1'],
            'ids.*' => ['integer', 'distinct'],
        ]);

        $ids = array_values($validated['ids']);
        $tracks = Track::whereIn('id', $ids)->get()->keyBy('id');
        $missingIds = array_values(array_diff($ids, $tracks->keys()->all()));

        if ($missingIds !== []) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Some tracks were not found',
                'missing_ids' => $missingIds,
            ], 404);
        }

        foreach ($ids as $id) {
            $this->deleteTrackAndArtifacts($tracks[$id]);
        }

        return response()->json([
            'message' => 'Tracks deleted successfully',
            'deleted_ids' => $ids,
            'deleted_count' => count($ids),
        ]);
    }

    /**
     * Retry analysis for a failed track without deleting it.
     */
    public function retryAnalysis(Request $request, Track $track): JsonResponse
    {
        if ($forbidden = $this->authorizeAdmin($request)) {
            return $forbidden;
        }

        if ($track->analysis_status !== 'failed') {
            return response()->json([
                'error' => 'Invalid status',
                'message' => 'Only failed tracks can be retried.',
            ], 400);
        }

        if (!$track->file_path || !Storage::disk('local')->exists($track->file_path)) {
            return response()->json([
                'error' => 'File not found',
                'message' => 'Audio file is missing. Retry is not possible.',
            ], 422);
        }

        QueueTask::query()
            ->where('task_type', QueueTask::TYPE_ANALYSIS)
            ->where('status', QueueTask::STATUS_FAILED)
            ->where('payload->track_id', $track->id)
            ->delete();

        $track->update(['analysis_status' => 'pending']);
        TrackAnalysisJob::dispatch($track->fresh());

        return response()->json([
            'success' => true,
            'message' => 'Track queued for re-analysis.',
            'track' => $track->fresh(),
        ], 202);
    }

    /**
     * Delete a track together with stored artifacts.
     *
     * @param Track $track
     * @return void
     */
    private function deleteTrackAndArtifacts(Track $track): void
    {
        $audioPath = is_string($track->file_path) ? trim($track->file_path) : '';
        if ($audioPath !== '') {
            $absoluteAudioPath = $this->isAbsolutePath($audioPath)
                ? $audioPath
                : Storage::disk('local')->path($audioPath);
            $this->deleteManagedFile($audioPath, 'local');
            $this->deleteManagedAbsoluteFile($absoluteAudioPath . '.meta.json');
            $this->deleteManagedAbsoluteFile($absoluteAudioPath . '.playback.json');
        }

        $this->deleteManagedFile($track->embedding_path, 'embeddings');
        $chromaPath = is_string($track->chroma_path) ? trim($track->chroma_path) : '';
        $this->deleteManagedFile($chromaPath);
        $this->deleteManagedFile($track->pitch_contour_path);
        if ($chromaPath !== '') {
            $absoluteChromaPath = $this->isAbsolutePath($chromaPath)
                ? $chromaPath
                : storage_path('app/' . ltrim(str_replace('\\', '/', $chromaPath), '/'));
            $this->deleteManagedAbsoluteFile(
                dirname($absoluteChromaPath) . DIRECTORY_SEPARATOR . 'chroma_fp_' . (int) $track->id . '.npy'
            );
        }

        $this->deleteManagedDirectory(storage_path('app/public_radio_variants/' . (int) $track->id));
        $this->deleteManagedDirectory(storage_path('app/public_radio_hls/' . (int) $track->id));
        $this->deleteManagedDirectory(storage_path('app/public_radio_waveforms/' . (int) $track->id));
        $this->deleteManagedDirectory(storage_path('app/stem_builds/track_' . (int) $track->id));

        $track->delete();
    }

    private function deleteManagedFile(?string $path, ?string $disk = null): void
    {
        if (!is_string($path) || trim($path) === '') {
            return;
        }

        $candidate = trim($path);
        if ($disk !== null && !$this->isAbsolutePath($candidate)) {
            $filesystem = Storage::disk($disk);
            if ($filesystem->exists($candidate)) {
                $filesystem->delete($candidate);
            }

            $candidate = $filesystem->path($candidate);
        }

        $this->deleteManagedAbsoluteFile($candidate);
    }

    private function deleteManagedAbsoluteFile(string $path): void
    {
        if (!$this->isManagedStoragePath($path) || !is_file($path)) {
            return;
        }

        @unlink($path);
    }

    private function deleteManagedDirectory(string $directory): void
    {
        if (!$this->isManagedStoragePath($directory)) {
            return;
        }

        if (is_link($directory) || is_file($directory)) {
            @unlink($directory);
            return;
        }

        if (is_dir($directory)) {
            $this->deleteDirectory($directory);
        }
    }

    private function isManagedStoragePath(string $path): bool
    {
        $normalized = str_replace(['\\', '/'], DIRECTORY_SEPARATOR, trim($path));
        if ($normalized === '') {
            return false;
        }

        $storageRoot = str_replace(['\\', '/'], DIRECTORY_SEPARATOR, storage_path('app'));

        return $normalized === $storageRoot
            || str_starts_with($normalized, $storageRoot . DIRECTORY_SEPARATOR);
    }

    private function isAbsolutePath(string $path): bool
    {
        return str_starts_with($path, '/')
            || preg_match('/^[A-Za-z]:[\\\\\\/]/', $path) === 1;
    }

    private function authorizeAdmin(Request $request): ?JsonResponse
    {
        $user = $request->user();

        if (!method_exists($user, 'isAdmin') || !$user->isAdmin()) {
            return response()->json(['message' => 'Forbidden. Admin access required.'], 403);
        }

        return null;
    }

    /**
     * @return array<int, array{id: int, name: string, email: string, liked_at: string|null}>
     */
    private function buildFriendLikesPayload(int $trackId, $viewer): array
    {
        $friendIds = $this->friendships->acceptedFriendIds($viewer);
        if ($friendIds === []) {
            return [];
        }

        return UserTrackLike::query()
            ->where('track_id', $trackId)
            ->whereIn('user_id', $friendIds)
            ->with('user:id,name,email')
            ->orderByDesc('created_at')
            ->get()
            ->filter(fn (UserTrackLike $like): bool => $like->user !== null)
            ->map(fn (UserTrackLike $like): array => [
                'id' => (int) $like->user->id,
                'name' => $like->user->name,
                'email' => $like->user->email,
                'liked_at' => $like->created_at?->toIso8601String(),
            ])
            ->values()
            ->all();
    }

    /**
     * Get queue status for imports and analysis.
     *
     * @return JsonResponse
     */
    public function queueStatus(): JsonResponse
    {
        // Get pending jobs from Redis queues
        // Note: Redis facade automatically adds prefix, so we use raw key names
        $redis = \Illuminate\Support\Facades\Redis::connection();
        
        // Count jobs in each queue (ready + delayed + reserved)
        $youtubeReady = $redis->llen('queues:youtube') ?: 0;
        $youtubeDelayed = $redis->zcard('queues:youtube:delayed') ?: 0;
        $youtubeReserved = $redis->zcard('queues:youtube:reserved') ?: 0;
        $youtubeTotal = $youtubeReady + $youtubeDelayed + $youtubeReserved;
        
        // Analysis queue
        $analysisReady = $redis->llen('queues:analysis') ?: 0;
        $analysisDelayed = $redis->zcard('queues:analysis:delayed') ?: 0;
        $analysisReserved = $redis->zcard('queues:analysis:reserved') ?: 0;
        $analysisTotal = $analysisReady + $analysisDelayed + $analysisReserved;
        
        // Default queue
        $defaultReady = $redis->llen('queues:default') ?: 0;
        $defaultDelayed = $redis->zcard('queues:default:delayed') ?: 0;
        $defaultReserved = $redis->zcard('queues:default:reserved') ?: 0;
        $defaultTotal = $defaultReady + $defaultDelayed + $defaultReserved;

        // Get recent tracks with their status
        $recentTracks = Track::orderBy('created_at', 'desc')
            ->limit(20)
            ->get(['id', 'title', 'artist', 'analysis_status', 'created_at', 'source']);

        // Count by status
        $statusCounts = Track::select('analysis_status', \DB::raw('COUNT(*) as count'))
            ->groupBy('analysis_status')
            ->get()
            ->keyBy('analysis_status');

        return response()->json([
            'queue' => [
                'youtube' => $youtubeTotal,
                'analysis' => $analysisTotal,
                'default' => $defaultTotal,
                'total' => $youtubeTotal + $analysisTotal + $defaultTotal,
            ],
            'tracks' => [
                'pending' => $statusCounts->get('pending')?->count ?? 0,
                'processing' => $statusCounts->get('processing')?->count ?? 0,
                'completed' => $statusCounts->get('completed')?->count ?? 0,
                'failed' => $statusCounts->get('failed')?->count ?? 0,
            ],
            'recent' => $recentTracks,
        ]);
    }

    /**
     * Stream audio file for playback.
     *
     * @param int $id
     * @return \Symfony\Component\HttpFoundation\StreamedResponse|\Illuminate\Http\JsonResponse
     */
    public function stream(Request $request, int $id)
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        if (!$track->file_path || !Storage::disk('local')->exists($track->file_path)) {
            return response()->json([
                'error' => 'File not found',
                'message' => 'Audio file not found',
            ], 404);
        }

        $path = Storage::disk('local')->path($track->file_path);
        $requestedBitrate = $this->normalizeRequestedBitrate($request->query('bitrate'));

        if ($requestedBitrate !== null) {
            try {
                $variantPath = $this->ensureBitrateVariant($track, $path, $requestedBitrate);
                return $this->streamFile($request, $variantPath, 'audio/mpeg');
            } catch (RuntimeException $e) {
                return response()->json([
                    'error' => 'Streaming unavailable',
                    'message' => $e->getMessage(),
                ], 500);
            }
        }

        $mimeType = mime_content_type($path) ?: 'audio/mpeg';
        return $this->streamFile($request, $path, $mimeType);
    }

    public function playbackOptions(int $id): JsonResponse
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        if (!$track->file_path || !Storage::disk('local')->exists($track->file_path)) {
            return response()->json([
                'error' => 'File not found',
                'message' => 'Audio file not found',
            ], 404);
        }

        return response()->json($this->trackPlaybackManifestService->manifestForTrack($track));
    }

    public function playbackModeStream(Request $request, int $id, string $modeKey)
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $mode = $this->trackPlaybackManifestService->resolveModeSource($track, $modeKey);
        if ($mode === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Playback mode not found',
            ], 404);
        }

        return $this->streamFile($request, $mode['path'], (string) ($mode['mime_type'] ?? 'audio/mpeg'));
    }

    public function playbackStemStream(Request $request, int $id, string $stemKey)
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $stem = $this->trackPlaybackManifestService->resolveStemSource($track, $stemKey);
        if ($stem === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Stem source not found',
            ], 404);
        }

        return $this->streamFile($request, $stem['path'], (string) ($stem['mime_type'] ?? 'audio/flac'));
    }

    public function hlsMaster(int $id)
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        try {
            $masterPath = $this->publicRadioHlsService->masterPath($track);
        } catch (RuntimeException $e) {
            if ($e->getMessage() === 'Audio file not found') {
                return response()->json([
                    'error' => 'File not found',
                    'message' => 'Audio file not found',
                ], 404);
            }

            return response()->json([
                'error' => 'Streaming unavailable',
                'message' => $e->getMessage(),
            ], 500);
        }

        return $this->serveStaticFile($masterPath, 'application/vnd.apple.mpegurl');
    }

    public function hlsVariant(int $id, string $profile)
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $normalizedProfile = $this->publicRadioHlsService->normalizeHlsProfile($profile);
        if ($normalizedProfile === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'HLS profile not found',
            ], 404);
        }

        try {
            $playlistPath = $this->publicRadioHlsService->variantPath($track, $normalizedProfile);
        } catch (RuntimeException $e) {
            if ($e->getMessage() === 'Audio file not found') {
                return response()->json([
                    'error' => 'File not found',
                    'message' => 'Audio file not found',
                ], 404);
            }

            if ($e->getMessage() === 'HLS playlist not found') {
                return response()->json([
                    'error' => 'Not found',
                    'message' => 'HLS playlist not found',
                ], 404);
            }

            return response()->json([
                'error' => 'Streaming unavailable',
                'message' => $e->getMessage(),
            ], 500);
        }

        return $this->serveStaticFile($playlistPath, 'application/vnd.apple.mpegurl');
    }

    public function hlsSegment(int $id, string $profile, string $segment)
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        $normalizedProfile = $this->publicRadioHlsService->normalizeHlsProfile($profile);
        if ($normalizedProfile === null) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'HLS profile not found',
            ], 404);
        }

        if (basename($segment) !== $segment) {
            return response()->json([
                'error' => 'Validation failed',
                'message' => 'Invalid segment name',
            ], 400);
        }

        try {
            $segmentPath = $this->publicRadioHlsService->segmentPath($track, $normalizedProfile, $segment);
        } catch (RuntimeException $e) {
            if ($e->getMessage() === 'Audio file not found') {
                return response()->json([
                    'error' => 'File not found',
                    'message' => 'Audio file not found',
                ], 404);
            }

            if ($e->getMessage() === 'HLS segment not found') {
                return response()->json([
                    'error' => 'Not found',
                    'message' => 'HLS segment not found',
                ], 404);
            }

            return response()->json([
                'error' => 'Streaming unavailable',
                'message' => $e->getMessage(),
            ], 500);
        }

        return $this->serveStaticFile($segmentPath, 'video/mp2t');
    }

    public function waveform(int $id)
    {
        $track = Track::find($id);

        if (!$track) {
            return response()->json([
                'error' => 'Not found',
                'message' => 'Track not found',
            ], 404);
        }

        try {
            $payload = $this->publicRadioWaveformService->getWaveformPayload($track);
        } catch (RuntimeException $e) {
            if ($e->getMessage() === 'Audio file not found') {
                return response()->json([
                    'error' => 'File not found',
                    'message' => 'Audio file not found',
                ], 404);
            }

            return response()->json([
                'error' => 'Waveform unavailable',
                'message' => $e->getMessage(),
            ], 500);
        }

        return response()->json($payload, 200, [
            'Cache-Control' => 'public, max-age=86400',
        ]);
    }

    private function normalizeRequestedBitrate(mixed $value): ?string
    {
        if (!is_string($value) || trim($value) === '') {
            return null;
        }

        $normalized = strtolower(trim($value));
        if (in_array($normalized, ['source', 'original', 'auto'], true)) {
            return null;
        }

        $allowed = [];
        foreach (self::RADIO_BITRATE_LADDER as $profile) {
            $allowed[str_replace('k', '', $profile)] = $profile;
            $allowed[$profile] = $profile;
        }

        return $allowed[$normalized] ?? null;
    }

    private function normalizeHlsProfile(string $profile): ?string
    {
        $normalized = strtolower(trim($profile));
        if ($normalized === '') {
            return null;
        }

        if (!str_ends_with($normalized, 'k')) {
            $normalized .= 'k';
        }

        return in_array($normalized, self::RADIO_BITRATE_LADDER, true) ? $normalized : null;
    }

    private function streamFile(Request $request, string $path, string $mimeType)
    {
        $size = filesize($path);

        $headers = [
            'Content-Type' => $mimeType,
            'Content-Length' => $size,
            'Accept-Ranges' => 'bytes',
            'Cache-Control' => 'public, max-age=86400',
        ];

        if ($request->hasHeader('Range')) {
            $range = $request->header('Range');
            preg_match('/bytes=(\d+)-(\d*)/', $range, $matches);

            $start = (int) ($matches[1] ?? 0);
            $end = isset($matches[2]) && $matches[2] !== '' ? (int) $matches[2] : $size - 1;
            $length = $end - $start + 1;

            $headers['Content-Range'] = "bytes {$start}-{$end}/{$size}";
            $headers['Content-Length'] = $length;

            return response()->stream(function () use ($path, $start, $length) {
                $handle = fopen($path, 'rb');
                fseek($handle, $start);
                echo fread($handle, $length);
                fclose($handle);
            }, 206, $headers);
        }

        return response()->stream(function () use ($path) {
            readfile($path);
        }, 200, $headers);
    }

    private function serveStaticFile(string $path, string $mimeType)
    {
        return response()->file($path, [
            'Content-Type' => $mimeType,
            'Cache-Control' => 'public, max-age=86400',
        ]);
    }

    private function ensureBitrateVariant(Track $track, string $sourcePath, string $bitrate): string
    {
        if (!function_exists('proc_open')) {
            throw new RuntimeException('Pre-generated bitrate variants are not available in this runtime');
        }

        $signature = sha1(implode('|', [
            (string) $track->id,
            (string) (filesize($sourcePath) ?: 0),
            (string) (filemtime($sourcePath) ?: 0),
            $bitrate,
        ]));

        $variantDirectory = storage_path('app/public_radio_variants/' . (int) $track->id);
        $variantPath = $variantDirectory . DIRECTORY_SEPARATOR . $signature . '-' . $bitrate . '.mp3';

        if (is_file($variantPath)) {
            return $variantPath;
        }

        $lockKey = 'public_radio_bitrate_variant:' . $track->id . ':' . $signature;

        return Cache::lock($lockKey, 120)->block(10, function () use ($variantDirectory, $variantPath, $sourcePath, $bitrate): string {
            if (is_file($variantPath)) {
                return $variantPath;
            }

            if (!is_dir($variantDirectory) && !mkdir($variantDirectory, 0775, true) && !is_dir($variantDirectory)) {
                throw new RuntimeException('Failed to create bitrate variant directory');
            }

            $tempPath = $variantPath . '.tmp'
                . bin2hex(random_bytes(4));

            $command = [
                'ffmpeg',
                '-y',
                '-v', 'error',
                '-nostdin',
                '-i', $sourcePath,
                '-vn',
                '-map', 'a:0',
                '-c:a', 'libmp3lame',
                '-b:a', $bitrate,
                '-f', 'mp3',
                $tempPath,
            ];

            $descriptors = [
                0 => ['pipe', 'r'],
                1 => ['pipe', 'w'],
                2 => ['pipe', 'w'],
            ];

            $process = proc_open($command, $descriptors, $pipes);
            if (!is_resource($process)) {
                throw new RuntimeException('Failed to start bitrate variant generator');
            }

            fclose($pipes[0]);
            $stdout = stream_get_contents($pipes[1]);
            $stderr = stream_get_contents($pipes[2]);
            fclose($pipes[1]);
            fclose($pipes[2]);
            $exitCode = proc_close($process);

            if ($exitCode !== 0 || !is_file($tempPath)) {
                @unlink($tempPath);
                throw new RuntimeException(trim((string) $stderr) !== '' ? trim((string) $stderr) : 'Bitrate variant generation failed');
            }

            if (!@rename($tempPath, $variantPath)) {
                @unlink($tempPath);
                throw new RuntimeException('Failed to finalize bitrate variant');
            }

            return $variantPath;
        });
    }

    /**
     * @return array{directory:string,master:string}
     */
    private function ensureHlsPackage(Track $track, string $sourcePath): array
    {
        if (!function_exists('proc_open')) {
            throw new RuntimeException('HLS packaging is not available in this runtime');
        }

        $signature = sha1(implode('|', [
            self::HLS_PACKAGE_VERSION,
            (string) $track->id,
            (string) (filesize($sourcePath) ?: 0),
            (string) (filemtime($sourcePath) ?: 0),
            implode(',', self::RADIO_BITRATE_LADDER),
            (string) self::HLS_SEGMENT_SECONDS,
        ]));

        $packageDirectory = storage_path('app/public_radio_hls/' . (int) $track->id . '/' . $signature);
        $masterPath = $packageDirectory . DIRECTORY_SEPARATOR . 'master.m3u8';

        if (is_file($masterPath)) {
            $this->publishCurrentHlsPackage($track, $packageDirectory);

            return [
                'directory' => $packageDirectory,
                'master' => $masterPath,
            ];
        }

        $lockKey = 'public_radio_hls_package:' . $track->id . ':' . $signature;

        return Cache::lock($lockKey, 600)->block(15, function () use ($track, $packageDirectory, $masterPath, $sourcePath): array {
            if (is_file($masterPath)) {
                $this->publishCurrentHlsPackage($track, $packageDirectory);

                return [
                    'directory' => $packageDirectory,
                    'master' => $masterPath,
                ];
            }

            if (!is_dir($packageDirectory) && !mkdir($packageDirectory, 0775, true) && !is_dir($packageDirectory)) {
                throw new RuntimeException('Failed to create HLS package directory');
            }

            foreach (self::RADIO_BITRATE_LADDER as $profile) {
                $profileDirectory = $packageDirectory . DIRECTORY_SEPARATOR . $profile;
                $playlistPath = $profileDirectory . DIRECTORY_SEPARATOR . 'index.m3u8';

                if (is_file($playlistPath)) {
                    continue;
                }

                if (!is_dir($profileDirectory) && !mkdir($profileDirectory, 0775, true) && !is_dir($profileDirectory)) {
                    throw new RuntimeException('Failed to create HLS profile directory');
                }

                $segmentPattern = $profileDirectory . DIRECTORY_SEPARATOR . 'segment_%03d.ts';
                $this->runFfmpegCommand([
                    'ffmpeg',
                    '-y',
                    '-v', 'error',
                    '-nostdin',
                    '-i', $sourcePath,
                    '-vn',
                    '-map', 'a:0',
                    '-c:a', 'aac',
                    '-b:a', $profile,
                    '-ac', '2',
                    '-ar', '44100',
                    '-f', 'hls',
                    '-hls_time', (string) self::HLS_SEGMENT_SECONDS,
                    '-hls_playlist_type', 'vod',
                    '-hls_flags', 'independent_segments',
                    '-hls_segment_filename', $segmentPattern,
                    $playlistPath,
                ], 'HLS profile generation failed');
            }

            $masterManifest = $this->buildHlsMasterManifest();
            if (@file_put_contents($masterPath, $masterManifest) === false) {
                throw new RuntimeException('Failed to write HLS master manifest');
            }

            $this->publishCurrentHlsPackage($track, $packageDirectory);

            return [
                'directory' => $packageDirectory,
                'master' => $masterPath,
            ];
        });
    }

    private function buildHlsMasterManifest(): string
    {
        $lines = ['#EXTM3U', '#EXT-X-VERSION:3'];

        foreach (self::RADIO_BITRATE_LADDER as $profile) {
            $bitrateKbps = (int) str_replace('k', '', $profile);
            $bandwidth = $bitrateKbps * 1024;
            $lines[] = sprintf('#EXT-X-STREAM-INF:BANDWIDTH=%d,CODECS="mp4a.40.2"', $bandwidth);
            $lines[] = $profile . '/index.m3u8';
        }

        return implode("\n", $lines) . "\n";
    }

    /**
     * @param array<int, string> $command
     */
    private function runFfmpegCommand(array $command, string $fallbackMessage): void
    {
        $descriptors = [
            0 => ['pipe', 'r'],
            1 => ['pipe', 'w'],
            2 => ['pipe', 'w'],
        ];

        $process = proc_open($command, $descriptors, $pipes);
        if (!is_resource($process)) {
            throw new RuntimeException($fallbackMessage);
        }

        fclose($pipes[0]);
        stream_get_contents($pipes[1]);
        fclose($pipes[1]);
        $stderr = stream_get_contents($pipes[2]);
        fclose($pipes[2]);
        $exitCode = proc_close($process);

        if ($exitCode !== 0) {
            throw new RuntimeException(trim((string) $stderr) !== '' ? trim((string) $stderr) : $fallbackMessage);
        }
    }

    private function publishCurrentHlsPackage(Track $track, string $packageDirectory): void
    {
        $trackDirectory = storage_path('app/public_radio_hls/' . (int) $track->id);
        $currentDirectory = $trackDirectory . DIRECTORY_SEPARATOR . 'current';

        if (is_link($currentDirectory) || is_file($currentDirectory)) {
            @unlink($currentDirectory);
        } elseif (is_dir($currentDirectory)) {
            $this->deleteDirectory($currentDirectory);
        }

        $relativeTarget = basename($packageDirectory);
        if (function_exists('symlink') && @symlink($relativeTarget, $currentDirectory)) {
            return;
        }

        $this->mirrorDirectory($packageDirectory, $currentDirectory);
    }

    private function mirrorDirectory(string $sourceDirectory, string $targetDirectory): void
    {
        if (!is_dir($targetDirectory) && !mkdir($targetDirectory, 0775, true) && !is_dir($targetDirectory)) {
            throw new RuntimeException('Failed to create static HLS mirror directory');
        }

        $items = scandir($sourceDirectory);
        if ($items === false) {
            throw new RuntimeException('Failed to read HLS package directory');
        }

        foreach ($items as $item) {
            if ($item === '.' || $item === '..') {
                continue;
            }

            $sourcePath = $sourceDirectory . DIRECTORY_SEPARATOR . $item;
            $targetPath = $targetDirectory . DIRECTORY_SEPARATOR . $item;

            if (is_dir($sourcePath)) {
                $this->mirrorDirectory($sourcePath, $targetPath);
                continue;
            }

            if (@copy($sourcePath, $targetPath) === false) {
                throw new RuntimeException('Failed to copy HLS package into static mirror');
            }
        }
    }

    private function deleteDirectory(string $directory): void
    {
        $items = scandir($directory);
        if ($items === false) {
            return;
        }

        foreach ($items as $item) {
            if ($item === '.' || $item === '..') {
                continue;
            }

            $path = $directory . DIRECTORY_SEPARATOR . $item;
            if (is_dir($path) && !is_link($path)) {
                $this->deleteDirectory($path);
                continue;
            }

            @unlink($path);
        }

        @rmdir($directory);
    }
}
