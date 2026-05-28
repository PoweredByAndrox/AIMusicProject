<?php

use App\Http\Controllers\ExportController;
use App\Http\Controllers\FriendController;
use App\Http\Controllers\FragmentController;
use App\Http\Controllers\ArtistController;
use App\Http\Controllers\LikeController;
use App\Http\Controllers\ListeningHistoryController;
use App\Http\Controllers\PlaylistController;
use App\Http\Controllers\AuthController;
use App\Http\Controllers\AdminController;
use App\Http\Controllers\PreferenceModelController;
use App\Http\Controllers\PublicRadioController;
use App\Http\Controllers\RadioAccessController;
use App\Http\Controllers\RecommendationController;
use App\Http\Controllers\SmartRadioController;
use App\Http\Controllers\StemTuningLabController;
use App\Http\Controllers\TrackController;
use App\Http\Controllers\TrackRatingController;
use App\Http\Controllers\UserTrackPreferenceController;
use App\Http\Controllers\ChatController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "api" middleware group. Make something great!
|
| AI Music Recommendations API
| Requirements: 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1
|
*/

Route::middleware('api')->group(function () {
    
    /*
    |--------------------------------------------------------------------------
    | Health Check
    |--------------------------------------------------------------------------
    */
    Route::get('/health', function () {
        return response()->json([
            'status' => 'ok',
            'timestamp' => now()->toIso8601String(),
        ]);
    });

    Route::post('auth/register', [AuthController::class, 'register']);
    Route::post('auth/login', [AuthController::class, 'login']);
    Route::get('tracks/{id}/stream', [TrackController::class, 'stream']);
    Route::get('tracks/{id}/playback-options', [TrackController::class, 'playbackOptions']);
    Route::get('tracks/{id}/playback/modes/{modeKey}/stream', [TrackController::class, 'playbackModeStream']);
    Route::get('tracks/{id}/playback/stems/{stemKey}/stream', [TrackController::class, 'playbackStemStream']);
    Route::get('tracks/{id}/waveform', [TrackController::class, 'waveform']);
    Route::get('tracks/{id}/hls/master.m3u8', [TrackController::class, 'hlsMaster']);
    Route::get('tracks/{id}/hls/{profile}/index.m3u8', [TrackController::class, 'hlsVariant']);
    Route::get('tracks/{id}/hls/{profile}/{segment}', [TrackController::class, 'hlsSegment']);
    Route::put('radio/personal-stations/{slug}/genres', [RadioAccessController::class, 'updatePersonalStationGenres']);
    Route::get('public/radio/{slug}', [PublicRadioController::class, 'show']);
    Route::get('public/radio/{slug}/history', [PublicRadioController::class, 'history']);
    Route::post('public/radio/{slug}/diagnostics', [PublicRadioController::class, 'diagnostics']);
    Route::get('radio/private-overlay/{track}', [RadioAccessController::class, 'privateOverlay']);

    Route::middleware('auth.token')->group(function () {
        Route::get('auth/me', [AuthController::class, 'me']);
        Route::post('auth/logout', [AuthController::class, 'logout']);

        /*
        |--------------------------------------------------------------------------
        | Tracks API
        |--------------------------------------------------------------------------
        | Requirements: 1.1, 2.1, 4.1, 11.1
        */
        // Queue status MUST be before apiResource to avoid {id} matching
        Route::get('tracks/queue-status', [TrackController::class, 'queueStatus']);
        Route::delete('tracks/batch', [TrackController::class, 'destroyBatch']);
        Route::post('tracks/{track}/retry-analysis', [TrackController::class, 'retryAnalysis']);
        Route::apiResource('tracks', TrackController::class)->except(['show']);
        Route::get('tracks/{id}', [TrackController::class, 'show']);
        Route::get('tracks/{track}/tuning-lab', [StemTuningLabController::class, 'show']);
        Route::post('tracks/{track}/tuning-lab/build-stems', [StemTuningLabController::class, 'buildStems']);
        Route::get('tracks/{track}/tuning-lab/build-stem-tasks/{taskId}', [StemTuningLabController::class, 'buildStatus']);
        Route::post('tracks/{track}/tuning-lab/render', [StemTuningLabController::class, 'render']);
        Route::get('tracks/{track}/tuning-lab/render-tasks/{taskId}', [StemTuningLabController::class, 'renderStatus']);
        Route::post('tracks/{track}/tuning-lab/render-tasks/{taskId}/cancel', [StemTuningLabController::class, 'cancelRender']);
        Route::get('tracks/{track}/tuning-lab/sources/{sourceKey}/stream', [StemTuningLabController::class, 'sourceStream']);
        Route::get('tracks/{track}/tuning-lab/sources/{sourceKey}/waveform', [StemTuningLabController::class, 'sourceWaveform']);
        Route::get('artists/{slug}', [ArtistController::class, 'show']);
        Route::put('artists/{slug}', [ArtistController::class, 'update']);
        Route::post('tracks/youtube', [TrackController::class, 'importYoutube']);
        Route::post('tracks/youtube/batch', [TrackController::class, 'importYoutubeBatch']);

        /*
        |--------------------------------------------------------------------------
        | Fragments API
        |--------------------------------------------------------------------------
        | Requirements: 6.1, 6.2, 6.5
        */
        Route::apiResource('tracks.fragments', FragmentController::class)->shallow();

        /*
        |--------------------------------------------------------------------------
        | Likes API
        |--------------------------------------------------------------------------
        | Requirements: 7.1, 7.2, 7.3
        */
        Route::post('tracks/{track}/like', [LikeController::class, 'likeTrack']);
        Route::delete('tracks/{track}/like', [LikeController::class, 'unlikeTrack']);
        Route::post('fragments/{fragment}/like', [LikeController::class, 'likeFragment']);
        Route::delete('fragments/{fragment}/like', [LikeController::class, 'unlikeFragment']);
        Route::get('likes', [LikeController::class, 'index']);

        /*
        |--------------------------------------------------------------------------
        | Recommendations API
        |--------------------------------------------------------------------------
        | Requirements: 9.1, 10.1
        */
        Route::get('recommendations', [RecommendationController::class, 'getRecommendations']);
        Route::get('tracks/{track}/similar', [RecommendationController::class, 'findSimilarTracks']);
        Route::get('fragments/{fragment}/similar', [RecommendationController::class, 'findSimilarFragments']);

        /*
        |--------------------------------------------------------------------------
        | Preference Model API
        |--------------------------------------------------------------------------
        | Requirements: 8.1, 8.5
        */
        Route::post('preference-model/train', [PreferenceModelController::class, 'train']);
        Route::get('preference-model/status', [PreferenceModelController::class, 'status']);

        /*
        |--------------------------------------------------------------------------
        | Export API
        |--------------------------------------------------------------------------
        | Requirements: 12.1
        */
        Route::get('export/csv', [ExportController::class, 'exportCsv']);

        /*
        |--------------------------------------------------------------------------
        | Playlists API
        |--------------------------------------------------------------------------
        | Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
        */
        Route::apiResource('playlists', PlaylistController::class);
        Route::post('playlists/{playlist}/tracks', [PlaylistController::class, 'addTrack']);
        Route::delete('playlists/{playlist}/tracks/{track}', [PlaylistController::class, 'removeTrack']);
        Route::put('playlists/{playlist}/tracks/reorder', [PlaylistController::class, 'reorderTracks']);
        Route::post('playlists/generate', [PlaylistController::class, 'generate']);

        /*
        |--------------------------------------------------------------------------
        | Track Ratings API (5-star)
        |--------------------------------------------------------------------------
        | Requirements: 11.1, 11.3, 14.3
        */
        Route::post('tracks/{track}/rating', [TrackRatingController::class, 'store']);
        Route::get('tracks/{track}/rating', [TrackRatingController::class, 'show']);
        Route::delete('tracks/{track}/rating', [TrackRatingController::class, 'destroy']);
        Route::get('tracks/{track}/rating/history', [TrackRatingController::class, 'history']);

        /*
        |--------------------------------------------------------------------------
        | Fragment Rating & Comments API
        |--------------------------------------------------------------------------
        | Requirements: 7.1, 7.2, 7.3, 7.4, 8.1, 8.3, 8.4
        */
        Route::post('tracks/{track}/fragments/rate', [FragmentController::class, 'rate']);
        Route::put('fragments/{fragment}/rating', [FragmentController::class, 'updateRating']);
        Route::delete('fragments/{fragment}/rating', [FragmentController::class, 'destroyRating']);
        Route::post('fragments/{fragment}/comment', [FragmentController::class, 'addComment']);
        Route::put('fragments/{fragment}/comment', [FragmentController::class, 'updateComment']);
        Route::delete('fragments/{fragment}/comment', [FragmentController::class, 'destroyComment']);

        /*
        |--------------------------------------------------------------------------
        | User Track Preferences API
        |--------------------------------------------------------------------------
        | Requirements: 4.1, 4.2
        */
        Route::get('tracks/{track}/preferences', [UserTrackPreferenceController::class, 'show']);
        Route::put('tracks/{track}/preferences', [UserTrackPreferenceController::class, 'update']);
        Route::delete('tracks/{track}/preferences', [UserTrackPreferenceController::class, 'destroy']);

        /*
        |--------------------------------------------------------------------------
        | Listening History & Skip Tracking API
        |--------------------------------------------------------------------------
        | Requirements: 16.3, 16.4, 18.1, 18.2
        */
        Route::post('tracks/{track}/play', [ListeningHistoryController::class, 'recordPlay']);
        Route::post('tracks/{track}/skip', [ListeningHistoryController::class, 'recordSkip']);
        Route::get('tracks/{track}/skip-stats', [ListeningHistoryController::class, 'getSkipStats']);
        Route::get('listening-history', [ListeningHistoryController::class, 'getRecentHistory']);
        Route::put('listening-history/{history}/complete', [ListeningHistoryController::class, 'recordComplete']);

        /*
        |--------------------------------------------------------------------------
        | Smart Radio API
        |--------------------------------------------------------------------------
        | Requirements: 15.1, 15.2, 17.1
        */
        Route::post('radio/enable', [SmartRadioController::class, 'enable']);
        Route::post('radio/disable', [SmartRadioController::class, 'disable']);
        Route::get('radio/status', [SmartRadioController::class, 'status']);
        Route::get('radio/recommendations', [SmartRadioController::class, 'recommendations']);
        Route::post('radio/feedback', [SmartRadioController::class, 'feedback']);
        Route::post('radio/access-token', [RadioAccessController::class, 'issueAccessToken']);
        Route::get('account/personal-radio', [RadioAccessController::class, 'personalSettings']);
        Route::put('account/personal-radio', [RadioAccessController::class, 'updatePersonalSettings']);

        /*
        |--------------------------------------------------------------------------
        | Friends & Direct Chat API
        |--------------------------------------------------------------------------
        */
        Route::get('friends/state', [FriendController::class, 'state']);
        Route::get('friends/search', [FriendController::class, 'search']);
        Route::post('friends/requests', [FriendController::class, 'sendRequest']);
        Route::post('friends/requests/{friendship}/respond', [FriendController::class, 'respond']);
        Route::get('friends/conversations/{user}', [FriendController::class, 'conversation']);
        Route::post('friends/messages', [FriendController::class, 'sendMessage']);
        Route::post('friends/conversations/{user}/read', [FriendController::class, 'markConversationRead']);

        /*
        |--------------------------------------------------------------------------
        | AI DJ API
        |--------------------------------------------------------------------------
        | Requirements: 27.6, 28.5
        */
        Route::get('dj/compatible/{trackId}', [\App\Http\Controllers\AIDJController::class, 'compatible']);

        /*
        |--------------------------------------------------------------------------
        | AI Music Chat API
        |--------------------------------------------------------------------------
        | Requirements: AI Music Chat 6.1, 6.2
        */
        Route::post('chat/search', [ChatController::class, 'search']);
        Route::get('chat/track/{id}', [ChatController::class, 'getTrack']);
        Route::get('chat/recommend', [ChatController::class, 'recommend']);
        Route::post('chat/ai', [ChatController::class, 'aiChat']);
        Route::get('chat/ai/health', [ChatController::class, 'aiHealth']);
        Route::post('debug/frontend-logs', [\App\Http\Controllers\DebugController::class, 'frontendLogs']);

        /*
        |--------------------------------------------------------------------------
        | Pattern Search API
        |--------------------------------------------------------------------------
        | Melodic pattern search and personalization
        */
        Route::apiResource('patterns', \App\Http\Controllers\PatternController::class);
        Route::post('patterns/{pattern}/examples', [\App\Http\Controllers\PatternController::class, 'addExample']);
        Route::delete('patterns/{pattern}/examples/{example}', [\App\Http\Controllers\PatternController::class, 'removeExample']);
        Route::post('patterns/{pattern}/search', [\App\Http\Controllers\PatternController::class, 'search']);
        Route::get('patterns/{pattern}/matches', [\App\Http\Controllers\PatternController::class, 'matches']);
        Route::put('patterns/{pattern}/matches/{match}/confirm', [\App\Http\Controllers\PatternController::class, 'confirmMatch']);
        Route::post('tracks/{track}/extract-chroma', [\App\Http\Controllers\PatternController::class, 'extractChroma']);
        Route::post('tracks/{track}/extract-pitch', [\App\Http\Controllers\PatternController::class, 'extractPitch']);

        /*
        |--------------------------------------------------------------------------
        | Segment Labeling API
        |--------------------------------------------------------------------------
        | For ML training - segment rating and labeling
        */
        Route::get('tracks/{track}/segments', [\App\Http\Controllers\PatternController::class, 'getTrackSegments']);
        Route::post('tracks/{track}/analyze-segments', [\App\Http\Controllers\PatternController::class, 'analyzeSegments']);
        Route::get('tracks/{track}/analyze-segments/status', [\App\Http\Controllers\PatternController::class, 'getAnalysisStatus']);
        Route::post('segments/{segment}/rate', [\App\Http\Controllers\PatternController::class, 'rateSegment']);
    });

    /*
    |--------------------------------------------------------------------------
    | Queue Status SSE
    |--------------------------------------------------------------------------
    */
    Route::middleware(['auth.token', 'role:admin'])->group(function () {
        Route::get('queue-status/stream', [\App\Http\Controllers\QueueStatusController::class, 'stream']);
        Route::get('queue-status/failed', [\App\Http\Controllers\QueueStatusController::class, 'failedTasks']);
        Route::post('queue-status/retry/{taskId}', [\App\Http\Controllers\QueueStatusController::class, 'retryTask']);
        Route::post('queue-status/retry-all', [\App\Http\Controllers\QueueStatusController::class, 'retryAllFailed']);
        Route::delete('queue-status/task/{taskId}', [\App\Http\Controllers\QueueStatusController::class, 'deleteTask']);
        Route::delete('queue-status/clear-failed', [\App\Http\Controllers\QueueStatusController::class, 'clearAllFailed']);
    });

    /*
    |--------------------------------------------------------------------------
    | Debug API (Development)
    |--------------------------------------------------------------------------
    */
    Route::middleware(['auth.token', 'role:admin'])->group(function () {
        Route::get('admin/overview', [AdminController::class, 'overview']);
        Route::post('admin/training-queue/repair', [AdminController::class, 'repairTrainingQueue']);
        Route::post('admin/users/{user}/preference-model/retrain', [AdminController::class, 'retrainUserModel']);
        Route::post('admin/training-tasks/{taskId}/retry', [AdminController::class, 'retryTrainingTask']);
        Route::post('admin/training-tasks/{taskId}/cancel', [AdminController::class, 'cancelTrainingTask']);
        Route::get('debug/tracks', [\App\Http\Controllers\DebugController::class, 'trackStats']);
        Route::get('debug/system', [\App\Http\Controllers\DebugController::class, 'systemStats']);
        Route::get('debug/tracks/{trackId}', [\App\Http\Controllers\DebugController::class, 'trackDetail']);
    });

});
