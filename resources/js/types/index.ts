/**
 * TypeScript types for AI Music Recommendations API
 * Requirements: 4.1, 4.2, 4.3, 11.1
 */

export interface TagScore {
    tag: string
    score?: number
    prob?: number
}

export interface ArtistCredit {
    name: string
    slug: string
    roles: string[]
}

export interface Track {
    id: number
    title: string
    artist: string | null
    artist_credits?: ArtistCredit[] | null
    artist_slugs?: string[] | null
    friend_likes?: Array<{
        id: number
        name: string
        email: string
        liked_at: string | null
    }>
    album: string | null
    year: number | null
    duration: number
    file_path: string
    source: 'upload' | 'youtube'
    youtube_url: string | null
    genre: string | null
    genre_tags: TagScore[] | null
    bpm: number | null
    key: string | null
    mood: string | null
    mood_tags: TagScore[] | null
    has_vocals: boolean | null
    is_acoustic: boolean | null
    loudness_integrated_lufs: number | null
    loudness_true_peak_db: number | null
    loudness_gain_db: number | null
    loudness_target_lufs: number | null
    is_liked: boolean
    manually_edited: boolean
    embedding_path: string | null
    analysis_status: 'pending' | 'processing' | 'analyzing' | 'completed' | 'failed'
    created_at: string
    updated_at: string
    fragments?: Fragment[]
}

export interface TrackLyricsCue {
    start_ms: number
    end_ms: number | null
    text: string
}

export interface TrackLyricsPayload {
    available: boolean
    kind: 'none' | 'plain_text' | 'timed_text'
    format: string | null
    text: string | null
    cues: TrackLyricsCue[]
    line_count: number
}

export interface TrackPlaybackMode {
    key: string
    label: string
    description: string
    kind: string
    recommended: boolean
    is_original: boolean
    file_name: string
    file_size_bytes: number
    mime_type: string | null
    extension: string | null
    stream_url: string
}

export interface TrackPlaybackStem {
    key: string
    label: string
    description: string
    family: string
    kind: string
    default_enabled: boolean
    file_name: string
    file_size_bytes: number
    mime_type: string | null
    extension: string | null
    stream_url: string
}

export interface TrackPlaybackManifest {
    track_id: number
    manifest_version: number
    default_mode_key: string
    modes: TrackPlaybackMode[]
    stems: TrackPlaybackStem[]
    lyrics: TrackLyricsPayload
    features: {
        karaoke: boolean
        stem_mixing: boolean
        lyrics: boolean
        synced_lyrics: boolean
    }
}

export interface AuthUser {
    id: number
    name: string
    email: string
    role: 'admin' | 'user'
    created_at: string | null
    updated_at: string | null
}

export interface AuthSession {
    token: string
    user: AuthUser
}

export interface AdminOverview {
    users: { total: number; admins: number; members: number }
    tracks: { total: number; completed: number; pending: number; processing: number; failed: number }
    ratings: { total: number; high: number; low: number }
    playlists: { total: number; ai_generated: number }
    queue: { pending: number; processing: number; completed: number; failed: number }
    model: {
        status: string
        trained_at: string | null
        model_path: string | null
        likes_count: number
        trained_users: number
        training_users: number
        stale_training_users: number
    }
    training_queue: { pending: number; processing: number; completed: number; failed: number }
    training_queue_drift: {
        healthy: boolean
        pending_db_count: number
        queued_redis_count: number
        missing_from_redis_count: number
        stale_in_redis_count: number
        pending_db_task_ids: number[]
        queued_redis_task_ids: number[]
        missing_from_redis_task_ids: number[]
        stale_in_redis_task_ids: number[]
    }
    training_scope_summary: {
        user_scoped: {
            tasks_pending: number
            tasks_processing: number
            tasks_completed: number
            tasks_failed: number
            runs_total: number
            latest_run_id: number | null
            latest_run_at: string | null
        }
        global_legacy: {
            tasks_pending: number
            tasks_processing: number
            tasks_completed: number
            tasks_failed: number
            runs_total: number
            latest_run_id: number | null
            latest_run_at: string | null
        }
    }
    user_models: Array<{
        user_id: number
        name: string
        email: string
        role: 'admin' | 'user'
        model_status: string
        model_trained_at: string | null
        model_path: string | null
        likes_count: number
        score_count: number
        liked_tracks: number
        liked_fragments: number
        disliked_fragments: number
        frequently_played_tracks: number
        rated_segments: number
        training_tasks_pending: number
        training_tasks_processing: number
        training_tasks_completed: number
        training_tasks_failed: number
        latest_training_run: null | {
            id: number
            queue_task_id: number | null
            queue_task_status: string | null
            model_path: string | null
            started_at: string | null
            finished_at: string | null
            final_loss: number | null
            positives_total: number
            explicit_negatives_total: number
        }
    }>
    recent_training_runs: Array<{
        id: number
        scope: 'user_scoped' | 'global_legacy'
        user_id: number | null
        user_name: string | null
        user_email: string | null
        queue_task_id: number | null
        queue_task_status: string | null
        model_path: string | null
        final_loss: number | null
        positives_total: number
        positives_tracks: number
        positives_fragments: number
        positives_segments: number
        explicit_negatives_total: number
        started_at: string | null
        finished_at: string | null
        created_at: string
    }>
    recent_training_tasks: Array<{
        id: number
        scope: 'user_scoped' | 'global_legacy'
        user_id: number | null
        user_name: string | null
        user_email: string | null
        status: string
        error_message: string | null
        created_at: string
        completed_at: string | null
    }>
    radio_diagnostics: {
        available: boolean
        log_files: string[]
        batches_seen: number
        throttled_batches: number
        total_events: number
        dropped_events: number
        unique_clients: number
        unique_tracks: number
        waiting_events: number
        stalled_events: number
        error_events: number
        fallback_events: number
        average_drift: number | null
        max_drift: number | null
        latest_logged_at: string | null
        top_events: Array<{
            event: string
            count: number
        }>
        recent_events: Array<{
            logged_at: string | null
            station_slug: string | null
            viewer_id: number | null
            client_id: string | null
            client_ip: string | null
            user_agent: string | null
            event: string
            timestamp: string | null
            playback_mode: string | null
            transport_preference: string | null
            selected_bitrate: string | null
            track_id: number | null
            drift: number | null
            playback_rate: number | null
            media_error_code: number | null
            waiting: boolean | null
            paused: boolean | null
            ready_state: number | null
            current_src: string | null
        }>
    }
}

export interface Fragment {
    id: number
    track_id: number
    start_time: number
    end_time: number
    name: string | null
    comment: string | null
    is_liked: boolean | null
    embedding_path: string | null
    created_at: string
    updated_at?: string | null
}

export interface PreferenceModelStatus {
    id: number
    status: 'not_trained' | 'training' | 'trained' | 'error'
    last_trained_at: string | null
    liked_tracks_count: number
    error_message: string | null
    created_at: string
    updated_at: string
}

export interface Recommendation {
    track: Track
    score: number
}

export interface ArtistProfile {
    artist: {
        name: string
        slug: string
    }
    primary_tracks: Track[]
    featured_tracks: Track[]
    remix_tracks: Track[]
}

export interface SimilarTrack {
    track: Track
    distance: number
}

export interface SimilarFragment {
    fragment: Fragment
    track: Track
    distance: number
}

export interface PaginatedResponse<T> {
    data: T[]
    current_page: number
    last_page: number
    per_page: number
    total: number
}

export interface TrackFilters {
    genre?: string
    mood?: string
    bpm_min?: number
    bpm_max?: number
    liked?: boolean
    status?: string
    search?: string
    sort_by?: string
    sort_order?: 'asc' | 'desc'
    page?: number
    per_page?: number
}

export interface UploadProgress {
    loaded: number
    total: number
    percentage: number
}

export interface YouTubeImportRequest {
    url: string
}

export interface YouTubeImportResponse {
    message: string
    task_id: number
    status: string
}

/**
 * AI Music Chat types
 * Requirements: AI Music Chat 4.1, 4.2, 4.3, 4.4
 */

export interface ChatMessage {
    id: string
    type: 'user' | 'system' | 'results' | 'error' | 'recommendations' | 'ai_tracks'
    content: string
    timestamp: Date
    results?: SearchResult[]
    tracks?: Track[]
    recommendations?: Array<{
        track: Track
        score: number
        reasons: string[]
        explanation: string
    }>
    query?: {
        trackId: number
        trackTitle: string
        startTime: number
        endTime: number
    }
    retryData?: {
        type: 'search' | 'recommendation'
        query?: string
        trackId?: number
        trackTitle?: string
        startTime?: number
        endTime?: number
    }
}

export interface SearchResult {
    fragment_id: number
    track_id: number
    track_title: string
    track_artist: string | null
    start_time: number
    end_time: number
    similarity: number
}

export interface ChatSearchRequest {
    track_id: number
    start_time: number
    end_time: number
    limit?: number
    min_similarity?: number
}

export interface ChatSearchResponse {
    query: {
        track_id: number
        start_time: number
        end_time: number
    }
    results: SearchResult[]
    total: number
    error?: string
}


/**
 * Pattern types for melodic pattern search
 */

export interface Pattern {
    id: number
    name: string
    description: string | null
    representation_type: 'pitch_contour' | 'chroma' | 'midi'
    transpose_invariant: boolean
    tempo_invariant: boolean
    match_threshold: number
    importance: number
    prototype_path: string | null
    created_at: string
    updated_at: string
    examples_count?: number
    matches_count?: number
    examples?: PatternExample[]
    matches?: PatternMatch[]
}

export interface PatternExample {
    id: number
    pattern_id: number
    track_id: number
    start_time: number
    end_time: number
    is_positive: boolean
    notes: string | null
    representation_path: string | null
    created_at: string
    updated_at: string
    track?: Track
}

export interface PatternMatch {
    id: number
    pattern_id: number
    track_id: number
    start_time: number
    end_time: number
    score: number
    confirmed: boolean | null
    created_at: string
    updated_at: string
    track?: Track
}

export interface StemTuningSource {
    key: string
    label: string
    description: string
    family: string
    kind: string
    editable: boolean
    recommended: boolean
    file_name: string
    file_size_bytes: number
    extension: string
    stream_url: string
    waveform_url: string
}

export interface StemTuningFocusWindow {
    id: string
    label: string
    start_sec: number
    end_sec: number
    kind: string
    notes: string
}

export interface StemTuningDetectorSelectedWindow {
    segmentId: string
    startSec: number
    endSec: number
    score: number
    segmentStartSec: number
    segmentEndSec: number
}

export interface StemTuningDetectorSegment {
    segmentId: string
    startSec: number
    endSec: number
    durationSec: number
    structureLabel: string
    kinds: string[]
    scores: {
        segmentPriority: number
        vocalActivityFraction: number
        vocalMeanPercentile: number
        vocalPeak: number
        accompanimentMeanPercentile: number
        accompanimentPeakPercentile: number
        accompanimentOnsetPercentile: number
        transitionLift: number
        transitionPercentile: number
        brightnessNormMean: number
    }
    metrics: {
        vocalMeanRms: number
        accompanimentMeanRms: number
        accompanimentOnsetMean: number
        brightnessMeanHz: number
    }
    window: {
        startSec: number
        endSec: number
        score: number
    }
}

export interface StemTuningDetectorSegments {
    track_id: number
    generated_at: string
    source: {
        structure_cache_path: string
        playback_manifest_path: string
        audio_path: string
        model_name: string
    }
    analysis: {
        duration_sec: number
        frame_sec: number
        window_sec: number
        sample_rate: number
        vocal_activity_threshold_rms: number
        accompaniment_activity_threshold_rms: number
    }
    segments: StemTuningDetectorSegment[]
    selected: {
        vocal_drop: StemTuningDetectorSelectedWindow[]
        drop_like: StemTuningDetectorSelectedWindow[]
    }
}

export interface StemTuningLayerState {
    enabled: boolean
    gainDb: number
    transientDb: number
    bodyDb: number
    harmonicDuckDb: number
    bleedDuckDb: number
}

export interface StemTuningProcessorConfig {
    bypass: boolean
    harmonicMargin: number
    percussiveMargin: number
    transientCutoffHz: number
    bodyLowHz: number
    bodyHighHz: number
    bleedLowHz: number
    bleedHighHz: number
}

export type StemTuningRenderMode = 'post_dsp' | 'hybrid_build' | 'stem_rewrite' | 'vocals_rewrite'

export interface StemTuningHybridBuildConfig {
    modelName: 'hybrid_htdemucs_ft_6s' | 'hybrid_htdemucs_ft_bs_roformer_6s'
    sharpenBleed: boolean
    includeMelodyStem: boolean
    aggressiveMelodyFirst: boolean
}

export interface StemTuningStemRewriteConfig {
    otherMove: number
    guitarMove: number
    pianoMove: number
    drumsMaskedMove: number
    targetMelodyRms: number
    maxMelodyGain: number
    stageDrumHarmonicMove: number
    stageHarmonicStemPercussiveMove: number
    stageCleanupDrumHarmonicMove: number
    stageCleanupHarmonicStemPercussiveMove: number
    harmonicCount: number
    confidenceFloor: number
}

export interface StemTuningVocalsRewriteConfig {
    taskMode: 'custom' | 'clean_vocals' | 'minus_from_vocals' | 'artifact_repair' | 'aggressive_vocal_suppression' | 'detail_preserve'
    vocalIsolationIntensity: number
    bleedRemovalIntensity: number
    airRestore: number
    bodyPreserve: number
    artifactSuppression: number
    centerVocalSuppression: number
    candidateMode: 'ft_vocals' | 'bs_vocals' | 'hybrid_vocals'
    lowBsWeight: number
    highBsWeight: number
    inactiveLowBsWeight: number
    inactiveHighBsWeight: number
    crossoverHz: number
    activityAbsFloorRatio: number
    activityRelFloor: number
    activitySmoothingFrames: number
    vocalSharpenAlpha: number
    tailGateStrength: number
    tailGateFloorRatio: number
    tailGateSmoothingFrames: number
    subtractiveStrength: number
    subtractivePower: number
    subtractiveResidualFloor: number
    subtractiveActiveRestore: number
    subtractiveTargetRmsRatio: number
    subtractiveMaxGain: number
    hardMaskThreshold: number
    hardMaskPower: number
    hardMaskResidualFloor: number
}

export interface StemTuningUnsafeHybridInternals {
    requestedDevice: string
    stage1Model: string
    stage2Model: string
    stage1RequestedDevice: string
    stage2RequestedDevice: string
    stage1Split: boolean
    stage1Progress: boolean
    stage1Shifts: number
    stage1Overlap: number
    stage1Segment: number
    stage2Split: boolean
    stage2Progress: boolean
    stage2Shifts: number
    stage2Overlap: number
    stage2Segment: number
}

export interface StemTuningUnsafeBsRoformerInternals {
    requestedDevice: string
    deviceIds: string[]
    forceCpu: boolean
    disableDetailedPbar: boolean
    repoPath: string
    configPath: string
    checkpointPath: string
    useTta: boolean
    extractInstrumental: boolean
    filenameTemplate: string
    configOverrideYaml: string
    extraCliArgs: string[]
}

export interface StemTuningUnsafeModelInternals {
    hybrid: StemTuningUnsafeHybridInternals
    bsRoformer: StemTuningUnsafeBsRoformerInternals
}

export interface StemTuningPreset {
    version: number
    sourceKey: string
    renderMode: StemTuningRenderMode
    layerStates: Record<string, StemTuningLayerState>
    hybridBuildConfig: StemTuningHybridBuildConfig
    stemRewriteConfig: StemTuningStemRewriteConfig
    vocalsRewriteConfig: StemTuningVocalsRewriteConfig
    unsafeModelInternals: StemTuningUnsafeModelInternals
    processorConfig: StemTuningProcessorConfig
    focusWindow: {
        enabled: boolean
        startSec: number
        endSec: number
        xfadeSec: number
        targetLayerKey: string
        layerGainDb: number
        otherLayersDuckDb: number
    }
}

export interface StemTuningLabSession {
    track: Track
    lab: {
        title: string
        description: string
        stem_key: string
        default_source_key: string
        sources: StemTuningSource[]
        focus_windows: StemTuningFocusWindow[]
        detector_segments?: StemTuningDetectorSegments | null
        default_preset: StemTuningPreset
    }
}

export interface StemTuningRenderTask {
    task_id: number
    status: 'pending' | 'processing' | 'completed' | 'cancelled' | 'failed'
    task_type?: 'stem_tuning_render' | 'stem_source_build' | string
    message?: string
    progress?: number
    error_message?: string | null
    completed_at?: string | null
    logs?: Array<{
        timestamp: string
        level: 'info' | 'error'
        progress: number
        message: string
        stage?: string | null
    }>
    payload?: {
        track_id: number
        render_id?: string
        build_id?: string
    }
}
