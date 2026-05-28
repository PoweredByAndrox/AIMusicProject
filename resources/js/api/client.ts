import axios from 'axios'
import type { AxiosInstance, AxiosProgressEvent } from 'axios'
import { AUTH_TOKEN_STORAGE_KEY } from '@/constants/auth'
import type {
    AdminOverview,
    ArtistProfile,
    AuthSession,
    Track,
    Fragment,
    PreferenceModelStatus,
    Recommendation,
    SimilarTrack,
    SimilarFragment,
    PaginatedResponse,
    TrackFilters,
    YouTubeImportRequest,
    YouTubeImportResponse,
    ChatSearchRequest,
    ChatSearchResponse,
    TrackPlaybackManifest,
    StemTuningLabSession,
    StemTuningRenderTask,
} from '@/types'

/**
 * API Client for AI Music Recommendations
 * Requirements: 1.1, 2.1, 4.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1
 */
class ApiClient {
    private client: AxiosInstance

    constructor() {
        this.client = axios.create({
            baseURL: '/api',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
        })

        this.client.interceptors.request.use((config) => {
            const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
            if (token) {
                config.headers = config.headers ?? {}
                config.headers.Authorization = `Bearer ${token}`
            }
            return config
        })
    }

    // Auth API
    async register(payload: {
        name: string
        email: string
        password: string
        password_confirmation: string
    }): Promise<AuthSession> {
        const { data } = await this.client.post('/auth/register', payload)
        return {
            token: data.token,
            user: data.user,
        }
    }

    async login(payload: { email: string; password: string }): Promise<AuthSession> {
        const { data } = await this.client.post('/auth/login', payload)
        return {
            token: data.token,
            user: data.user,
        }
    }

    async getCurrentUser(): Promise<{ user: import('@/types').AuthUser }> {
        const { data } = await this.client.get('/auth/me')
        return data
    }

    async logout(): Promise<void> {
        await this.client.post('/auth/logout')
    }

    async getAdminOverview(): Promise<AdminOverview> {
        const { data } = await this.client.get('/admin/overview')
        return data
    }

    async repairTrainingQueue(): Promise<{
        success: boolean
        message: string
        repair: {
            pending_db_task_ids: number[]
            queued_before_task_ids: number[]
            kept_pending_task_ids: number[]
            removed_stale_task_ids: number[]
            added_missing_task_ids: number[]
            remaining_training_task_ids: number[]
        }
    }> {
        const { data } = await this.client.post('/admin/training-queue/repair')
        return data
    }

    async retrainUserModel(userId: number): Promise<{
        success: boolean
        message: string
        task_id: number
        user_id: number
        stats: Record<string, number | string | null>
    }> {
        const { data } = await this.client.post(`/admin/users/${userId}/preference-model/retrain`)
        return data
    }

    async retryTrainingTask(taskId: number): Promise<{
        success: boolean
        message: string
        task_id: number
        user_id: number | null
        queue_cleanup: {
            removed: number
            remaining: number
        }
    }> {
        const { data } = await this.client.post(`/admin/training-tasks/${taskId}/retry`)
        return data
    }

    async cancelTrainingTask(taskId: number): Promise<{
        success: boolean
        message: string
        task_id: number
        user_id: number | null
        queue_cleanup: {
            removed: number
            remaining: number
        }
    }> {
        const { data } = await this.client.post(`/admin/training-tasks/${taskId}/cancel`)
        return data
    }

    // Tracks API
    async getTracks(filters?: TrackFilters): Promise<PaginatedResponse<Track>> {
        const { data } = await this.client.get('/tracks', { params: filters })
        return data
    }

    async getTrack(id: number): Promise<Track> {
        const { data } = await this.client.get(`/tracks/${id}`)
        return data
    }

    async getTrackPlaybackOptions(trackId: number): Promise<TrackPlaybackManifest> {
        const { data } = await this.client.get(`/tracks/${trackId}/playback-options`)
        return data
    }

    async getStemTuningLab(trackId: number): Promise<StemTuningLabSession> {
        const { data } = await this.client.get(`/tracks/${trackId}/tuning-lab`)
        return data
    }

    async buildStemTuningSources(trackId: number): Promise<StemTuningRenderTask> {
        const { data } = await this.client.post(`/tracks/${trackId}/tuning-lab/build-stems`)
        return data
    }

    async getStemTuningBuildStatus(trackId: number, taskId: number): Promise<StemTuningRenderTask> {
        const { data } = await this.client.get(`/tracks/${trackId}/tuning-lab/build-stem-tasks/${taskId}`)
        return data
    }

    async renderStemTuningLab(trackId: number, preset: Record<string, unknown>): Promise<StemTuningRenderTask> {
        const { data } = await this.client.post(`/tracks/${trackId}/tuning-lab/render`, { preset })
        return data
    }

    async getStemTuningRenderStatus(trackId: number, taskId: number): Promise<StemTuningRenderTask> {
        const { data } = await this.client.get(`/tracks/${trackId}/tuning-lab/render-tasks/${taskId}`)
        return data
    }

    async cancelStemTuningRender(trackId: number, taskId: number, reason?: string): Promise<StemTuningRenderTask> {
        const { data } = await this.client.post(`/tracks/${trackId}/tuning-lab/render-tasks/${taskId}/cancel`, {
            reason,
        })
        return data
    }

    async getArtist(slug: string): Promise<ArtistProfile> {
        const { data } = await this.client.get(`/artists/${slug}`)
        return data
    }

    async updateArtist(slug: string, name: string): Promise<{
        message: string
        artist: {
            name: string
            slug: string
        }
        updated_tracks: number
    }> {
        const { data } = await this.client.put(`/artists/${slug}`, { name })
        return data
    }

    async uploadTrack(
        file: File,
        onProgress?: (progress: number) => void
    ): Promise<Track> {
        const formData = new FormData()
        formData.append('audio', file)

        const { data } = await this.client.post('/tracks', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (event: AxiosProgressEvent) => {
                if (onProgress && event.total) {
                    onProgress(Math.round((event.loaded * 100) / event.total))
                }
            },
        })
        return data
    }

    async updateTrack(id: number, updates: Partial<Track>): Promise<Track> {
        const { data } = await this.client.put(`/tracks/${id}`, updates)
        return data
    }

    async deleteTrack(id: number): Promise<void> {
        await this.client.delete(`/tracks/${id}`)
    }

    async deleteTracks(ids: number[]): Promise<{
        message: string
        deleted_ids: number[]
        deleted_count: number
    }> {
        const { data } = await this.client.delete('/tracks/batch', {
            data: { ids },
        })
        return data
    }

    async retryTrackAnalysis(trackId: number): Promise<{
        success: boolean
        message: string
        track: Track
    }> {
        const { data } = await this.client.post(`/tracks/${trackId}/retry-analysis`)
        return data
    }

    async importYouTube(request: YouTubeImportRequest): Promise<YouTubeImportResponse> {
        const { data } = await this.client.post('/tracks/youtube', request)
        return data
    }

    async importYouTubeBatch(request: { urls: string }): Promise<{
        message: string
        queued: number
        invalid: number
        errors: string[]
        status: string
    }> {
        const { data } = await this.client.post('/tracks/youtube/batch', request)
        return data
    }

    async getQueueStatus(): Promise<{
        queue: { youtube: number; analysis: number; default: number; total: number }
        tracks: { pending: number; processing: number; completed: number; failed: number }
        recent: Array<{
            id: number
            title: string
            artist: string | null
            analysis_status: string
            created_at: string
            source: string
        }>
    }> {
        const { data } = await this.client.get('/tracks/queue-status')
        return data
    }

    // Fragments API
    async getFragments(trackId: number): Promise<Fragment[]> {
        const { data } = await this.client.get(`/tracks/${trackId}/fragments`)
        // API returns { fragments: [...], total: N }
        return data.fragments || []
    }

    async createFragment(
        trackId: number,
        fragment: { start_time: number; end_time: number; name?: string; comment?: string; is_liked?: boolean }
    ): Promise<Fragment> {
        const { data } = await this.client.post(`/tracks/${trackId}/fragments`, fragment)
        return data.fragment ?? data
    }

    async deleteFragment(fragmentId: number): Promise<void> {
        await this.client.delete(`/fragments/${fragmentId}`)
    }

    // Likes API
    async likeTrack(trackId: number): Promise<{
        message: string
        track_id: number
        is_liked: boolean
        effective_rating?: number | null
        explicit_track_rating?: number | null
        computed_track_rating?: number | null
        feedback?: {
            track_id: number
            track_rating?: number | null
            explicit_track_rating?: number | null
            computed_track_rating?: number | null
            preference_score?: number | null
            sentiment?: string | null
            fragments?: Array<{
                id: number
                track_id: number
                start_time: number
                end_time: number
                is_liked?: boolean | null
                name?: string | null
                comment?: string | null
            }>
            segments?: Array<{
                id: number
                start: number
                end: number
                type?: string | null
                display_label?: string | null
                importance?: number | null
                rating?: number | null
            }>
        } | null
        stats?: Record<string, number | string | null>
    }> {
        const { data } = await this.client.post(`/tracks/${trackId}/like`)
        return data
    }

    async unlikeTrack(trackId: number): Promise<{
        message: string
        track_id: number
        is_liked: boolean
        effective_rating?: number | null
        explicit_track_rating?: number | null
        computed_track_rating?: number | null
        feedback?: {
            track_id: number
            track_rating?: number | null
            explicit_track_rating?: number | null
            computed_track_rating?: number | null
            preference_score?: number | null
            sentiment?: string | null
            fragments?: Array<{
                id: number
                track_id: number
                start_time: number
                end_time: number
                is_liked?: boolean | null
                name?: string | null
                comment?: string | null
            }>
            segments?: Array<{
                id: number
                start: number
                end: number
                type?: string | null
                display_label?: string | null
                importance?: number | null
                rating?: number | null
            }>
        } | null
    }> {
        const { data } = await this.client.delete(`/tracks/${trackId}/like`)
        return data
    }

    async likeFragment(fragmentId: number): Promise<Fragment> {
        const { data } = await this.client.post(`/fragments/${fragmentId}/like`)
        return data
    }

    async unlikeFragment(fragmentId: number): Promise<Fragment> {
        const { data } = await this.client.delete(`/fragments/${fragmentId}/like`)
        return data
    }

    // Recommendations API
    async getRecommendations(filters?: TrackFilters): Promise<{
        recommendations: Recommendation[]
        ratings?: Record<number, number>
        segments?: Record<number, any[]>
        total?: number
        model_status?: string
    }> {
        const { data } = await this.client.get('/recommendations', { params: filters })
        return data
    }

    async getSimilarTracks(trackId: number, k?: number): Promise<SimilarTrack[]> {
        const { data } = await this.client.get(`/tracks/${trackId}/similar`, {
            params: { k },
        })
        return data
    }

    async getSimilarFragments(fragmentId: number, k?: number): Promise<SimilarFragment[]> {
        const { data } = await this.client.get(`/fragments/${fragmentId}/similar`, {
            params: { k },
        })
        return data
    }

    // Preference Model API
    async trainPreferenceModel(): Promise<{ message: string; task_id: number }> {
        const { data } = await this.client.post('/preference-model/train')
        return data
    }

    async getPreferenceModelStatus(): Promise<PreferenceModelStatus> {
        const { data } = await this.client.get('/preference-model/status')
        return data
    }

    // Export API
    async exportCsv(filters?: TrackFilters): Promise<Blob> {
        const { data } = await this.client.get('/export/csv', {
            params: filters,
            responseType: 'blob',
        })
        return data
    }

    // Smart Radio API
    async enableSmartRadio(seedTrackId: number): Promise<void> {
        await this.client.post('/radio/enable', { seed_track_id: seedTrackId })
    }

    async disableSmartRadio(): Promise<void> {
        await this.client.post('/radio/disable')
    }

    async sendRadioFeedback(trackId: number, feedbackType: 'more' | 'less'): Promise<void> {
        await this.client.post('/radio/feedback', {
            track_id: trackId,
            feedback_type: feedbackType,
        })
    }

    // AI DJ API
    async getCompatibleTracks(trackId: number): Promise<Array<{
        track: Track
        total_score: number
        bpm_score: number
        key_score: number
        energy_score: number
    }>> {
        const { data } = await this.client.get(`/dj/compatible/${trackId}`)
        return data
    }

    // Track Rating API
    async getTrackRating(trackId: number): Promise<{
        rating: number | null
        explicit_rating?: number | null
        computed_segment_rating?: number | null
        sentiment?: string | null
    }> {
        const { data } = await this.client.get(`/tracks/${trackId}/rating`)
        return data
    }

    async setTrackRating(
        trackId: number,
        ratingOrPayload: number | { rating?: number; sentiment?: 'dislike' | 'like' | 'love' | 'super_like' }
    ): Promise<{
        message: string
        rating: { rating: number } | number | null
        effective_rating?: number | null
        explicit_track_rating?: number | null
        computed_track_rating?: number | null
        sentiment?: string | null
        feedback?: {
            track_id: number
            track_rating?: number | null
            explicit_track_rating?: number | null
            computed_track_rating?: number | null
            preference_score?: number | null
            sentiment?: string | null
            segments?: Array<{
                id: number
                start: number
                end: number
                type?: string | null
                display_label?: string | null
                importance?: number | null
                rating?: number | null
            }>
        } | null
    }> {
        const payload = typeof ratingOrPayload === 'number'
            ? { rating: ratingOrPayload }
            : ratingOrPayload

        const { data } = await this.client.post(`/tracks/${trackId}/rating`, payload)
        return data
    }

    async getTrackFeedbackSnapshot(trackId: number): Promise<{
        track_id: number
        track_rating?: number | null
        explicit_track_rating?: number | null
        computed_track_rating?: number | null
        preference_score?: number | null
        sentiment?: string | null
        fragments?: Array<{
            id: number
            track_id: number
            start_time: number
            end_time: number
            is_liked?: boolean | null
            name?: string | null
            comment?: string | null
        }>
        segments?: Array<{
            id: number
            start: number
            end: number
            type?: string | null
            display_label?: string | null
            importance?: number | null
            is_liked?: boolean | null
            rating?: number | null
        }>
    }> {
        const { data } = await this.client.get(`/tracks/${trackId}/segments`)
        return data
    }

    async analyzeTrackSegments(trackId: number): Promise<{
        message: string
        track_id: number
        task_id: string
    }> {
        const { data } = await this.client.post(`/tracks/${trackId}/analyze-segments`)
        return data
    }

    async getTrackSegmentAnalysisStatus(trackId: number): Promise<{
        status: string
        track_id: number
        progress?: number
        message?: string
        segments_created?: number
        task_id?: string
    }> {
        const { data } = await this.client.get(`/tracks/${trackId}/analyze-segments/status`)
        return data
    }

    async rateSegment(segmentId: number, rating: number): Promise<{
        message: string
        segment: {
            id: number
            pattern_example_id: number
            track_id: number
            start_time: number
            end_time: number
            rating: number
            is_positive: boolean
        }
        track_rating?: number | null
        computed_track_rating?: number | null
        sentiment?: string | null
        feedback?: {
            track_id: number
            track_rating?: number | null
            explicit_track_rating?: number | null
            computed_track_rating?: number | null
            preference_score?: number | null
            sentiment?: string | null
            fragments?: Array<{
                id: number
                track_id: number
                start_time: number
                end_time: number
                is_liked?: boolean | null
                name?: string | null
                comment?: string | null
            }>
            segments?: Array<{
                id: number
                start: number
                end: number
                type?: string | null
                display_label?: string | null
                importance?: number | null
                is_liked?: boolean | null
                rating?: number | null
            }>
        } | null
    }> {
        const { data } = await this.client.post(`/segments/${segmentId}/rate`, { rating })
        return data
    }

    async deleteTrackRating(trackId: number): Promise<{
        message: string
        effective_rating?: number | null
        explicit_track_rating?: number | null
        computed_track_rating?: number | null
        sentiment?: string | null
        feedback?: {
            track_id: number
            track_rating?: number | null
            explicit_track_rating?: number | null
            computed_track_rating?: number | null
            preference_score?: number | null
            sentiment?: string | null
            fragments?: Array<{
                id: number
                track_id: number
                start_time: number
                end_time: number
                is_liked?: boolean | null
                name?: string | null
                comment?: string | null
            }>
            segments?: Array<{
                id: number
                start: number
                end: number
                type?: string | null
                display_label?: string | null
                importance?: number | null
                rating?: number | null
            }>
        } | null
    }> {
        const { data } = await this.client.delete(`/tracks/${trackId}/rating`)
        return data
    }

    async issueRadioAccessToken(): Promise<{
        access_token: string
        expires_at: string
        expires_in: number
    }> {
        const { data } = await this.client.post('/radio/access-token')
        return data
    }

    async getPersonalRadioSettings(): Promise<{
        enabled: boolean
        access_mode: 'all' | 'selected_users'
        allowed_user_ids: number[]
        available_users: Array<{
            id: number
            name: string
            email: string
        }>
        station_slug: string
        public_url: string
        owner: {
            id: number
            name: string
            email: string
        }
    }> {
        const { data } = await this.client.get('/account/personal-radio')
        return data
    }

    async updatePersonalRadioSettings(payload: {
        enabled: boolean
        access_mode: 'all' | 'selected_users'
        allowed_user_ids: number[]
    }): Promise<{
        enabled: boolean
        access_mode: 'all' | 'selected_users'
        allowed_user_ids: number[]
        available_users: Array<{
            id: number
            name: string
            email: string
        }>
        station_slug: string
        public_url: string
        owner: {
            id: number
            name: string
            email: string
        }
    }> {
        const { data } = await this.client.put('/account/personal-radio', payload)
        return data
    }

    // Playlist API
    async getPlaylists(): Promise<{ data: Array<{
        id: number
        name: string
        description: string | null
        is_ai_generated: boolean
        track_count: number
        total_duration: number
        created_at: string
    }> }> {
        const { data } = await this.client.get('/playlists')
        return data
    }

    async getPlaylist(playlistId: number): Promise<{
        id: number
        name: string
        description: string | null
        is_ai_generated: boolean
        track_count: number
        total_duration: number
        tracks: Track[]
    }> {
        const { data } = await this.client.get(`/playlists/${playlistId}`)
        return data
    }

    async createPlaylist(name: string, description?: string | null): Promise<{
        id: number
        name: string
        description: string | null
    }> {
        const { data } = await this.client.post('/playlists', { name, description })
        return data
    }

    async deletePlaylist(playlistId: number): Promise<void> {
        await this.client.delete(`/playlists/${playlistId}`)
    }

    async reorderPlaylistTracks(playlistId: number, trackOrder: Array<{ track_id: number; position: number }>): Promise<void> {
        await this.client.put(`/playlists/${playlistId}/reorder`, { track_order: trackOrder })
    }

    async removeTrackFromPlaylist(playlistId: number, trackId: number): Promise<void> {
        await this.client.delete(`/playlists/${playlistId}/tracks/${trackId}`)
    }

    async generateAIPlaylist(params: {
        name: string
        duration: number
        genre?: string
        mood?: string
        min_bpm?: number
        max_bpm?: number
        track_count?: number
    }): Promise<any> {
        const { data } = await this.client.post('/playlists/generate', params)
        return data
    }

    // AI Music Chat API
    async searchSimilarSegments(request: ChatSearchRequest): Promise<ChatSearchResponse> {
        const { data } = await this.client.post('/chat/search', request)
        return data
    }

    async getChatTrackInfo(trackId: number): Promise<{
        id: number
        title: string
        artist: string | null
        duration: number
        file_path: string
    } | null> {
        try {
            const { data } = await this.client.get(`/chat/track/${trackId}`)
            return data
        } catch {
            return null
        }
    }

    async getChatRecommendations(limit?: number): Promise<{
        recommendations: Array<{
            track: Track
            score: number
            reasons: string[]
            explanation: string
        }>
        total: number
        preference_summary?: {
            top_genres: string[]
            top_moods: string[]
            avg_bpm: number
            tracks_analyzed: number
        }
        has_preferences: boolean
        message?: string
    }> {
        const { data } = await this.client.get('/chat/recommend', { params: { limit } })
        return data
    }

    // LLM AI Chat API
    async aiChat(message: string, context?: { current_track?: Track }): Promise<{
        type: 'search' | 'recommendations' | 'similar' | 'stats' | 'info' | 'chat'
        message: string
        tracks: Track[]
        query?: Record<string, any>
        stats?: Record<string, number>
        error?: string
    }> {
        const { data } = await this.client.post('/chat/ai', { message, context })
        return data
    }

    async checkAIHealth(): Promise<{
        status: 'ok' | 'error'
        ollama_url?: string
        model?: string
        model_loaded?: boolean
        available_models?: string[]
        message?: string
    }> {
        const { data } = await this.client.get('/chat/ai/health')
        return data
    }

    // Pattern API
    async getPatterns(): Promise<import('@/types').Pattern[]> {
        const { data } = await this.client.get('/patterns')
        return data
    }

    async getPattern(id: number): Promise<import('@/types').Pattern> {
        const { data } = await this.client.get(`/patterns/${id}`)
        return data
    }

    async createPattern(pattern: {
        name: string
        description?: string
        representation_type?: 'pitch_contour' | 'chroma' | 'midi'
        transpose_invariant?: boolean
        tempo_invariant?: boolean
        match_threshold?: number
        importance?: number
    }): Promise<import('@/types').Pattern> {
        const { data } = await this.client.post('/patterns', pattern)
        return data
    }

    async updatePattern(id: number, updates: Partial<import('@/types').Pattern>): Promise<import('@/types').Pattern> {
        const { data } = await this.client.put(`/patterns/${id}`, updates)
        return data
    }

    async deletePattern(id: number): Promise<void> {
        await this.client.delete(`/patterns/${id}`)
    }

    async addPatternExample(patternId: number, example: {
        track_id: number
        start_time: number
        end_time: number
        is_positive?: boolean
        notes?: string
    }): Promise<import('@/types').PatternExample> {
        const { data } = await this.client.post(`/patterns/${patternId}/examples`, example)
        return data
    }

    async removePatternExample(patternId: number, exampleId: number): Promise<void> {
        await this.client.delete(`/patterns/${patternId}/examples/${exampleId}`)
    }

    async searchPattern(patternId: number): Promise<{ message: string; task_id: number }> {
        const { data } = await this.client.post(`/patterns/${patternId}/search`)
        return data
    }

    async getPatternMatches(patternId: number): Promise<import('@/types').PatternMatch[]> {
        const { data } = await this.client.get(`/patterns/${patternId}/matches`)
        return data
    }

    async confirmPatternMatch(patternId: number, matchId: number, confirmed: boolean): Promise<import('@/types').PatternMatch> {
        const { data } = await this.client.put(`/patterns/${patternId}/matches/${matchId}/confirm`, { confirmed })
        return data
    }

    async extractChroma(trackId: number): Promise<{ message: string; task_id: number }> {
        const { data } = await this.client.post(`/tracks/${trackId}/extract-chroma`)
        return data
    }

    async extractPitch(trackId: number): Promise<{ message: string; task_id: number }> {
        const { data } = await this.client.post(`/tracks/${trackId}/extract-pitch`)
        return data
    }
}

export const api = new ApiClient()
export default api
