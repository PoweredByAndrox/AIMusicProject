<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { toast } from 'vue3-toastify'
import api from '@/api/client'
import RatingWidget from '@/components/RatingWidget.vue'
import WaveformWithFragments from '@/components/WaveformWithFragments.vue'
import { usePlayerStore } from '@/stores/playerStore'
import type { Fragment, PreferenceModelStatus, Recommendation, Track, TrackFilters } from '@/types'

interface Segment {
  id: number
  start: number
  end: number
  type: string
  display_label?: string
  importance: number
  rating: number | null
}

interface TrackFeedbackSnapshot {
  track_id: number
  track_rating?: number | null
  explicit_track_rating?: number | null
  computed_track_rating?: number | null
  preference_score?: number | null
  sentiment?: string | null
  fragments?: Fragment[]
  segments?: Segment[]
}

const WAVEFORM_SEGMENT_ID_OFFSET = 1000000000

const router = useRouter()
const playerStore = usePlayerStore()

const recommendations = ref<Recommendation[]>([])
const modelStatus = ref<PreferenceModelStatus | null>(null)
const loading = ref(false)
const training = ref(false)
const trackRatings = ref<Record<number, number | null>>({})

const trackSegments = ref<Record<number, Segment[]>>({})
const trackFragments = ref<Record<number, Fragment[]>>({})
const analyzingTracks = ref<Record<number, boolean>>({})
const analysisPolls = new Map<number, ReturnType<typeof setInterval>>()
const expandedWaveformTrackId = ref<number | null>(null)
const waveformRef = ref<InstanceType<typeof WaveformWithFragments> | null>(null)
const waveformLocalTime = ref(0)

const hoverStars = ref<Record<string, number>>({})

const currentSegmentByTrack = computed(() => {
    const result: Record<number, Segment | null> = {}
    if (!playerStore.currentTrack || !playerStore.isPlaying) return result

    const trackId = playerStore.currentTrack.id
    const segments = trackSegments.value[trackId] || []
    const time = playerStore.currentTime
    const currentSeg = segments.find((seg) => time >= seg.start && time < seg.end) || null

    result[trackId] = currentSeg
    return result
})

const expandedTrackInGlobalPlayer = computed(() => (
    expandedWaveformTrackId.value !== null && playerStore.currentTrack?.id === expandedWaveformTrackId.value
))

const filters = ref<TrackFilters>({
    genre: undefined,
    mood: undefined,
    bpm_min: undefined,
    bpm_max: undefined,
})

const genres = [
    'Rock', 'Pop', 'Jazz', 'Classical', 'Hip-Hop', 'Metal', 'Folk',
    'R&B', 'Country', 'Blues', 'Reggae', 'Punk', 'Soul', 'Indie',
    'House', 'Techno', 'Trance', 'Dubstep', 'Drum & Bass',
    'Ambient', 'EDM', 'Electro', 'Trap', 'Future Bass', 'Synthwave',
    'Lo-Fi', 'Deep House', 'Tech House', 'Hardstyle', 'Downtempo',
]
const moods = ['Happy', 'Sad', 'Energetic', 'Calm', 'Aggressive', 'Romantic']

const stats = computed(() => modelStatus.value?.stats || {
    liked_tracks: 0, liked_fragments: 0, high_rated_tracks: 0,
    frequently_played: 0, total_positive: 0, high_rated_segments: 0,
    training_scope: 'global_feedback',
})

const canTrain = computed(() => stats.value.total_positive >= 1)

const recommendationStats = computed(() => ([
    {
        label: 'Positive signals',
        value: stats.value.total_positive.toString(),
        detail: 'available for model training',
    },
    {
        label: 'High-rated tracks',
        value: stats.value.high_rated_tracks.toString(),
        detail: 'tracks rated 4 stars or above',
    },
    {
        label: 'Segment ratings',
        value: String(stats.value.high_rated_segments || 0),
        detail: 'segment feedback retained by the model',
    },
    {
        label: 'Recommendations',
        value: recommendations.value.length.toString(),
        detail: modelStatus.value?.status === 'trained' ? 'current visible results' : 'awaiting training',
    },
]))

const loadRecommendations = async () => {
    loading.value = true
    try {
        const response = await api.getRecommendations(filters.value)
        recommendations.value = response.recommendations || []

        if (response.ratings) {
            trackRatings.value = response.ratings
            Object.entries(response.ratings).forEach(([trackId, rating]) => {
                playerStore.rememberTrackRating(Number(trackId), rating)
            })
        }

        if (response.segments) {
            trackSegments.value = response.segments
        }
    } catch {
        recommendations.value = []
    } finally {
        loading.value = false
    }
}

const loadModelStatus = async () => {
    try {
        modelStatus.value = await api.getPreferenceModelStatus()
    } catch (error) {
        console.error('Failed to load model status:', error)
    }
}

const trainModel = async () => {
    training.value = true
    toast.info('Starting model training...')
    try {
        await api.trainPreferenceModel()
        let pollCount = 0
        const maxPolls = 150
        const pollStatus = setInterval(async () => {
            pollCount++
            await loadModelStatus()
            if (modelStatus.value?.status !== 'training' || pollCount >= maxPolls) {
                clearInterval(pollStatus)
                training.value = false
                if (modelStatus.value?.status === 'trained') {
                    toast.success('Model training complete.')
                    await loadRecommendations()
                    await playerStore.refreshRadioQueue(true)
                } else if (pollCount >= maxPolls) {
                    toast.error('Training took too long. Check the status again in a moment.')
                }
            }
        }, 2000)
    } catch (error: any) {
        toast.error(error.response?.data?.message || 'Training failed')
        training.value = false
    }
}

const toggleLike = async (rec: Recommendation) => {
    try {
        const response = rec.track.is_liked
            ? await api.unlikeTrack(rec.track.id)
            : await api.likeTrack(rec.track.id)
        const feedback = response.feedback as TrackFeedbackSnapshot | null | undefined

        rec.track.is_liked = response.is_liked

        if (typeof feedback?.track_rating === 'number') {
            trackRatings.value[rec.track.id] = feedback.track_rating
        } else if (typeof response.effective_rating === 'number') {
            trackRatings.value[rec.track.id] = response.effective_rating
        } else {
            trackRatings.value[rec.track.id] = null
        }
        playerStore.rememberTrackRating(rec.track.id, trackRatings.value[rec.track.id])
        void playerStore.enforceSkipDislikedTracks(rec.track.id)

        if (Array.isArray(feedback?.segments)) {
            trackSegments.value[rec.track.id] = feedback.segments
        }
        if (Array.isArray(feedback?.fragments)) {
            trackFragments.value = {
                ...trackFragments.value,
                [rec.track.id]: feedback.fragments,
            }
        }

        await loadModelStatus()
    } catch {
        toast.error('Like update failed')
    }
}

const rateTrack = async (trackId: number, rating: number) => {
    try {
        const response = await api.setTrackRating(trackId, rating)
        const feedback = response.feedback as TrackFeedbackSnapshot | null | undefined
        const effectiveRating = typeof feedback?.track_rating === 'number'
            ? feedback.track_rating
            : (typeof response.effective_rating === 'number' ? response.effective_rating : rating)
        const explicitRating = typeof feedback?.explicit_track_rating === 'number'
            ? feedback.explicit_track_rating
            : (typeof response.explicit_track_rating === 'number' ? response.explicit_track_rating : null)

        trackRatings.value[trackId] = effectiveRating
        playerStore.rememberTrackRating(trackId, effectiveRating)
        void playerStore.enforceSkipDislikedTracks(trackId)
        if (Array.isArray(feedback?.segments)) {
            trackSegments.value[trackId] = feedback.segments
        }
        if (Array.isArray(feedback?.fragments)) {
            trackFragments.value = {
                ...trackFragments.value,
                [trackId]: feedback.fragments,
            }
        }

        const recommendation = recommendations.value.find((entry) => entry.track.id === trackId)
        if (recommendation) {
            recommendation.track.is_liked = explicitRating === 5
        }

        toast.success(`Track rated ${rating} star(s).`)

        await loadModelStatus()
    } catch {
        toast.error('Track rating failed')
    }
}

const openTrack = (trackId: number) => {
    router.push({ name: 'track', params: { id: trackId } })
}

const playTrack = (track: Track) => {
    if (playerStore.currentTrack?.id === track.id) {
        playerStore.play()
        return
    }

    playerStore.playTrack(track, false)
}

const getTrackAudioUrl = (trackId: number): string => `/api/tracks/${trackId}/stream`

const toggleWaveform = (trackId: number) => {
    const nextTrackId = expandedWaveformTrackId.value === trackId ? null : trackId
    expandedWaveformTrackId.value = nextTrackId
    waveformLocalTime.value = 0
    if (nextTrackId !== null) {
        void loadTrackFragments(nextTrackId)
    }
}

const getSegmentDisplayLabel = (segment: Segment): string => {
    return segment.display_label || segment.type.replace(/_/g, ' ').toUpperCase()
}

const getWaveformFragments = (trackId: number) => {
    const savedFragments = trackFragments.value[trackId] || []
    const autoSegments = (trackSegments.value[trackId] || []).map((seg) => ({
        id: seg.id + WAVEFORM_SEGMENT_ID_OFFSET,
        track_id: trackId,
        start_time: seg.start,
        end_time: seg.end,
        is_liked: seg.rating === null ? null : seg.rating >= 4,
        name: getSegmentDisplayLabel(seg),
        comment: seg.rating === null
            ? `${getSegmentDisplayLabel(seg)} (${formatTime(seg.start)} - ${formatTime(seg.end)})`
            : `${getSegmentDisplayLabel(seg)} • ${seg.rating}★`,
    }))

    return [...savedFragments, ...autoSegments].sort((left, right) => left.start_time - right.start_time)
}

const loadTrackFragments = async (trackId: number) => {
    try {
        trackFragments.value = {
            ...trackFragments.value,
            [trackId]: await api.getFragments(trackId),
        }
    } catch {
        trackFragments.value = {
            ...trackFragments.value,
            [trackId]: [],
        }
    }
}

const onWaveformTimeUpdate = (time: number) => {
    waveformLocalTime.value = time
}

const onWaveformPlayRequested = (track: Track) => {
    if (playerStore.currentTrack?.id === track.id) {
        playerStore.play()
        return
    }
    playerStore.playTrack(track, false)
}

const onWaveformPauseRequested = () => {
    playerStore.pause()
}

const onWaveformSeekRequested = (track: Track, time: number) => {
    if (playerStore.currentTrack?.id === track.id) {
        playerStore.seek(time)
        return
    }
    playerStore.playTrack(track, false)
    setTimeout(() => {
        if (playerStore.currentTrack?.id !== track.id) return
        playerStore.seek(time)
    }, 100)
}

const onWaveformFragmentCreated = async (track: Track, data: { start_time: number; end_time: number; is_liked: boolean }) => {
    try {
        await api.createFragment(track.id, data)
        await loadTrackFragments(track.id)
        toast.success(data.is_liked ? 'Saved positive fragment.' : 'Saved negative fragment.')
    } catch {
        toast.error('Failed to save fragment feedback.')
    }
}

const isTrackAnalyzing = (trackId: number): boolean => Boolean(analyzingTracks.value[trackId])

const setTrackAnalyzing = (trackId: number, value: boolean) => {
    if (value) {
        analyzingTracks.value = { ...analyzingTracks.value, [trackId]: true }
        return
    }

    const next = { ...analyzingTracks.value }
    delete next[trackId]
    analyzingTracks.value = next
}

const getAnalysisToastId = (trackId: number) => `segment-analysis-${trackId}`

const clearAnalysisPoll = (trackId: number) => {
    const poll = analysisPolls.get(trackId)
    if (poll !== undefined) {
        clearInterval(poll)
        analysisPolls.delete(trackId)
    }
}

const analyzeTrack = async (track: Track) => {
    if (isTrackAnalyzing(track.id)) {
        return
    }

    setTrackAnalyzing(track.id, true)
    clearAnalysisPoll(track.id)
    try {
        await api.analyzeTrackSegments(track.id)

        toast.info('Segment analysis started...', {
            toastId: getAnalysisToastId(track.id),
            autoClose: false,
            closeOnClick: false,
        })

        const poll = setInterval(async () => {
            const status = await api.getTrackSegmentAnalysisStatus(track.id)
            if (status.status === 'completed') {
                clearAnalysisPoll(track.id)
                setTrackAnalyzing(track.id, false)
                toast.remove(getAnalysisToastId(track.id))
                toast.success(`Created ${status.segments_created} segment(s).`, {
                    toastId: `${getAnalysisToastId(track.id)}-done`,
                })
                const data = await api.getTrackFeedbackSnapshot(track.id)
                trackSegments.value[track.id] = data.segments || []
                trackFragments.value = {
                    ...trackFragments.value,
                    [track.id]: data.fragments || [],
                }
            } else if (status.status === 'failed') {
                clearAnalysisPoll(track.id)
                setTrackAnalyzing(track.id, false)
                toast.remove(getAnalysisToastId(track.id))
                toast.error('Segment analysis failed.', {
                    toastId: `${getAnalysisToastId(track.id)}-error`,
                })
            }
        }, 1000)
        analysisPolls.set(track.id, poll)
    } catch {
        clearAnalysisPoll(track.id)
        setTrackAnalyzing(track.id, false)
        toast.remove(getAnalysisToastId(track.id))
        toast.error('Could not start segment analysis.', {
            toastId: `${getAnalysisToastId(track.id)}-start-error`,
        })
    }
}

const rateSegment = async (trackId: number, segment: Segment, rating: number) => {
    segment.rating = rating
    try {
        const payload = await api.rateSegment(segment.id, rating)
        const feedback = payload.feedback as TrackFeedbackSnapshot | null | undefined
        const effectiveTrackRating = typeof feedback?.track_rating === 'number'
            ? feedback.track_rating
            : (typeof payload.track_rating === 'number'
                ? payload.track_rating
                : (typeof payload.computed_track_rating === 'number'
                    ? Math.max(1, Math.min(5, Math.round(payload.computed_track_rating)))
                    : null))

        if (effectiveTrackRating !== null) {
            trackRatings.value[trackId] = effectiveTrackRating
        }
        playerStore.rememberTrackRating(trackId, effectiveTrackRating)
        void playerStore.enforceSkipDislikedTracks(trackId)

        if (Array.isArray(feedback?.segments)) {
            trackSegments.value[trackId] = feedback.segments
        }
        if (Array.isArray(feedback?.fragments)) {
            trackFragments.value = {
                ...trackFragments.value,
                [trackId]: feedback.fragments,
            }
        }

        if (!feedback && effectiveTrackRating !== null) {
            trackRatings.value[trackId] = effectiveTrackRating
        }

        window.dispatchEvent(new CustomEvent('track-feedback-updated', {
            detail: { trackId },
        }))
    } catch {
        console.error('Failed to rate segment')
    }
}

const playSegment = (track: Track, segment: Segment) => {
    playerStore.playTrack(track, false)
    setTimeout(() => {
        if (playerStore.currentTrack?.id !== track.id) return
        playerStore.seek(segment.start)
    }, 100)
}

const getTypeIcon = (type: string): string => {
    const icons: Record<string, string> = {
        drop_high: 'Drop',
        drop_bass: 'Bass',
        melodic: 'Lead',
        verse: 'Verse',
        buildup: 'Build',
        break: 'Break',
        ambient: 'Pad',
        intro: 'Intro',
        outro: 'Outro',
    }
    return icons[type] || 'Part'
}

const getStarClass = (trackId: number, seg: Segment, star: number): string => {
    const hoverKey = `${trackId}-${seg.id}`
    const hoverValue = hoverStars.value[hoverKey] || 0

    if (hoverValue > 0) {
        return star <= hoverValue ? 'text-yellow-400' : 'text-gray-300'
    }

    return seg.rating && seg.rating >= star ? 'text-yellow-400' : 'text-gray-300'
}

const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

const formatDuration = (seconds: number): string => formatTime(seconds)

const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
        not_trained: 'text-slate-500',
        training: 'text-blue-600',
        trained: 'text-emerald-600',
        error: 'text-rose-600',
    }
    return colors[status] || 'text-slate-500'
}

onMounted(() => {
    loadModelStatus()
    loadRecommendations()
})

watch(() => playerStore.currentTime, (time) => {
    if (!expandedTrackInGlobalPlayer.value || !waveformRef.value) return
    if (Math.abs(waveformLocalTime.value - time) <= 0.5) return
    waveformRef.value.syncSeekTo(time)
})

watch(() => playerStore.isPlaying, (playing) => {
    if (!expandedTrackInGlobalPlayer.value || !waveformRef.value) return
    if (playing) {
        waveformRef.value.syncPlay()
        return
    }
    waveformRef.value.syncPause()
})

watch(() => playerStore.currentTrack?.id, () => {
    if (!expandedTrackInGlobalPlayer.value || !waveformRef.value) return
    waveformRef.value.syncSeekTo(playerStore.currentTime)
    if (playerStore.isPlaying) {
        waveformRef.value.syncPlay()
        return
    }
    waveformRef.value.syncPause()
})

onBeforeUnmount(() => {
    for (const poll of analysisPolls.values()) {
        clearInterval(poll)
    }
    analysisPolls.clear()
})
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p class="section-kicker">Recommendations</p>
                    <h1 class="page-title">Personalized discovery</h1>
                    <p class="page-copy">
                        Train the current user model, filter the candidate space, and review recommendations with inline waveform and segment feedback.
                    </p>
                </div>
                <button type="button" class="action-primary" :disabled="training || !canTrain" @click="trainModel">
                    {{ training ? 'Training...' : 'Train model' }}
                </button>
            </div>
        </section>

        <section class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <article v-for="stat in recommendationStats" :key="stat.label" class="metric-card">
                <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{{ stat.label }}</p>
                <p class="mt-3 text-3xl font-semibold text-slate-950">{{ stat.value }}</p>
                <p class="mt-2 text-sm text-slate-500">{{ stat.detail }}</p>
            </article>
        </section>

        <section class="surface-card !rounded-[1.75rem] !p-5">
            <div class="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
                <div class="flex-1">
                    <h2 class="text-xl font-semibold text-slate-950">Model status</h2>
                    <p v-if="modelStatus" class="mt-2 text-sm text-slate-600">
                        Status:
                        <span :class="getStatusColor(modelStatus.status)">{{ modelStatus.status }}</span>
                        <span v-if="modelStatus.trained_at" class="ml-2 text-slate-500">
                            {{ new Date(modelStatus.trained_at).toLocaleDateString() }}
                        </span>
                    </p>
                    <div class="mt-4 flex flex-wrap gap-2">
                        <span v-if="stats.training_scope !== 'user_ratings_only'" class="badge-soft">Liked tracks {{ stats.liked_tracks }}</span>
                        <span class="badge-soft">High ratings {{ stats.high_rated_tracks }}</span>
                        <span v-if="stats.training_scope !== 'user_ratings_only'" class="badge-soft">Segment ratings {{ stats.high_rated_segments || 0 }}</span>
                        <span class="badge-soft !border-emerald-200 !bg-emerald-50 !text-emerald-700">Total positive {{ stats.total_positive }}</span>
                    </div>
                    <p v-if="stats.training_scope === 'user_ratings_only'" class="mt-3 text-xs leading-6 text-slate-500">
                        This model currently trains only from explicit track ratings.
                    </p>
                    <p v-else-if="stats.training_scope === 'user_scoped_feedback'" class="mt-3 text-xs leading-6 text-slate-500">
                        This model already uses user-scoped likes, ratings, segment feedback, and listening history.
                    </p>
                </div>

                <div class="grid gap-4 md:grid-cols-2 xl:w-[420px]">
                    <div>
                        <label class="field-label" for="recommend-genre">Genre</label>
                        <select id="recommend-genre" v-model="filters.genre" class="field-select" @change="loadRecommendations">
                            <option :value="undefined">All genres</option>
                            <option v-for="genre in genres" :key="genre" :value="genre">{{ genre }}</option>
                        </select>
                    </div>
                    <div>
                        <label class="field-label" for="recommend-mood">Mood</label>
                        <select id="recommend-mood" v-model="filters.mood" class="field-select" @change="loadRecommendations">
                            <option :value="undefined">All moods</option>
                            <option v-for="mood in moods" :key="mood" :value="mood">{{ mood }}</option>
                        </select>
                    </div>
                    <div>
                        <label class="field-label" for="recommend-bpm-min">Min BPM</label>
                        <input id="recommend-bpm-min" v-model.number="filters.bpm_min" type="number" placeholder="90" class="field-control" @change="loadRecommendations">
                    </div>
                    <div>
                        <label class="field-label" for="recommend-bpm-max">Max BPM</label>
                        <input id="recommend-bpm-max" v-model.number="filters.bpm_max" type="number" placeholder="170" class="field-control" @change="loadRecommendations">
                    </div>
                    <div class="md:col-span-2">
                        <button type="button" class="action-secondary w-full" @click="loadRecommendations">
                            Refresh recommendations
                        </button>
                    </div>
                </div>
            </div>
        </section>

        <section class="surface-card !rounded-[1.75rem] !p-5">
            <div v-if="loading" class="empty-state">
                <p class="text-lg font-semibold text-slate-900">Loading recommendations...</p>
            </div>

            <div v-else-if="modelStatus?.status !== 'trained'" class="empty-state">
                <p class="text-lg font-semibold text-slate-900">Train the model first</p>
                <p class="mt-2 text-sm text-slate-500">Recommendations appear once the current preference model finishes training.</p>
            </div>

            <div v-else-if="recommendations.length === 0" class="empty-state">
                <p class="text-lg font-semibold text-slate-900">No recommendations</p>
                <p class="mt-2 text-sm text-slate-500">Try broader filters or add more preference signals.</p>
            </div>

            <div v-else class="space-y-4">
                <article
                    v-for="rec in recommendations"
                    :key="rec.track?.id"
                    class="overflow-hidden rounded-[1.5rem] border border-slate-200 bg-slate-50/80"
                >
                    <div class="flex flex-col gap-4 p-5 lg:flex-row lg:items-center lg:justify-between">
                        <div class="flex items-start gap-4">
                            <div class="flex flex-col gap-2">
                                <button type="button" class="action-primary !rounded-xl !px-3 !py-2" @click.stop="playTrack(rec.track)">
                                    Play
                                </button>
                                <button type="button" class="action-secondary !rounded-xl !px-3 !py-2" @click.stop="toggleLike(rec)">
                                    {{ rec.track.is_liked ? 'Liked' : 'Like' }}
                                </button>
                            </div>
                            <div class="min-w-0">
                                <button type="button" class="text-left text-lg font-semibold text-slate-950 hover:text-blue-700" @click="openTrack(rec.track.id)">
                                    {{ rec.track.title }}
                                </button>
                                <p class="mt-1 text-sm text-slate-500">
                                    {{ rec.track.artist || 'Unknown artist' }} • {{ rec.track.genre || 'Unknown genre' }}
                                </p>
                                <div class="mt-3 flex flex-wrap gap-2">
                                    <span class="badge-soft">{{ formatDuration(rec.track.duration) }}</span>
                                    <span class="badge-soft !border-blue-200 !bg-blue-50 !text-blue-700">{{ (rec.score * 100).toFixed(1) }}% score</span>
                                </div>
                            </div>
                        </div>

                        <div class="flex flex-wrap items-center gap-3">
                            <RatingWidget :model-value="trackRatings[rec.track.id]" size="sm" @rate="(rating) => rateTrack(rec.track.id, rating)" />
                            <button type="button" class="action-secondary !rounded-xl !px-3 !py-2 !text-xs" @click.stop="toggleWaveform(rec.track.id)">
                                {{ expandedWaveformTrackId === rec.track.id ? 'Hide waveform' : 'Waveform' }}
                            </button>
                        </div>
                    </div>

                    <div v-if="expandedWaveformTrackId === rec.track.id" class="min-w-0 overflow-hidden border-t border-slate-200 bg-slate-950 px-4 py-4">
                        <WaveformWithFragments
                            ref="waveformRef"
                            :audio-url="getTrackAudioUrl(rec.track.id)"
                            :fragments="getWaveformFragments(rec.track.id)"
                            :duration="rec.track.duration"
                            :track-id="rec.track.id"
                            :external-audio="true"
                            @fragment-created="(data) => onWaveformFragmentCreated(rec.track, data)"
                            @time-update="onWaveformTimeUpdate"
                            @play-requested="onWaveformPlayRequested(rec.track)"
                            @pause-requested="onWaveformPauseRequested"
                            @seek="(time) => onWaveformSeekRequested(rec.track, time)"
                        />
                    </div>

                    <div class="border-t border-slate-200 bg-white/70 px-4 py-4">
                        <div v-if="!trackSegments[rec.track.id]?.length" class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <span class="text-sm text-slate-500">No analyzed segments yet.</span>
                            <button type="button" class="action-secondary !rounded-xl !px-3 !py-2 !text-xs" :disabled="isTrackAnalyzing(rec.track.id)" @click="analyzeTrack(rec.track)">
                                {{ isTrackAnalyzing(rec.track.id) ? 'Analyzing...' : 'Analyze segments' }}
                            </button>
                        </div>

                        <div v-else class="space-y-2">
                            <div class="max-h-56 space-y-2 overflow-y-auto">
                                <div
                                    v-for="seg in trackSegments[rec.track.id]"
                                    :key="seg.id"
                                    class="flex flex-col gap-3 rounded-[1.1rem] border border-slate-200 px-3 py-3 transition sm:flex-row sm:items-center sm:justify-between"
                                    :class="currentSegmentByTrack[rec.track.id]?.id === seg.id ? 'bg-blue-50' : 'bg-slate-50/80'"
                                    @click="playSegment(rec.track, seg)"
                                >
                                    <div class="flex items-center gap-3">
                                        <span class="badge-soft !normal-case !tracking-normal">{{ getTypeIcon(seg.type) }}</span>
                                        <div>
                                            <div class="text-sm font-semibold text-slate-900">{{ formatTime(seg.start) }} - {{ formatTime(seg.end) }}</div>
                                            <div class="text-xs uppercase tracking-[0.16em] text-slate-500">{{ getSegmentDisplayLabel(seg) }}</div>
                                        </div>
                                    </div>
                                    <div class="flex items-center gap-1" @click.stop @mouseleave="hoverStars[`${rec.track.id}-${seg.id}`] = 0">
                                        <button
                                            v-for="star in [1, 2, 3, 4, 5]"
                                            :key="star"
                                            type="button"
                                            class="text-lg transition-transform hover:scale-125"
                                            :class="getStarClass(rec.track.id, seg, star)"
                                            @click="rateSegment(rec.track.id, seg, star)"
                                            @mouseenter="hoverStars[`${rec.track.id}-${seg.id}`] = star"
                                        >
                                            ★
                                        </button>
                                    </div>
                                </div>
                            </div>
                            <div class="flex justify-end">
                                <button type="button" class="action-secondary !rounded-xl !px-3 !py-2 !text-xs" :disabled="isTrackAnalyzing(rec.track.id)" @click="analyzeTrack(rec.track)">
                                    Reanalyze
                                </button>
                            </div>
                        </div>
                    </div>
                </article>
            </div>
        </section>
    </div>
</template>
