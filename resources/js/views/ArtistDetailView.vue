<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { toast } from 'vue3-toastify'
import api from '@/api/client'
import ArtistCreditsInline from '@/components/ArtistCreditsInline.vue'
import RatingWidget from '@/components/RatingWidget.vue'
import { useAuthStore } from '@/stores/authStore'
import { usePlayerStore } from '@/stores/playerStore'
import type { ArtistProfile, Track } from '@/types'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()
const playerStore = usePlayerStore()

const loading = ref(true)
const profile = ref<ArtistProfile | null>(null)
const trackRatings = ref<Record<number, number | null>>({})
const ratingStates = ref<Record<number, 'idle' | 'loading' | 'saving'>>({})
const isEditingArtistName = ref(false)
const isSavingArtistName = ref(false)
const artistNameDraft = ref('')

const artistSlug = computed(() => String(route.params.slug || ''))
const canManageArtist = computed(() => authStore.isAdmin)
const allTracks = computed(() => {
    if (!profile.value) {
        return []
    }

    const seen = new Set<number>()

    return [
        ...profile.value.primary_tracks,
        ...profile.value.featured_tracks,
        ...profile.value.remix_tracks,
    ].filter((track) => {
        if (seen.has(track.id)) {
            return false
        }

        seen.add(track.id)
        return true
    })
})

const sections = computed(() => {
    if (!profile.value) {
        return []
    }

    return [
        { key: 'primary', title: 'Tracks', tracks: profile.value.primary_tracks },
        { key: 'featured', title: 'Appears On', tracks: profile.value.featured_tracks },
        { key: 'remix', title: 'Remixes', tracks: profile.value.remix_tracks },
    ].filter((section) => section.tracks.length > 0)
})

const totalTracks = computed(() => (
    (profile.value?.primary_tracks.length || 0)
    + (profile.value?.featured_tracks.length || 0)
    + (profile.value?.remix_tracks.length || 0)
))

const setTrackRatingState = (trackId: number, state: 'idle' | 'loading' | 'saving') => {
    ratingStates.value = {
        ...ratingStates.value,
        [trackId]: state,
    }
}

const updateTrackLikeProjection = (trackId: number, explicitRating: number | null) => {
    if (!profile.value) {
        return
    }

    for (const section of [profile.value.primary_tracks, profile.value.featured_tracks, profile.value.remix_tracks]) {
        const match = section.find((track) => track.id === trackId)
        if (match) {
            match.is_liked = explicitRating === 5
        }
    }
}

const loadTrackRatings = async () => {
    const tracks = allTracks.value

    if (tracks.length === 0) {
        trackRatings.value = {}
        return
    }

    await Promise.allSettled(tracks.map(async (track) => {
        setTrackRatingState(track.id, 'loading')

        try {
            const rating = await playerStore.fetchTrackEffectiveRating(track.id)
            trackRatings.value = {
                ...trackRatings.value,
                [track.id]: rating,
            }
        } finally {
            setTrackRatingState(track.id, 'idle')
        }
    }))
}

const loadArtist = async () => {
    loading.value = true
    profile.value = null
    trackRatings.value = {}
    ratingStates.value = {}
    isEditingArtistName.value = false

    try {
        profile.value = await api.getArtist(artistSlug.value)
        artistNameDraft.value = profile.value.artist.name
        await loadTrackRatings()
    } catch (error) {
        console.error('Failed to load artist:', error)
        router.push({ name: 'library' })
    } finally {
        loading.value = false
    }
}

const playTrack = async (track: Track) => {
    await playerStore.playTrack(track)
}

const startArtistNameEdit = () => {
    if (!profile.value || !canManageArtist.value) {
        return
    }

    artistNameDraft.value = profile.value.artist.name
    isEditingArtistName.value = true
}

const cancelArtistNameEdit = () => {
    artistNameDraft.value = profile.value?.artist.name || ''
    isEditingArtistName.value = false
}

const saveArtistName = async () => {
    if (!profile.value || !canManageArtist.value || isSavingArtistName.value) {
        return
    }

    const nextName = artistNameDraft.value.trim()
    if (nextName === '') {
        toast.error('Artist name is required.')
        return
    }

    if (nextName === profile.value.artist.name) {
        isEditingArtistName.value = false
        return
    }

    isSavingArtistName.value = true

    try {
        const response = await api.updateArtist(profile.value.artist.slug, nextName)
        toast.success(`Updated artist on ${response.updated_tracks} track(s).`)
        isEditingArtistName.value = false

        if (response.artist.slug !== artistSlug.value) {
            await router.replace({ name: 'artist', params: { slug: response.artist.slug } })
            return
        }

        await loadArtist()
    } catch (error) {
        console.error('Failed to rename artist:', error)
        toast.error('Failed to update artist name.')
    } finally {
        isSavingArtistName.value = false
    }
}

const rateTrack = async (track: Track, rating: number) => {
    setTrackRatingState(track.id, 'saving')

    try {
        const response = await api.setTrackRating(track.id, rating)
        const feedback = response.feedback
        const effectiveRating = typeof feedback?.track_rating === 'number'
            ? feedback.track_rating
            : (typeof response.effective_rating === 'number' ? response.effective_rating : rating)
        const explicitRating = typeof feedback?.explicit_track_rating === 'number'
            ? feedback.explicit_track_rating
            : (typeof response.explicit_track_rating === 'number' ? response.explicit_track_rating : null)

        trackRatings.value = {
            ...trackRatings.value,
            [track.id]: effectiveRating,
        }
        playerStore.rememberTrackRating(track.id, effectiveRating)
        void playerStore.enforceSkipDislikedTracks(track.id)
        updateTrackLikeProjection(track.id, explicitRating)
        window.dispatchEvent(new CustomEvent('track-feedback-updated', {
            detail: { trackId: track.id },
        }))
        toast.success(`Track rated ${rating} star(s).`)
    } catch (error) {
        console.error('Failed to rate track:', error)
        toast.error('Failed to save track rating.')
        trackRatings.value = {
            ...trackRatings.value,
            [track.id]: await playerStore.fetchTrackEffectiveRating(track.id, { force: true }),
        }
    } finally {
        setTrackRatingState(track.id, 'idle')
    }
}

const clearTrackRating = async (track: Track) => {
    setTrackRatingState(track.id, 'saving')

    try {
        const response = await api.deleteTrackRating(track.id)
        const feedback = response.feedback
        const effectiveRating = typeof feedback?.track_rating === 'number'
            ? feedback.track_rating
            : (typeof response.effective_rating === 'number' ? response.effective_rating : null)
        const explicitRating = typeof feedback?.explicit_track_rating === 'number'
            ? feedback.explicit_track_rating
            : (typeof response.explicit_track_rating === 'number' ? response.explicit_track_rating : null)

        trackRatings.value = {
            ...trackRatings.value,
            [track.id]: effectiveRating,
        }
        playerStore.rememberTrackRating(track.id, effectiveRating)
        void playerStore.enforceSkipDislikedTracks(track.id)
        updateTrackLikeProjection(track.id, explicitRating)
        window.dispatchEvent(new CustomEvent('track-feedback-updated', {
            detail: { trackId: track.id },
        }))
        toast.info('Track rating removed.')
    } catch (error) {
        console.error('Failed to clear track rating:', error)
        toast.error('Failed to remove track rating.')
        trackRatings.value = {
            ...trackRatings.value,
            [track.id]: await playerStore.fetchTrackEffectiveRating(track.id, { force: true }),
        }
    } finally {
        setTrackRatingState(track.id, 'idle')
    }
}

const openTrack = (track: Track) => {
    router.push({ name: 'track', params: { id: track.id } })
}

const formatTime = (seconds: number): string => {
    const totalSeconds = Math.max(0, Math.round(seconds))
    const minutes = Math.floor(totalSeconds / 60)
    const remainingSeconds = totalSeconds % 60

    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
}

watch(() => route.params.slug, () => {
    void loadArtist()
})

onMounted(() => {
    void loadArtist()
})
</script>

<template>
    <div v-if="loading" class="surface-card text-center text-sm text-slate-500">
        Loading artist...
    </div>

    <div v-else-if="profile" class="space-y-6 pb-10">
        <section class="surface-card !p-6">
            <p class="section-kicker">Artist</p>
            <div class="mt-3 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div>
                    <h1 class="text-3xl font-semibold text-slate-950">{{ profile.artist.name }}</h1>
                </div>
                <div v-if="canManageArtist" class="flex shrink-0 flex-wrap items-center gap-2">
                    <button
                        v-if="!isEditingArtistName"
                        type="button"
                        class="action-secondary !rounded-xl !px-3 !py-2 !text-xs"
                        @click="startArtistNameEdit"
                    >
                        Rename artist
                    </button>
                </div>
            </div>
            <div class="mt-4 flex flex-wrap gap-2">
                <span class="badge-soft">{{ totalTracks }} linked tracks</span>
                <span v-if="profile.primary_tracks.length > 0" class="badge-soft">{{ profile.primary_tracks.length }} primary</span>
                <span v-if="profile.featured_tracks.length > 0" class="badge-soft !border-emerald-200 !bg-emerald-50 !text-emerald-700">
                    {{ profile.featured_tracks.length }} features
                </span>
                <span v-if="profile.remix_tracks.length > 0" class="badge-soft !border-amber-200 !bg-amber-50 !text-amber-700">
                    {{ profile.remix_tracks.length }} remixes
                </span>
            </div>
            <div v-if="canManageArtist && isEditingArtistName" class="mt-5 flex flex-col gap-3 rounded-[1.15rem] border border-slate-200 bg-slate-50/80 p-4">
                <label class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500" for="artist-name-input">
                    Artist name
                </label>
                <div class="flex flex-col gap-3 lg:flex-row lg:items-center">
                    <input
                        id="artist-name-input"
                        v-model="artistNameDraft"
                        type="text"
                        class="field-control"
                        :disabled="isSavingArtistName"
                    >
                    <div class="flex flex-wrap gap-2">
                        <button
                            type="button"
                            class="action-primary"
                            :disabled="isSavingArtistName"
                            @click="saveArtistName"
                        >
                            {{ isSavingArtistName ? 'Saving...' : 'Save name' }}
                        </button>
                        <button
                            type="button"
                            class="action-secondary"
                            :disabled="isSavingArtistName"
                            @click="cancelArtistNameEdit"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </section>

        <section
            v-for="section in sections"
            :key="section.key"
            class="surface-card !p-6"
        >
            <div class="flex items-center justify-between gap-4">
                <div>
                    <p class="section-kicker">Artist Library</p>
                    <h2 class="mt-2 text-2xl font-semibold text-slate-950">{{ section.title }}</h2>
                </div>
                <span class="badge-soft">{{ section.tracks.length }}</span>
            </div>

            <div class="mt-5 space-y-3">
                <article
                    v-for="track in section.tracks"
                    :key="track.id"
                    class="rounded-[1.35rem] border border-slate-200 bg-white/80 px-4 py-4"
                >
                    <div class="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                        <div class="min-w-0">
                            <button
                                type="button"
                                class="text-left text-lg font-semibold text-slate-950 transition hover:text-sky-700"
                                @click="openTrack(track)"
                            >
                                {{ track.title }}
                            </button>
                            <div class="mt-1 text-sm text-slate-500">
                                <ArtistCreditsInline
                                    :credits="track.artist_credits"
                                    :fallback="track.artist"
                                    link-class="transition hover:text-sky-700"
                                />
                            </div>
                            <div class="mt-3 flex flex-wrap gap-2">
                                <span class="badge-soft">Duration: {{ formatTime(track.duration) }}</span>
                                <span class="badge-soft">Source: {{ track.source }}</span>
                                <span :class="['status-pill', track.analysis_status === 'completed' ? 'status-pill--ready' : 'status-pill--working']">
                                    {{ track.analysis_status }}
                                </span>
                            </div>
                        </div>

                        <button
                            type="button"
                            class="action-secondary !rounded-xl !px-3 !py-2"
                            @click="playTrack(track)"
                        >
                            Play
                        </button>
                    </div>

                    <div class="mt-4 flex flex-wrap items-center gap-3 border-t border-slate-200 pt-4">
                        <RatingWidget
                            v-model="trackRatings[track.id]"
                            size="sm"
                            :readonly="ratingStates[track.id] === 'loading' || ratingStates[track.id] === 'saving'"
                            @rate="(rating) => rateTrack(track, rating)"
                            @clear="clearTrackRating(track)"
                        />
                        <span v-if="ratingStates[track.id] === 'loading'" class="text-xs font-medium uppercase tracking-[0.18em] text-slate-400">
                            Loading rating
                        </span>
                        <span v-else-if="ratingStates[track.id] === 'saving'" class="text-xs font-medium uppercase tracking-[0.18em] text-slate-400">
                            Saving rating
                        </span>
                    </div>
                </article>
            </div>
        </section>
    </div>
</template>
