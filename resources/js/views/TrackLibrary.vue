<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/api/client'
import ArtistCreditsInline from '@/components/ArtistCreditsInline.vue'
import type { Track, TrackFilters } from '@/types'
import { usePlayerStore } from '@/stores/playerStore'
import { useAuthStore } from '@/stores/authStore'

const router = useRouter()
const playerStore = usePlayerStore()
const authStore = useAuthStore()

const createDefaultFilters = (): TrackFilters => ({
    genre: undefined,
    mood: undefined,
    bpm_min: undefined,
    bpm_max: undefined,
    liked: undefined,
    status: undefined,
    search: '',
    sort_by: 'created_at',
    sort_order: 'desc',
    per_page: 20,
})

const tracks = ref<Track[]>([])
const loading = ref(false)
const deletingSelected = ref(false)
const retryingTrackIds = ref<number[]>([])
const currentPage = ref(1)
const totalPages = ref(1)
const total = ref(0)
const selectedTrackIds = ref<number[]>([])
const filters = ref<TrackFilters>(createDefaultFilters())

const genres = [
    'Rock', 'Pop', 'Jazz', 'Classical', 'Hip-Hop', 'Metal', 'Folk',
    'R&B', 'Country', 'Blues', 'Reggae', 'Punk', 'Soul', 'Indie',
    'House', 'Techno', 'Trance', 'Dubstep', 'Drum & Bass',
    'Ambient', 'EDM', 'Electro', 'Trap', 'Future Bass', 'Synthwave',
    'Lo-Fi', 'Deep House', 'Tech House', 'Hardstyle', 'Downtempo',
]

const moods = ['Happy', 'Sad', 'Energetic', 'Calm', 'Aggressive', 'Romantic']

const statuses = [
    { value: 'completed', label: 'Ready' },
    { value: 'pending', label: 'Queued' },
    { value: 'processing', label: 'Processing' },
    { value: 'failed', label: 'Failed' },
]

const statusLabels: Record<string, string> = {
    pending: 'Queued',
    processing: 'Processing',
    analyzing: 'Analyzing',
    completed: 'Ready',
    failed: 'Failed',
}

const loadTracks = async () => {
    loading.value = true
    try {
        const response = await api.getTracks({
            ...filters.value,
            page: currentPage.value,
        })
        tracks.value = response.data
        totalPages.value = response.last_page
        total.value = response.total
        selectedTrackIds.value = []
    } catch (error) {
        console.error('Failed to load tracks:', error)
    } finally {
        loading.value = false
    }
}

const selectedTracksCount = computed(() => selectedTrackIds.value.length)
const canDeleteTracks = computed(() => authStore.isAdmin)
const canRetryTracks = computed(() => authStore.isAdmin)

const allVisibleSelected = computed(() => (
    tracks.value.length > 0 && tracks.value.every((track) => selectedTrackIds.value.includes(track.id))
))

const hasTracks = computed(() => tracks.value.length > 0)

const hasActiveFilters = computed(() => Boolean(
    filters.value.search
    || filters.value.genre
    || filters.value.mood
    || filters.value.bpm_min
    || filters.value.bpm_max
    || filters.value.liked
    || filters.value.status,
))

const libraryStats = computed(() => ([
    {
        label: 'Visible tracks',
        value: hasTracks.value ? tracks.value.length.toString() : '0',
        detail: `${total.value} total in current query`,
    },
    {
        label: 'Selected',
        value: selectedTracksCount.value.toString(),
        detail: selectedTracksCount.value > 0 ? 'ready for batch actions' : 'nothing selected',
    },
    {
        label: 'Ready now',
        value: tracks.value.filter((track) => track.analysis_status === 'completed').length.toString(),
        detail: 'playable and analyzed',
    },
    {
        label: 'Liked',
        value: tracks.value.filter((track) => track.is_liked).length.toString(),
        detail: 'in this page view',
    },
]))

const toggleTrackSelection = (trackId: number, checked: boolean) => {
    if (checked) {
        if (!selectedTrackIds.value.includes(trackId)) {
            selectedTrackIds.value = [...selectedTrackIds.value, trackId]
        }
        return
    }

    selectedTrackIds.value = selectedTrackIds.value.filter((id) => id !== trackId)
}

const toggleAllVisibleTracks = (checked: boolean) => {
    selectedTrackIds.value = checked ? tracks.value.map((track) => track.id) : []
}

const getCheckboxState = (event: Event): boolean => Boolean((event.target as HTMLInputElement | null)?.checked)

const deleteTracksByIds = async (ids: number[]) => {
    if (!canDeleteTracks.value) {
        alert('Only admins can delete tracks.')
        return {
            deletedCount: 0,
            failedCount: ids.length,
        }
    }

    try {
        const result = await api.deleteTracks(ids)
        const deletedIds = result.deleted_ids

        if (deletedIds.length > 0) {
            tracks.value = tracks.value.filter((track) => !deletedIds.includes(track.id))
            selectedTrackIds.value = selectedTrackIds.value.filter((id) => !deletedIds.includes(id))
            total.value = Math.max(0, total.value - deletedIds.length)
        }

        if (tracks.value.length === 0 && currentPage.value > 1) {
            currentPage.value--
        }

        await loadTracks()

        return {
            deletedCount: deletedIds.length,
            failedCount: ids.length - deletedIds.length,
        }
    } catch (error) {
        console.error('Failed to delete selected tracks:', error)
        return {
            deletedCount: 0,
            failedCount: ids.length,
        }
    }
}

const toggleLike = async (track: Track) => {
    try {
        if (track.is_liked) {
            const response = await api.unlikeTrack(track.id)
            track.is_liked = false
            const effectiveRating = typeof response.feedback?.track_rating === 'number'
                ? response.feedback.track_rating
                : (typeof response.effective_rating === 'number' ? response.effective_rating : null)
            playerStore.rememberTrackRating(track.id, effectiveRating)
            void playerStore.enforceSkipDislikedTracks(track.id)
        } else {
            const response = await api.likeTrack(track.id)
            track.is_liked = true
            const effectiveRating = typeof response.feedback?.track_rating === 'number'
                ? response.feedback.track_rating
                : (typeof response.effective_rating === 'number' ? response.effective_rating : null)
            playerStore.rememberTrackRating(track.id, effectiveRating)
            void playerStore.enforceSkipDislikedTracks(track.id)
        }
    } catch (error) {
        console.error('Failed to toggle like:', error)
    }
}

const deleteTrack = async (track: Track) => {
    if (!confirm(`Delete "${track.title}"?`)) return
    const result = await deleteTracksByIds([track.id])
    if (result.failedCount > 0) {
        console.error('Failed to delete track:', track.id)
    }
}

const deleteSelectedTracks = async () => {
    if (selectedTrackIds.value.length === 0 || deletingSelected.value) return

    const count = selectedTrackIds.value.length
    if (!confirm(`Delete ${count} selected track${count === 1 ? '' : 's'}?`)) return

    deletingSelected.value = true
    try {
        const result = await deleteTracksByIds([...selectedTrackIds.value])
        if (result.failedCount > 0) {
            alert(`Deleted ${result.deletedCount} track(s). Failed to delete ${result.failedCount}.`)
        }
    } finally {
        deletingSelected.value = false
    }
}

const retryTrackAnalysis = async (track: Track, event: Event) => {
    event.stopPropagation()

    if (!canRetryTracks.value) {
        alert('Only admins can retry failed analysis.')
        return
    }

    if (track.analysis_status !== 'failed') {
        return
    }

    if (retryingTrackIds.value.includes(track.id)) {
        return
    }

    retryingTrackIds.value = [...retryingTrackIds.value, track.id]

    try {
        const response = await api.retryTrackAnalysis(track.id)
        track.analysis_status = response.track.analysis_status
        await loadTracks()
    } catch (error) {
        console.error('Failed to retry track analysis:', error)
        alert('Failed to queue track for re-analysis.')
    } finally {
        retryingTrackIds.value = retryingTrackIds.value.filter((id) => id !== track.id)
    }
}

const openTrack = (track: Track) => {
    router.push({ name: 'track', params: { id: track.id } })
}

const exportCsv = async () => {
    try {
        const blob = await api.exportCsv(filters.value)
        const url = URL.createObjectURL(blob)
        const anchor = document.createElement('a')
        anchor.href = url
        anchor.download = `music_library_${new Date().toISOString().split('T')[0]}.csv`
        anchor.click()
        URL.revokeObjectURL(url)
    } catch (error) {
        console.error('Failed to export CSV:', error)
    }
}

const playTrack = (track: Track, event: Event) => {
    event.stopPropagation()
    playerStore.playTrack(track, false)
}

const playAllTracks = () => {
    if (tracks.value.length > 0) {
        playerStore.playPlaylist(tracks.value, 0)
    }
}

const addToQueue = (track: Track, event: Event) => {
    event.stopPropagation()
    playerStore.addToQueue(track, 'end', 'manual')
}

const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

const formatStatusLabel = (status: string) => statusLabels[status] || status

const getStatusBadge = (status: string) => {
    const badges: Record<string, string> = {
        pending: 'status-pill status-pill--queued',
        processing: 'status-pill status-pill--working',
        analyzing: 'status-pill status-pill--working',
        completed: 'status-pill status-pill--ready',
        failed: 'status-pill status-pill--failed',
    }

    return badges[status] || 'status-pill'
}

const clearFilters = () => {
    filters.value = createDefaultFilters()
    currentPage.value = 1
}

watch(filters, () => {
    currentPage.value = 1
    loadTracks()
}, { deep: true })

onMounted(() => {
    loadTracks()
})
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card overflow-hidden">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p class="section-kicker">Library</p>
                    <h1 class="page-title">Music library</h1>
                    <p class="page-copy">
                        Search the catalog, batch-manage tracks, and move directly into playback, queueing, export, or detailed analysis.
                    </p>
                </div>

                <div class="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
                    <button
                        v-if="canDeleteTracks"
                        type="button"
                        class="action-danger w-full sm:w-auto"
                        :disabled="selectedTracksCount === 0 || deletingSelected"
                        @click="deleteSelectedTracks"
                    >
                        {{ deletingSelected ? 'Deleting...' : `Delete selected (${selectedTracksCount})` }}
                    </button>
                    <button
                        type="button"
                        class="action-secondary w-full sm:w-auto"
                        :disabled="!hasTracks"
                        @click="playAllTracks"
                    >
                        Play visible
                    </button>
                    <button
                        type="button"
                        class="action-secondary w-full sm:w-auto"
                        @click="exportCsv"
                    >
                        Export CSV
                    </button>
                    <router-link to="/upload" class="action-primary w-full sm:w-auto">
                        Add track
                    </router-link>
                </div>
            </div>
        </section>

        <section class="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <article
                v-for="stat in libraryStats"
                :key="stat.label"
                class="metric-card"
            >
                <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{{ stat.label }}</p>
                <p class="mt-3 text-3xl font-semibold text-slate-950">{{ stat.value }}</p>
                <p class="mt-2 text-sm text-slate-500">{{ stat.detail }}</p>
            </article>
        </section>

        <section class="surface-card space-y-6">
            <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                    <p class="section-kicker">Filters</p>
                    <h2 class="mt-2 text-2xl font-semibold text-slate-950">Refine the current catalog view</h2>
                </div>

                <button
                    v-if="hasActiveFilters"
                    type="button"
                    class="action-secondary w-full sm:w-auto"
                    @click="clearFilters"
                >
                    Clear filters
                </button>
            </div>

            <div class="grid gap-4 sm:grid-cols-2 xl:grid-cols-6">
                <div class="xl:col-span-2">
                    <label class="field-label" for="library-search">Search</label>
                    <input
                        id="library-search"
                        v-model="filters.search"
                        type="text"
                        placeholder="Title, artist, or album"
                        class="field-control"
                    >
                </div>

                <div>
                    <label class="field-label" for="library-genre">Genre</label>
                    <select id="library-genre" v-model="filters.genre" class="field-select">
                        <option :value="undefined">All genres</option>
                        <option v-for="genre in genres" :key="genre" :value="genre">
                            {{ genre }}
                        </option>
                    </select>
                </div>

                <div>
                    <label class="field-label" for="library-mood">Mood</label>
                    <select id="library-mood" v-model="filters.mood" class="field-select">
                        <option :value="undefined">All moods</option>
                        <option v-for="mood in moods" :key="mood" :value="mood">
                            {{ mood }}
                        </option>
                    </select>
                </div>

                <div>
                    <label class="field-label" for="library-bpm-min">Min BPM</label>
                    <input id="library-bpm-min" v-model.number="filters.bpm_min" type="number" placeholder="80" class="field-control">
                </div>

                <div>
                    <label class="field-label" for="library-bpm-max">Max BPM</label>
                    <input id="library-bpm-max" v-model.number="filters.bpm_max" type="number" placeholder="160" class="field-control">
                </div>

                <div class="xl:col-span-2">
                    <label class="field-label" for="library-status">Analysis status</label>
                    <select id="library-status" v-model="filters.status" class="field-select">
                        <option :value="undefined">All statuses</option>
                        <option v-for="status in statuses" :key="status.value" :value="status.value">
                            {{ status.label }}
                        </option>
                    </select>
                </div>

                <div class="flex items-end sm:col-span-2 xl:col-span-1">
                    <label class="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white/70 px-4 py-3 text-sm font-medium text-slate-700">
                        <input v-model="filters.liked" type="checkbox" class="field-checkbox">
                        Liked only
                    </label>
                </div>
            </div>
        </section>

        <section class="surface-card space-y-5">
            <div class="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                <div>
                    <p class="section-kicker">Results</p>
                    <h2 class="mt-2 text-2xl font-semibold text-slate-950">Track catalog</h2>
                </div>
                <div class="text-sm text-slate-500">
                    Page {{ currentPage }} of {{ totalPages }} • {{ total }} total tracks
                </div>
            </div>

            <div v-if="loading" class="empty-state">
                <p class="text-lg font-semibold text-slate-800">Loading library...</p>
                <p class="mt-2 text-sm text-slate-500">Fetching the current catalog and filter state.</p>
            </div>

            <div v-else-if="!hasTracks" class="empty-state">
                <p class="text-lg font-semibold text-slate-800">No tracks found</p>
                <p class="mt-2 text-sm text-slate-500">
                    {{ hasActiveFilters ? 'This filter combination returned no tracks.' : 'Import your first tracks to populate the library.' }}
                </p>
                <div class="mt-5 flex flex-wrap justify-center gap-3">
                    <button
                        v-if="hasActiveFilters"
                        type="button"
                        class="action-secondary w-full sm:w-auto"
                        @click="clearFilters"
                    >
                        Reset filters
                    </button>
                    <router-link to="/upload" class="action-primary w-full sm:w-auto">
                        Open import tools
                    </router-link>
                </div>
            </div>

            <div v-else class="space-y-4">
                <div class="space-y-3 md:hidden">
                    <div
                        v-if="canDeleteTracks"
                        class="flex items-center justify-between rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-3"
                    >
                        <div>
                            <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Selection</div>
                            <div class="mt-1 text-sm font-medium text-slate-700">{{ selectedTracksCount }} selected</div>
                        </div>
                        <label class="flex items-center gap-3 text-sm font-medium text-slate-700">
                            <input
                                type="checkbox"
                                :checked="allVisibleSelected"
                                :disabled="!hasTracks"
                                class="field-checkbox"
                                title="Select all visible tracks"
                                @change="toggleAllVisibleTracks(getCheckboxState($event))"
                            >
                            All visible
                        </label>
                    </div>

                    <article
                        v-for="track in tracks"
                        :key="`mobile-${track.id}`"
                        class="rounded-[1.3rem] border border-slate-200 bg-white/90 p-4 shadow-[0_16px_40px_rgba(15,23,42,0.06)]"
                        @click="openTrack(track)"
                    >
                        <div class="flex items-start gap-3">
                            <div v-if="canDeleteTracks" class="pt-1" @click.stop>
                                <input
                                    type="checkbox"
                                    :checked="selectedTrackIds.includes(track.id)"
                                    class="field-checkbox"
                                    :title="`Select ${track.title}`"
                                    @change="toggleTrackSelection(track.id, getCheckboxState($event))"
                                >
                            </div>

                            <div class="min-w-0 flex-1">
                                <div class="flex items-start justify-between gap-3">
                                    <div class="min-w-0">
                                        <div class="flex flex-wrap items-center gap-2">
                                            <p class="truncate text-base font-semibold text-slate-950">{{ track.title }}</p>
                                            <span v-if="track.is_liked" class="badge-soft !border-rose-200 !bg-rose-50 !text-rose-600">Liked</span>
                                        </div>
                                        <p class="mt-1 text-sm text-slate-500">
                                            <ArtistCreditsInline
                                                :credits="track.artist_credits"
                                                :fallback="track.artist"
                                                link-class="transition hover:text-sky-700"
                                            />
                                        </p>
                                        <p v-if="track.album" class="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">
                                            {{ track.album }}
                                        </p>
                                    </div>

                                    <button
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        title="Play track"
                                        @click="playTrack(track, $event)"
                                    >
                                        Play
                                    </button>
                                </div>

                                <div class="mt-3 flex flex-wrap gap-2">
                                    <span v-if="track.genre" class="badge-soft">
                                        {{ track.genre }}
                                    </span>
                                    <span v-if="track.mood" class="badge-soft !border-emerald-200 !bg-emerald-50 !text-emerald-700">
                                        {{ track.mood }}
                                    </span>
                                    <span :class="getStatusBadge(track.analysis_status)">
                                        {{ formatStatusLabel(track.analysis_status) }}
                                    </span>
                                </div>

                                <div class="mt-3 grid grid-cols-3 gap-2 rounded-[1rem] border border-slate-200 bg-slate-50/80 p-3 text-sm">
                                    <div>
                                        <div class="text-[0.65rem] font-semibold uppercase tracking-[0.16em] text-slate-500">BPM</div>
                                        <div class="mt-1 font-semibold text-slate-800">{{ track.bpm ? Math.round(track.bpm) : '-' }}</div>
                                    </div>
                                    <div>
                                        <div class="text-[0.65rem] font-semibold uppercase tracking-[0.16em] text-slate-500">Length</div>
                                        <div class="mt-1 font-semibold text-slate-800">{{ formatDuration(track.duration) }}</div>
                                    </div>
                                    <div>
                                        <div class="text-[0.65rem] font-semibold uppercase tracking-[0.16em] text-slate-500">Source</div>
                                        <div class="mt-1 font-semibold text-slate-800">{{ track.source === 'youtube' ? 'YouTube' : 'Upload' }}</div>
                                    </div>
                                </div>

                                <div class="mt-3 grid grid-cols-2 gap-2" @click.stop>
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        @click="toggleLike(track)"
                                    >
                                        {{ track.is_liked ? 'Unlike' : 'Like' }}
                                    </button>
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        @click="addToQueue(track, $event)"
                                    >
                                        Queue
                                    </button>
                                    <button
                                        v-if="track.analysis_status === 'failed' && canRetryTracks"
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        :disabled="retryingTrackIds.includes(track.id)"
                                        @click="retryTrackAnalysis(track, $event)"
                                    >
                                        {{ retryingTrackIds.includes(track.id) ? 'Retrying...' : 'Retry' }}
                                    </button>
                                    <button
                                        v-if="canDeleteTracks"
                                        type="button"
                                        class="action-danger !rounded-xl !px-3 !py-2"
                                        @click="deleteTrack(track)"
                                    >
                                        Delete
                                    </button>
                                </div>
                            </div>
                        </div>
                    </article>
                </div>

                <div class="hidden md:block table-shell overflow-x-auto">
                <table class="min-w-[980px] w-full">
                    <thead class="table-header">
                        <tr>
                            <th v-if="canDeleteTracks" class="px-4 py-4 text-center">
                                <input
                                    type="checkbox"
                                    :checked="allVisibleSelected"
                                    :disabled="!hasTracks"
                                    class="field-checkbox"
                                    title="Select all visible tracks"
                                    @change="toggleAllVisibleTracks(getCheckboxState($event))"
                                >
                            </th>
                            <th class="px-4 py-4 text-left">Track</th>
                            <th class="px-4 py-4 text-left">Profile</th>
                            <th class="px-4 py-4 text-left">BPM</th>
                            <th class="px-4 py-4 text-left">Length</th>
                            <th class="px-4 py-4 text-left">Status</th>
                            <th class="px-4 py-4 text-right">Actions</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100">
                        <tr
                            v-for="track in tracks"
                            :key="track.id"
                            class="table-row cursor-pointer"
                            @click="openTrack(track)"
                        >
                            <td v-if="canDeleteTracks" class="px-4 py-4 text-center" @click.stop>
                                <input
                                    type="checkbox"
                                    :checked="selectedTrackIds.includes(track.id)"
                                    class="field-checkbox"
                                    :title="`Select ${track.title}`"
                                    @change="toggleTrackSelection(track.id, getCheckboxState($event))"
                                >
                            </td>
                            <td class="px-4 py-4">
                                <div class="flex items-start gap-3">
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        title="Play track"
                                        @click="playTrack(track, $event)"
                                    >
                                        Play
                                    </button>
                                    <div class="min-w-0">
                                        <div class="flex flex-wrap items-center gap-2">
                                            <p class="text-base font-semibold text-slate-950">{{ track.title }}</p>
                                            <span v-if="track.is_liked" class="badge-soft !border-rose-200 !bg-rose-50 !text-rose-600">Liked</span>
                                        </div>
                                        <p class="mt-1 text-sm text-slate-500">
                                            <ArtistCreditsInline
                                                :credits="track.artist_credits"
                                                :fallback="track.artist"
                                                link-class="transition hover:text-sky-700"
                                            />
                                        </p>
                                        <p v-if="track.album" class="mt-1 text-xs uppercase tracking-[0.18em] text-slate-400">
                                            {{ track.album }}
                                        </p>
                                    </div>
                                </div>
                            </td>
                            <td class="px-4 py-4">
                                <div class="flex flex-wrap gap-2">
                                    <span v-if="track.genre" class="badge-soft">
                                        {{ track.genre }}
                                    </span>
                                    <span v-if="track.mood" class="badge-soft !border-emerald-200 !bg-emerald-50 !text-emerald-700">
                                        {{ track.mood }}
                                    </span>
                                    <span v-if="!track.genre && !track.mood" class="table-meta">No tags yet</span>
                                </div>
                            </td>
                            <td class="px-4 py-4 text-sm font-semibold text-slate-700">
                                {{ track.bpm ? Math.round(track.bpm) : '—' }}
                            </td>
                            <td class="px-4 py-4 table-meta">
                                {{ formatDuration(track.duration) }}
                            </td>
                            <td class="px-4 py-4">
                                <span :class="getStatusBadge(track.analysis_status)">
                                    {{ formatStatusLabel(track.analysis_status) }}
                                </span>
                            </td>
                            <td class="px-4 py-4">
                                <div class="flex justify-end gap-2" @click.stop>
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        @click="toggleLike(track)"
                                    >
                                        {{ track.is_liked ? 'Unlike' : 'Like' }}
                                    </button>
                                    <button
                                        v-if="track.analysis_status === 'failed' && canRetryTracks"
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        :disabled="retryingTrackIds.includes(track.id)"
                                        @click="retryTrackAnalysis(track, $event)"
                                    >
                                        {{ retryingTrackIds.includes(track.id) ? 'Retrying...' : 'Retry' }}
                                    </button>
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-xl !px-3 !py-2"
                                        @click="addToQueue(track, $event)"
                                    >
                                        Queue
                                    </button>
                                    <button
                                        v-if="canDeleteTracks"
                                        type="button"
                                        class="action-danger !rounded-xl !px-3 !py-2"
                                        @click="deleteTrack(track)"
                                    >
                                        Delete
                                    </button>
                                </div>
                            </td>
                        </tr>
                    </tbody>
                </table>
                </div>
            </div>

            <div
                v-if="!loading && hasTracks && totalPages > 1"
                class="flex flex-col gap-3 border-t border-slate-200 pt-5 md:flex-row md:items-center md:justify-between"
            >
                <div class="text-sm text-slate-500">
                    Showing {{ tracks.length }} visible tracks from {{ total }} total
                </div>
                <div class="flex flex-wrap items-center gap-2">
                    <button
                        type="button"
                        class="action-secondary !rounded-xl !px-4 !py-2"
                        :disabled="currentPage === 1"
                        @click="currentPage--; loadTracks()"
                    >
                        Previous
                    </button>
                    <span class="badge-soft !normal-case !tracking-normal">
                        {{ currentPage }} / {{ totalPages }}
                    </span>
                    <button
                        type="button"
                        class="action-secondary !rounded-xl !px-4 !py-2"
                        :disabled="currentPage === totalPages"
                        @click="currentPage++; loadTracks()"
                    >
                        Next
                    </button>
                </div>
            </div>
        </section>
    </div>
</template>
