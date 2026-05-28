<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { AUTH_TOKEN_STORAGE_KEY } from '@/constants/auth'

interface TrackStat {
    id: number
    title: string
    artist: string
    genre: string
    bpm: number
    key: string
    mood: string
    signals: {
        is_liked: boolean
        rating: number | null
        play_count: number
        completed_plays: number
        last_played: string | null
    }
    preference_score: number
    is_positive: boolean
}

interface SystemStats {
    preference_stats: {
        liked_tracks: number
        liked_fragments: number
        disliked_fragments: number
        fragments_with_comments: number
        high_rated_tracks: number
        frequently_played: number
        total_plays: number
        total_positive: number
    }
    model_status: {
        status: string
        trained_at: string | null
        likes_count: number
    }
    queue_stats: {
        pending_tasks: number
        processing_tasks: number
        completed_tasks: number
        failed_tasks: number
        training_tasks: number
    }
    redis_queue_length: number
    analysis_stats: {
        total_tracks: number
        analyzed: number
        pending: number
        processing: number
        failed: number
    }
    recent_plays: Array<{
        track: string
        artist: string
        started_at: string
        completed: boolean
        duration_played: number
    }>
}

const trackStats = ref<TrackStat[]>([])
const systemStats = ref<SystemStats | null>(null)
const isLoading = ref(false)
const error = ref<string | null>(null)
const sortBy = ref<'score' | 'plays' | 'id'>('score')
const filterPositive = ref(false)
const autoRefresh = ref(false)
let refreshInterval: number | null = null

function getDebugHeaders(): HeadersInit {
    const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
    return token
        ? {
            Accept: 'application/json',
            Authorization: `Bearer ${token}`,
        }
        : {
            Accept: 'application/json',
        }
}

function isTrackStatsPayload(value: unknown): value is { tracks: TrackStat[] } {
    return Boolean(
        value &&
        typeof value === 'object' &&
        Array.isArray((value as { tracks?: unknown }).tracks),
    )
}

function isSystemStatsPayload(value: unknown): value is SystemStats {
    if (!value || typeof value !== 'object') return false

    const payload = value as Partial<SystemStats>

    return Boolean(
        payload.preference_stats &&
        payload.model_status &&
        typeof payload.model_status.status === 'string' &&
        payload.queue_stats &&
        payload.analysis_stats &&
        Array.isArray(payload.recent_plays),
    )
}

const sortedTracks = computed(() => {
    let tracks = [...trackStats.value]

    if (filterPositive.value) {
        tracks = tracks.filter((track) => track.is_positive)
    }

    switch (sortBy.value) {
        case 'score':
            return tracks.sort((a, b) => b.preference_score - a.preference_score)
        case 'plays':
            return tracks.sort((a, b) => b.signals.play_count - a.signals.play_count)
        case 'id':
            return tracks.sort((a, b) => b.id - a.id)
        default:
            return tracks
    }
})

const summaryCards = computed(() => {
    if (!systemStats.value) return []

    return [
        {
            label: 'Model status',
            value: systemStats.value.model_status.status,
            detail: `Trained: ${formatDate(systemStats.value.model_status.trained_at)}`,
        },
        {
            label: 'Positive signals',
            value: systemStats.value.preference_stats.total_positive.toString(),
            detail: `${systemStats.value.preference_stats.liked_tracks} liked • ${systemStats.value.preference_stats.high_rated_tracks} highly rated`,
        },
        {
            label: 'ML queue',
            value: systemStats.value.redis_queue_length.toString(),
            detail: `${systemStats.value.queue_stats.pending_tasks} pending • ${systemStats.value.queue_stats.completed_tasks} completed`,
        },
        {
            label: 'Analyzed tracks',
            value: `${systemStats.value.analysis_stats.analyzed}/${systemStats.value.analysis_stats.total_tracks}`,
            detail: `${systemStats.value.analysis_stats.pending} pending • ${systemStats.value.analysis_stats.failed} failed`,
        },
    ]
})

async function loadData() {
    isLoading.value = true
    error.value = null

    try {
        const [tracksResponse, systemResponse] = await Promise.all([
            fetch('/api/debug/tracks', { headers: getDebugHeaders() }),
            fetch('/api/debug/system', { headers: getDebugHeaders() }),
        ])

        if (!tracksResponse.ok) {
            throw new Error(`Debug tracks request failed (${tracksResponse.status})`)
        }

        if (!systemResponse.ok) {
            throw new Error(`Debug system request failed (${systemResponse.status})`)
        }

        const tracksPayload: unknown = await tracksResponse.json()
        const systemPayload: unknown = await systemResponse.json()

        if (!isTrackStatsPayload(tracksPayload)) {
            throw new Error('Debug tracks payload is invalid')
        }

        if (!isSystemStatsPayload(systemPayload)) {
            throw new Error('Debug system payload is invalid')
        }

        trackStats.value = tracksPayload.tracks
        systemStats.value = systemPayload
    } catch (errorValue: any) {
        error.value = errorValue.message || 'Failed to load data'
    } finally {
        isLoading.value = false
    }
}

function toggleAutoRefresh() {
    autoRefresh.value = !autoRefresh.value

    if (autoRefresh.value) {
        refreshInterval = window.setInterval(loadData, 5000)
    } else if (refreshInterval) {
        clearInterval(refreshInterval)
        refreshInterval = null
    }
}

function getScoreClass(score: number): string {
    if (score >= 0.7) return 'text-emerald-600'
    if (score >= 0.5) return 'text-amber-600'
    if (score >= 0.3) return 'text-orange-600'
    return 'text-slate-500'
}

function formatDate(date: string | null): string {
    if (!date) return '—'
    return new Date(date).toLocaleString()
}

onMounted(() => {
    loadData()
})

onBeforeUnmount(() => {
    if (refreshInterval) {
        clearInterval(refreshInterval)
        refreshInterval = null
    }
})
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p class="section-kicker">Debug</p>
                    <h1 class="page-title">Training debug console</h1>
                    <p class="page-copy">Inspect preference signals, queue state, and per-track training features without leaving the frontend workspace.</p>
                </div>
                <div class="flex flex-wrap gap-3">
                    <button type="button" :class="autoRefresh ? 'action-primary' : 'action-secondary'" @click="toggleAutoRefresh">
                        {{ autoRefresh ? 'Stop auto refresh' : 'Auto refresh' }}
                    </button>
                    <button type="button" class="action-secondary" :disabled="isLoading" @click="loadData">
                        {{ isLoading ? 'Refreshing...' : 'Refresh now' }}
                    </button>
                </div>
            </div>
        </section>

        <div v-if="error" class="rounded-[1.4rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-700">
            {{ error }}
        </div>

        <section v-if="systemStats" class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <article v-for="card in summaryCards" :key="card.label" class="metric-card">
                <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{{ card.label }}</p>
                <p class="mt-3 text-3xl font-semibold text-slate-950">{{ card.value }}</p>
                <p class="mt-2 text-sm text-slate-500">{{ card.detail }}</p>
            </article>
        </section>

        <section v-if="systemStats?.recent_plays?.length" class="surface-card !rounded-[1.75rem] !p-5">
            <h2 class="text-xl font-semibold text-slate-950">Recent plays</h2>
            <div class="mt-4 flex flex-wrap gap-2">
                <span
                    v-for="(play, index) in systemStats.recent_plays.slice(0, 8)"
                    :key="index"
                    class="badge-soft !normal-case !tracking-normal"
                    :class="play.completed ? '!border-emerald-200 !bg-emerald-50 !text-emerald-700' : ''"
                >
                    {{ play.track }}{{ play.completed ? ' completed' : '' }}
                </span>
            </div>
        </section>

        <section class="surface-card !rounded-[1.75rem] !p-5">
            <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                    <h2 class="text-xl font-semibold text-slate-950">Track training data</h2>
                    <p class="mt-1 text-sm text-slate-500">{{ sortedTracks.length }} track rows in the current debug view.</p>
                </div>
                <div class="flex flex-wrap items-center gap-3">
                    <label class="flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-medium text-slate-700">
                        <input v-model="filterPositive" type="checkbox" class="field-checkbox">
                        Positive only
                    </label>
                    <select v-model="sortBy" class="field-select !w-auto !min-w-[180px]">
                        <option value="score">Sort by score</option>
                        <option value="plays">Sort by plays</option>
                        <option value="id">Sort by id</option>
                    </select>
                </div>
            </div>

            <div v-if="isLoading" class="empty-state mt-5">
                <p class="text-lg font-semibold text-slate-900">Refreshing debug data...</p>
            </div>

            <div v-else class="table-shell mt-5 overflow-x-auto">
                <table class="min-w-[1080px] w-full text-sm">
                    <thead class="table-header">
                        <tr>
                            <th class="px-4 py-4 text-left">ID</th>
                            <th class="px-4 py-4 text-left">Track</th>
                            <th class="px-4 py-4 text-left">Genre</th>
                            <th class="px-4 py-4 text-center">BPM</th>
                            <th class="px-4 py-4 text-center">Liked</th>
                            <th class="px-4 py-4 text-center">Rating</th>
                            <th class="px-4 py-4 text-center">Plays</th>
                            <th class="px-4 py-4 text-center">Completed</th>
                            <th class="px-4 py-4 text-center">Score</th>
                            <th class="px-4 py-4 text-center">Positive</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100">
                        <tr v-for="track in sortedTracks" :key="track.id" class="table-row">
                            <td class="px-4 py-4 table-meta">{{ track.id }}</td>
                            <td class="px-4 py-4">
                                <div class="font-semibold text-slate-950">{{ track.title }}</div>
                                <div class="mt-1 text-xs text-slate-500">{{ track.artist }}</div>
                            </td>
                            <td class="px-4 py-4 table-meta">{{ track.genre || '—' }}</td>
                            <td class="px-4 py-4 text-center table-meta">{{ track.bpm ? Math.round(track.bpm) : '—' }}</td>
                            <td class="px-4 py-4 text-center">{{ track.signals.is_liked ? 'Yes' : 'No' }}</td>
                            <td class="px-4 py-4 text-center">{{ track.signals.rating || '—' }}</td>
                            <td class="px-4 py-4 text-center">{{ track.signals.play_count }}</td>
                            <td class="px-4 py-4 text-center">{{ track.signals.completed_plays }}</td>
                            <td class="px-4 py-4 text-center font-mono font-semibold" :class="getScoreClass(track.preference_score)">
                                {{ track.preference_score.toFixed(2) }}
                            </td>
                            <td class="px-4 py-4 text-center">{{ track.is_positive ? 'Yes' : 'No' }}</td>
                        </tr>
                    </tbody>
                </table>
            </div>

            <div class="mt-5 rounded-[1.3rem] border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-600">
                Legend: liked, rating, play count, completed plays, and positive-signal classification are the main features visible here. This page is intentionally diagnostic, so the emphasis is on density and traceability rather than playback controls.
            </div>
        </section>
    </div>
</template>
