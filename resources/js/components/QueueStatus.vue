<script setup lang="ts">
/**
 * QueueStatus - Real-time queue status using Server-Sent Events.
 *
 * Uses SSE instead of polling for efficient real-time updates.
 * Includes failed tasks management.
 */
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { AUTH_TOKEN_STORAGE_KEY } from '@/constants/auth'

interface QueueData {
    queue: { youtube: number; analysis: number; segments: number; other: number; default: number; total: number }
    tracks: { pending: number; processing: number; completed: number; failed: number }
    recent: Array<{
        id: number
        title: string
        artist: string | null
        analysis_status: string
        created_at: string
        source: string
    }>
    timestamp?: string
}

interface FailedTask {
    id: number
    task_type: string
    track_id: number | null
    track_title: string | null
    track_artist: string | null
    error_message: string | null
    created_at: string
    completed_at: string | null
}

const data = ref<QueueData | null>(null)
const loading = ref(true)
const expanded = ref(false)
const connected = ref(false)
const error = ref<string | null>(null)

const showFailedPanel = ref(false)
const failedTasks = ref<FailedTask[]>([])
const failedLoading = ref(false)
const retryingAll = ref(false)
const retryingTask = ref<number | null>(null)

let eventSource: EventSource | null = null
let reconnectTimeout: number | null = null
let snapshotInterval: number | null = null

const getAuthToken = (): string | null => localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)

const buildApiUrl = (path: string): string => {
    const token = getAuthToken()
    if (!token) {
        return `/api/${path}`
    }

    const separator = path.includes('?') ? '&' : '?'
    return `/api/${path}${separator}token=${encodeURIComponent(token)}`
}

const authHeaders = (): HeadersInit => {
    const token = getAuthToken()
    return token ? { Authorization: `Bearer ${token}` } : {}
}

const normalizeQueueData = (payload: Partial<QueueData> & Record<string, any>): QueueData => ({
    queue: {
        youtube: Number(payload.queue?.youtube ?? 0),
        analysis: Number(payload.queue?.analysis ?? 0),
        segments: Number(payload.queue?.segments ?? 0),
        other: Number(payload.queue?.other ?? 0),
        default: Number(payload.queue?.default ?? 0),
        total: Number(payload.queue?.total ?? 0),
    },
    tracks: {
        pending: Number(payload.tracks?.pending ?? 0),
        processing: Number(payload.tracks?.processing ?? 0),
        completed: Number(payload.tracks?.completed ?? 0),
        failed: Number(payload.tracks?.failed ?? 0),
    },
    recent: Array.isArray(payload.recent) ? payload.recent : [],
    timestamp: typeof payload.timestamp === 'string' ? payload.timestamp : undefined,
})

const loadSnapshot = async () => {
    try {
        const response = await fetch('/api/tracks/queue-status', { headers: authHeaders() })
        if (!response.ok) {
            throw new Error(`Queue snapshot request failed with status ${response.status}`)
        }

        data.value = normalizeQueueData(await response.json())
        loading.value = false
        error.value = null
    } catch (e) {
        console.error('Failed to load queue snapshot:', e)
        if (!data.value) {
            error.value = 'Failed to load queue status'
            loading.value = false
        }
    }
}

const connectSSE = () => {
    if (eventSource) {
        eventSource.close()
    }

    error.value = null

    try {
        const token = getAuthToken()
        if (!token) {
            error.value = 'Authentication token is required'
            loading.value = false
            return
        }

        eventSource = new EventSource(buildApiUrl('queue-status/stream'))

        eventSource.onopen = () => {
            connected.value = true
            loading.value = false
            error.value = null
            console.log('Queue status SSE connected')
        }

        eventSource.onmessage = (event) => {
            try {
                const newData = normalizeQueueData(JSON.parse(event.data))
                data.value = newData
                loading.value = false
            } catch (e) {
                console.error('Failed to parse SSE data:', e)
            }
        }

        eventSource.onerror = () => {
            connected.value = false
            console.warn('Queue status SSE error, reconnecting...')

            if (reconnectTimeout) clearTimeout(reconnectTimeout)
            reconnectTimeout = window.setTimeout(() => {
                connectSSE()
            }, 5000)
        }
    } catch (e) {
        console.error('Failed to create EventSource:', e)
        error.value = 'Failed to connect'
        loading.value = false
    }
}

const disconnect = () => {
    if (eventSource) {
        eventSource.close()
        eventSource = null
    }
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
        reconnectTimeout = null
    }
    connected.value = false
}

const startSnapshotPolling = () => {
    if (snapshotInterval) {
        clearInterval(snapshotInterval)
    }

    snapshotInterval = window.setInterval(() => {
        if (!expanded.value) {
            void loadSnapshot()
        }
    }, 30000)
}

const stopSnapshotPolling = () => {
    if (snapshotInterval) {
        clearInterval(snapshotInterval)
        snapshotInterval = null
    }
}

const statusIcon = (status: string) => {
    switch (status) {
        case 'completed': return 'OK'
        case 'processing': return '...'
        case 'pending': return 'Q'
        case 'failed': return 'X'
        default: return '?'
    }
}

const statusPillClass = (status: string) => {
    switch (status) {
        case 'completed': return 'status-pill--ready'
        case 'processing': return 'status-pill--working'
        case 'failed': return 'status-pill--failed'
        default: return 'status-pill--queued'
    }
}

const formatTime = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' })
}

const loadFailedTasks = async () => {
    failedLoading.value = true
    try {
        const response = await fetch('/api/queue-status/failed', { headers: authHeaders() })
        const result = await response.json()
        failedTasks.value = result.tasks
    } catch (e) {
        console.error('Failed to load failed tasks:', e)
    } finally {
        failedLoading.value = false
    }
}

const openFailedPanel = async () => {
    showFailedPanel.value = true
    await loadFailedTasks()
}

const retryTask = async (taskId: number) => {
    retryingTask.value = taskId
    try {
        const response = await fetch(`/api/queue-status/retry/${taskId}`, { method: 'POST', headers: authHeaders() })
        const result = await response.json()
        if (result.success) {
            failedTasks.value = failedTasks.value.filter(t => t.id !== taskId)
        }
    } catch (e) {
        console.error('Failed to retry task:', e)
    } finally {
        retryingTask.value = null
    }
}

const retryAllFailed = async () => {
    retryingAll.value = true
    try {
        const response = await fetch('/api/queue-status/retry-all', { method: 'POST', headers: authHeaders() })
        const result = await response.json()
        if (result.success) {
            await loadFailedTasks()
        }
    } catch (e) {
        console.error('Failed to retry all:', e)
    } finally {
        retryingAll.value = false
    }
}

const deleteTask = async (taskId: number) => {
    try {
        const response = await fetch(`/api/queue-status/task/${taskId}`, { method: 'DELETE', headers: authHeaders() })
        const result = await response.json()
        if (result.success) {
            failedTasks.value = failedTasks.value.filter(t => t.id !== taskId)
        }
    } catch (e) {
        console.error('Failed to delete task:', e)
    }
}

const clearAllFailed = async () => {
    if (!confirm('Delete all failed tasks? This action cannot be undone.')) return

    try {
        const response = await fetch('/api/queue-status/clear-failed', { method: 'DELETE', headers: authHeaders() })
        const result = await response.json()
        if (result.success) {
            failedTasks.value = []
        }
    } catch (e) {
        console.error('Failed to clear all:', e)
    }
}

const truncateError = (msg: string | null, maxLen: number = 100) => {
    if (!msg) return 'No error details'
    return msg.length > maxLen ? msg.substring(0, maxLen) + '...' : msg
}

onMounted(() => {
    void loadSnapshot()
    startSnapshotPolling()
})

onUnmounted(() => {
    disconnect()
    stopSnapshotPolling()
})

watch(expanded, (isExpanded) => {
    if (isExpanded) {
        stopSnapshotPolling()
        connectSSE()
        return
    }

    disconnect()
    void loadSnapshot()
    startSnapshotPolling()
})
</script>

<template>
    <div class="fixed bottom-[10.5rem] left-2 right-2 z-[60] sm:bottom-24 sm:left-auto sm:right-4">
        <button
            v-if="!expanded"
            @click="expanded = true"
            class="relative z-[60] w-full rounded-[1.4rem] border border-white/80 bg-white/90 px-4 py-3 shadow-[0_20px_50px_rgba(15,23,42,0.14)] backdrop-blur-xl transition hover:-translate-y-[1px] hover:shadow-[0_24px_60px_rgba(15,23,42,0.18)] sm:w-auto"
        >
            <span v-if="loading" class="text-sm font-medium text-slate-500">Loading queue...</span>
            <template v-else-if="data">
                <div class="flex items-center gap-3">
                    <span class="flex h-10 w-10 items-center justify-center rounded-[1rem] bg-sky-100 text-xs font-semibold tracking-[0.14em] text-sky-900">
                        {{ data.tracks.pending > 0 || data.tracks.processing > 0 ? 'RUN' : 'Q' }}
                    </span>
                    <div class="text-left">
                        <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Queue</div>
                        <div class="mt-1 text-sm font-semibold text-slate-950">
                            {{ data.tracks.pending + data.tracks.processing }} active
                        </div>
                    </div>
                    <span
                        class="h-2.5 w-2.5 rounded-full"
                        :class="connected ? 'bg-emerald-500' : 'bg-rose-500'"
                        :title="connected ? 'Connected' : 'Disconnected'"
                    ></span>
                </div>
            </template>
        </button>

        <div
            v-if="expanded"
            class="surface-card !w-full !max-h-[min(60vh,36rem)] !overflow-hidden !rounded-[1.8rem] !p-0 sm:!w-[22rem] sm:!max-h-[36rem]"
        >
            <div class="border-b border-slate-200/80 bg-slate-50/85 px-5 py-4">
                <div class="flex items-center justify-between gap-4">
                    <div class="flex items-center gap-3">
                        <div class="flex h-11 w-11 items-center justify-center rounded-[1.1rem] bg-sky-100 text-sm font-semibold tracking-[0.14em] text-sky-900">QS</div>
                        <div>
                            <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Live Status</div>
                            <div class="mt-1 text-base font-semibold text-slate-950">Queue Status</div>
                        </div>
                    </div>
                    <div class="flex items-center gap-3">
                        <span
                            class="h-2.5 w-2.5 rounded-full"
                            :class="connected ? 'bg-emerald-500' : 'animate-pulse bg-rose-500'"
                            :title="connected ? 'Live' : 'Reconnecting...'"
                        ></span>
                        <button @click="expanded = false" class="text-sm font-semibold text-slate-500 transition hover:text-slate-800">
                            Close
                        </button>
                    </div>
                </div>
            </div>

            <div v-if="loading" class="px-5 py-10 text-center text-sm text-slate-500">
                Connecting...
            </div>

            <div v-else-if="error" class="px-5 py-4">
                <div class="rounded-[1.2rem] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                    {{ error }}
                </div>
            </div>

            <div v-else-if="data" class="max-h-[32rem] overflow-y-auto">
                <div class="grid grid-cols-2 gap-3 px-5 py-5">
                    <div class="rounded-[1.15rem] border border-slate-200 bg-slate-50/80 p-3">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Downloading</div>
                        <div class="mt-2 text-xl font-semibold text-slate-950">{{ data.queue.youtube }}</div>
                    </div>
                    <div class="rounded-[1.15rem] border border-slate-200 bg-slate-50/80 p-3">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Track Queue</div>
                        <div class="mt-2 text-xl font-semibold text-slate-950">{{ data.queue.analysis }}</div>
                    </div>
                    <div class="rounded-[1.15rem] border border-slate-200 bg-slate-50/80 p-3">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Segments</div>
                        <div class="mt-2 text-xl font-semibold text-slate-950">{{ data.queue.segments }}</div>
                    </div>
                    <div class="rounded-[1.15rem] border border-slate-200 bg-slate-50/80 p-3">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Other</div>
                        <div class="mt-2 text-xl font-semibold text-slate-950">{{ data.queue.other }}</div>
                    </div>
                    <div class="rounded-[1.15rem] border border-slate-200 bg-white/85 p-3">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Pending</div>
                        <div class="mt-2 text-xl font-semibold text-blue-700">{{ data.tracks.pending }}</div>
                    </div>
                    <div class="rounded-[1.15rem] border border-slate-200 bg-white/85 p-3">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Processing</div>
                        <div class="mt-2 text-xl font-semibold text-amber-600">{{ data.tracks.processing }}</div>
                    </div>
                    <div class="rounded-[1.15rem] border border-slate-200 bg-white/85 p-3">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Completed</div>
                        <div class="mt-2 text-xl font-semibold text-emerald-700">{{ data.tracks.completed }}</div>
                    </div>
                    <button
                        type="button"
                        class="rounded-[1.15rem] border border-rose-200 bg-rose-50/80 p-3 text-left transition hover:bg-rose-100/80"
                        :disabled="data.tracks.failed === 0"
                        @click="data.tracks.failed > 0 && openFailedPanel()"
                    >
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-rose-500">Failed</div>
                        <div class="mt-2 text-xl font-semibold text-rose-700">{{ data.tracks.failed }}</div>
                    </button>
                </div>

                <div
                    v-if="data.queue.youtube > 0 || data.tracks.pending > 0 || data.tracks.processing > 0 || data.queue.segments > 0"
                    class="mx-5 rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-600"
                >
                    <div class="flex items-start gap-2">
                        <span class="mt-0.5 text-slate-500">&gt;&gt;</span>
                        <span>
                            {{ data.queue.youtube > 0 ? `Downloading: ${data.queue.youtube}` : '' }}
                            {{ data.tracks.pending > 0 ? `Pending analysis: ${data.tracks.pending}` : '' }}
                            {{ data.tracks.processing > 0 ? `Processing: ${data.tracks.processing}` : '' }}
                            {{ data.queue.segments > 0 ? `Segments in queue: ${data.queue.segments}` : '' }}
                        </span>
                    </div>
                </div>

                <div class="mt-5 border-t border-slate-200/80">
                    <div class="sticky top-0 border-b border-slate-200/70 bg-white/90 px-5 py-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-500 backdrop-blur-sm">
                        Recent Imports
                    </div>
                    <div v-if="data.recent.length === 0" class="px-5 py-8 text-center text-sm text-slate-400">
                        No recent imports
                    </div>
                    <div
                        v-for="track in data.recent"
                        :key="track.id"
                        class="flex items-center gap-3 px-5 py-3 transition hover:bg-sky-50/60"
                    >
                        <span class="flex h-9 w-9 items-center justify-center rounded-[0.9rem] bg-slate-100 text-sm font-semibold text-slate-700">
                            {{ statusIcon(track.analysis_status) }}
                        </span>
                        <div class="min-w-0 flex-1">
                            <div class="truncate font-medium text-slate-950">{{ track.title }}</div>
                            <div class="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                                <span :class="['status-pill', statusPillClass(track.analysis_status)]">{{ track.analysis_status }}</span>
                                <span>{{ formatTime(track.created_at) }}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

            <div
                v-if="showFailedPanel"
                class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-4 backdrop-blur-sm"
                @click.self="showFailedPanel = false"
        >
            <div class="surface-card !w-full !max-h-[80vh] !overflow-hidden !rounded-[1.9rem] !p-0 sm:!w-[38rem]">
                <div class="border-b border-rose-200 bg-rose-50 px-5 py-4">
                    <div class="flex items-center justify-between gap-4">
                        <div class="flex items-center gap-3">
                            <div class="flex h-11 w-11 items-center justify-center rounded-[1rem] bg-rose-100 text-sm font-semibold text-rose-700">X</div>
                            <div>
                                <div class="text-xs font-semibold uppercase tracking-[0.2em] text-rose-500">Attention</div>
                                <div class="mt-1 text-lg font-semibold text-rose-900">Failed Tasks ({{ failedTasks.length }})</div>
                            </div>
                        </div>
                        <button @click="showFailedPanel = false" class="text-sm font-semibold text-slate-500 transition hover:text-slate-800">
                            Close
                        </button>
                    </div>
                </div>

                <div class="border-b border-slate-200/80 bg-slate-50/80 px-5 py-3">
                    <div class="flex flex-wrap gap-2">
                        <button
                            @click="retryAllFailed"
                            :disabled="retryingAll || failedTasks.length === 0"
                            class="action-primary !rounded-xl !px-3 !py-2 !text-xs disabled:opacity-50 disabled:shadow-none"
                        >
                            {{ retryingAll ? 'Retrying...' : 'Retry All' }}
                        </button>
                        <button
                            @click="loadFailedTasks"
                            :disabled="failedLoading"
                            class="action-secondary !rounded-xl !px-3 !py-2 !text-xs"
                        >
                            Refresh
                        </button>
                        <button
                            @click="clearAllFailed"
                            :disabled="failedTasks.length === 0"
                            class="action-danger ml-auto !rounded-xl !px-3 !py-2 !text-xs"
                        >
                            Clear All
                        </button>
                    </div>
                </div>

                <div v-if="failedLoading" class="px-5 py-10 text-center text-sm text-slate-500">
                    Loading failed tasks...
                </div>

                <div v-else-if="failedTasks.length === 0" class="px-5 py-10 text-center text-sm text-slate-500">
                    No failed tasks!
                </div>

                <div v-else class="max-h-[60vh] overflow-y-auto">
                    <div
                        v-for="task in failedTasks"
                        :key="task.id"
                        class="border-b border-slate-200/70 px-5 py-4 transition hover:bg-slate-50/80"
                    >
                        <div class="flex items-start gap-4">
                            <div class="min-w-0 flex-1">
                                <div class="font-medium text-slate-950">
                                    {{ task.track_title || `Track #${task.track_id}` }}
                                </div>
                                <div class="mt-1 text-sm text-slate-500">
                                    {{ task.track_artist || 'Unknown artist' }}
                                </div>
                                <div class="mt-3 rounded-[1rem] border border-rose-200 bg-rose-50 px-3 py-2 font-mono text-xs text-rose-700 break-all">
                                    {{ truncateError(task.error_message) }}
                                </div>
                                <div class="mt-2 text-xs text-slate-400">
                                    Task #{{ task.id }} | {{ task.task_type }} | {{ formatTime(task.created_at) }}
                                </div>
                            </div>
                            <div class="flex flex-col gap-2">
                                <button
                                    @click="retryTask(task.id)"
                                    :disabled="retryingTask === task.id"
                                    class="action-primary !rounded-xl !px-3 !py-2 !text-xs disabled:opacity-50 disabled:shadow-none"
                                >
                                    {{ retryingTask === task.id ? 'Retrying...' : 'Retry' }}
                                </button>
                                <button
                                    @click="deleteTask(task.id)"
                                    class="action-secondary !rounded-xl !px-3 !py-2 !text-xs"
                                >
                                    Delete
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>
