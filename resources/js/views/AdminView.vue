<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import api from '@/api/client'
import type { AdminOverview } from '@/types'

type TrainingScope = 'user_scoped' | 'global_legacy'

const trainingScopes: Array<{ key: TrainingScope; title: string }> = [
    { key: 'user_scoped', title: 'User-scoped' },
    { key: 'global_legacy', title: 'Global/legacy' },
]

const overview = ref<AdminOverview | null>(null)
const loading = ref(true)
const error = ref('')
const actionMessage = ref('')
const actionError = ref('')
const activeActionKey = ref('')
const lastLoadedAt = ref<string | null>(null)

const radioDiagnostics = computed(() => overview.value?.radio_diagnostics ?? null)
const recentRadioEvents = computed(() => radioDiagnostics.value?.recent_events ?? [])
const topRadioEvents = computed(() => radioDiagnostics.value?.top_events ?? [])

async function loadOverview(): Promise<void> {
    loading.value = true
    error.value = ''

    try {
        overview.value = await api.getAdminOverview()
        lastLoadedAt.value = new Date().toISOString()
    } catch (err: any) {
        error.value = err?.response?.data?.message ?? 'Failed to load admin overview'
    } finally {
        loading.value = false
    }
}

onMounted(loadOverview)

async function runAdminAction(actionKey: string, runner: () => Promise<{ message?: string }>): Promise<void> {
    activeActionKey.value = actionKey
    actionMessage.value = ''
    actionError.value = ''

    try {
        const result = await runner()
        actionMessage.value = result.message ?? 'Admin action completed'
        await loadOverview()
    } catch (err: any) {
        actionError.value = err?.response?.data?.message ?? 'Admin action failed'
    } finally {
        activeActionKey.value = ''
    }
}

const repairTrainingQueue = () => runAdminAction('repair-training-queue', () => api.repairTrainingQueue())
const retrainUser = (userId: number) => runAdminAction(`retrain-${userId}`, () => api.retrainUserModel(userId))
const retryTrainingTask = (taskId: number) => runAdminAction(`retry-${taskId}`, () => api.retryTrainingTask(taskId))
const cancelTrainingTask = (taskId: number) => runAdminAction(`cancel-${taskId}`, () => api.cancelTrainingTask(taskId))

const scopeTasks = (scope: TrainingScope) => overview.value?.recent_training_tasks.filter((task) => task.scope === scope) ?? []
const scopeRuns = (scope: TrainingScope) => overview.value?.recent_training_runs.filter((run) => run.scope === scope) ?? []

function formatDate(value: string | null | undefined): string {
    return value ? new Date(value).toLocaleString() : '--'
}

function formatLoss(value: number | null | undefined): string {
    return value === null || value === undefined ? '--' : value.toFixed(4)
}

function formatDrift(value: number | null | undefined): string {
    return value === null || value === undefined ? '--' : `${value.toFixed(3)}s`
}

function formatTaskIds(ids: number[] | null | undefined): string {
    return !ids || ids.length === 0 ? 'none' : ids.join(', ')
}

function formatClientId(value: string | null | undefined): string {
    if (!value) return '--'
    return value.length <= 16 ? value : `${value.slice(0, 8)}...${value.slice(-4)}`
}

function compactUserAgent(value: string | null | undefined): string {
    if (!value) return '--'
    return value.length <= 48 ? value : `${value.slice(0, 45)}...`
}

function statusPillClass(status: string | null | undefined): string {
    switch (status) {
        case 'completed':
        case 'trained':
        case 'direct':
            return 'status-pill--ready'
        case 'processing':
        case 'training':
        case 'hls-js':
        case 'hls-native':
            return 'status-pill--working'
        case 'failed':
        case 'error':
            return 'status-pill--failed'
        default:
            return 'status-pill--queued'
    }
}

function radioEventPillClass(event: string): string {
    if (event === 'waiting' || event === 'stalled' || event === 'audio-error' || event.startsWith('hls-error:')) return 'status-pill--failed'
    if (event === 'native-hls-fallback-armed' || event.startsWith('recovery:')) return 'status-pill--working'
    if (event === 'play' || event === 'loadedmetadata') return 'status-pill--ready'
    return 'status-pill--queued'
}
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card overflow-hidden !p-0">
            <div class="grid gap-0 xl:grid-cols-[minmax(0,1.45fr)_23rem]">
                <div class="bg-[linear-gradient(135deg,rgba(14,116,144,0.12),rgba(249,115,22,0.10),rgba(255,255,255,0.94))] px-6 py-7 md:px-8">
                    <p class="section-kicker !text-sky-700">Admin</p>
                    <h1 class="page-title">Operations cockpit</h1>
                    <p class="page-copy max-w-3xl">Service health, training control, and live radio telemetry in one admin workspace.</p>
                    <div class="mt-5 flex flex-wrap gap-3 text-xs font-semibold uppercase tracking-[0.16em]">
                        <span :class="['status-pill', overview?.training_queue_drift.healthy ? 'status-pill--ready' : 'status-pill--failed']">{{ overview?.training_queue_drift.healthy ? 'Queue aligned' : 'Queue drift' }}</span>
                        <span :class="['status-pill', radioDiagnostics?.available ? 'status-pill--working' : 'status-pill--queued']">{{ radioDiagnostics?.available ? 'Radio telemetry active' : 'Radio telemetry idle' }}</span>
                        <span class="status-pill status-pill--queued">Loaded {{ formatDate(lastLoadedAt) }}</span>
                    </div>
                </div>
                <div class="border-t border-slate-200/80 bg-white/80 px-6 py-6 xl:border-l xl:border-t-0">
                    <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Quick actions</div>
                    <div class="mt-4 grid gap-3">
                        <button class="action-primary !justify-center !rounded-xl !px-4 !py-3" :disabled="loading || activeActionKey !== ''" @click="loadOverview">{{ loading ? 'Refreshing...' : 'Refresh overview' }}</button>
                        <button class="action-secondary !justify-center !rounded-xl !px-4 !py-3" :disabled="activeActionKey !== ''" @click="repairTrainingQueue">{{ activeActionKey === 'repair-training-queue' ? 'Repairing queue...' : 'Repair training queue' }}</button>
                    </div>
                    <div class="mt-5 space-y-2 rounded-[1.25rem] border border-slate-200 bg-slate-50/85 px-4 py-4 text-sm text-slate-600">
                        <div class="flex items-center justify-between gap-4"><span class="font-medium text-slate-950">Telemetry batches</span><span>{{ radioDiagnostics?.batches_seen ?? 0 }}</span></div>
                        <div class="flex items-center justify-between gap-4"><span class="font-medium text-slate-950">Dropped events</span><span :class="(radioDiagnostics?.dropped_events ?? 0) > 0 ? 'text-amber-700 font-semibold' : 'text-emerald-600 font-semibold'">{{ radioDiagnostics?.dropped_events ?? 0 }}</span></div>
                        <div class="flex items-center justify-between gap-4"><span class="font-medium text-slate-950">Log files</span><span>{{ radioDiagnostics?.log_files.length ?? 0 }}</span></div>
                    </div>
                </div>
            </div>
        </section>

        <div v-if="loading" class="surface-card text-sm text-slate-500">Loading admin overview...</div>
        <div v-else-if="error" class="surface-card border-rose-200 bg-rose-50 text-sm font-medium text-rose-700 shadow-none">{{ error }}</div>

        <template v-else-if="overview">
            <div v-if="actionMessage" class="rounded-[1.4rem] border border-emerald-200 bg-emerald-50 px-5 py-4 text-sm font-medium text-emerald-700">{{ actionMessage }}</div>
            <div v-if="actionError" class="rounded-[1.4rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-700">{{ actionError }}</div>

            <section class="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Users</div><div class="mt-3 text-3xl font-bold text-slate-950">{{ overview.users.total }}</div><div class="mt-2 text-sm text-slate-500">Admins: {{ overview.users.admins }} | Members: {{ overview.users.members }}</div></article>
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Tracks</div><div class="mt-3 text-3xl font-bold text-slate-950">{{ overview.tracks.total }}</div><div class="mt-2 text-sm text-slate-500">Completed: {{ overview.tracks.completed }} | Failed: {{ overview.tracks.failed }}</div></article>
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Training queue</div><div class="mt-3 text-3xl font-bold text-slate-950">{{ overview.training_queue.pending }}</div><div class="mt-2 text-sm text-slate-500">Processing: {{ overview.training_queue.processing }} | Failed: {{ overview.training_queue.failed }}</div></article>
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Radio diagnostics</div><div class="mt-3 text-3xl font-bold text-slate-950">{{ radioDiagnostics?.total_events ?? 0 }}</div><div class="mt-2 text-sm text-slate-500">Clients: {{ radioDiagnostics?.unique_clients ?? 0 }} | Tracks: {{ radioDiagnostics?.unique_tracks ?? 0 }}</div></article>
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Preference model</div><div class="mt-3 text-2xl font-bold text-slate-950">{{ overview.model.status }}</div><div class="mt-2 text-sm text-slate-500">Trained users: {{ overview.model.trained_users }} | Training: {{ overview.model.training_users }}</div></article>
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Worst drift</div><div class="mt-3 text-3xl font-bold text-slate-950">{{ formatDrift(radioDiagnostics?.max_drift) }}</div><div class="mt-2 text-sm text-slate-500">Average: {{ formatDrift(radioDiagnostics?.average_drift) }}</div></article>
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Ratings</div><div class="mt-3 text-3xl font-bold text-slate-950">{{ overview.ratings.total }}</div><div class="mt-2 text-sm text-slate-500">4-5 stars: {{ overview.ratings.high }} | 1-2 stars: {{ overview.ratings.low }}</div></article>
                <article class="metric-card"><div class="text-sm font-semibold text-slate-500">Playlists</div><div class="mt-3 text-3xl font-bold text-slate-950">{{ overview.playlists.total }}</div><div class="mt-2 text-sm text-slate-500">AI-generated: {{ overview.playlists.ai_generated }}</div></article>
            </section>

            <section class="surface-card !rounded-[1.8rem] !p-6">
                <div class="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                    <div>
                        <p class="section-kicker !text-rose-700">Radio Diagnostics</p>
                        <h2 class="text-2xl font-semibold text-slate-950">Client telemetry</h2>
                        <p class="mt-2 text-sm leading-6 text-slate-500">Recent radio playback diagnostics shipped from clients, sampled on the frontend and rate-capped on ingestion.</p>
                    </div>
                    <div class="rounded-[1.25rem] border border-slate-200 bg-slate-50/85 px-4 py-4 text-sm text-slate-600">
                        <div class="font-medium text-slate-950">Latest telemetry</div>
                        <div class="mt-2">{{ formatDate(radioDiagnostics?.latest_logged_at) }}</div>
                        <div class="mt-3 text-xs uppercase tracking-[0.16em] text-slate-400">Log files</div>
                        <div class="mt-2 break-all text-xs text-slate-500">{{ (radioDiagnostics?.log_files ?? []).join(', ') || 'No diagnostics files yet' }}</div>
                    </div>
                </div>

                <div class="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/85 p-4"><div class="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Waiting / stalled</div><div class="mt-2 text-2xl font-bold text-slate-950">{{ radioDiagnostics?.waiting_events ?? 0 }} / {{ radioDiagnostics?.stalled_events ?? 0 }}</div></div>
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/85 p-4"><div class="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Errors / fallbacks</div><div class="mt-2 text-2xl font-bold text-slate-950">{{ radioDiagnostics?.error_events ?? 0 }} / {{ radioDiagnostics?.fallback_events ?? 0 }}</div></div>
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/85 p-4"><div class="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Dropped / throttled</div><div class="mt-2 text-2xl font-bold" :class="(radioDiagnostics?.dropped_events ?? 0) > 0 ? 'text-amber-700' : 'text-emerald-600'">{{ radioDiagnostics?.dropped_events ?? 0 }} / {{ radioDiagnostics?.throttled_batches ?? 0 }}</div></div>
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/85 p-4"><div class="text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">Clients / tracks</div><div class="mt-2 text-2xl font-bold text-slate-950">{{ radioDiagnostics?.unique_clients ?? 0 }} / {{ radioDiagnostics?.unique_tracks ?? 0 }}</div></div>
                </div>

                <div class="mt-5 flex flex-wrap gap-2">
                    <span v-for="item in topRadioEvents" :key="item.event" class="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-slate-600">{{ item.event }} · {{ item.count }}</span>
                    <span v-if="topRadioEvents.length === 0" class="text-sm text-slate-400">No telemetry events recorded yet.</span>
                </div>

                <div class="mt-5 overflow-x-auto">
                    <div class="table-shell min-w-full">
                        <table class="min-w-full text-sm">
                            <thead class="table-header"><tr><th class="px-4 py-3 text-left">Event</th><th class="px-4 py-3 text-left">Station / Client</th><th class="px-4 py-3 text-left">Transport</th><th class="px-4 py-3 text-left">Track</th><th class="px-4 py-3 text-left">Drift</th><th class="px-4 py-3 text-left">When</th></tr></thead>
                            <tbody class="divide-y divide-slate-200/80">
                                <tr v-for="event in recentRadioEvents" :key="`${event.logged_at}-${event.client_id}-${event.event}-${event.track_id}`" class="table-row">
                                    <td class="px-4 py-4 align-top text-slate-800"><div><span :class="['status-pill', radioEventPillClass(event.event)]">{{ event.event }}</span></div><div class="table-meta mt-2">UA: {{ compactUserAgent(event.user_agent) }}</div><div v-if="event.media_error_code !== null" class="table-meta mt-1">MediaError: {{ event.media_error_code }}</div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div class="font-medium text-slate-950">{{ event.station_slug ?? '--' }}</div><div class="table-meta mt-1">Client: {{ formatClientId(event.client_id) }}</div><div class="table-meta mt-1">IP: {{ event.client_ip ?? '--' }}</div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div><span :class="['status-pill', statusPillClass(event.playback_mode)]">{{ event.playback_mode ?? '--' }}</span></div><div class="table-meta mt-2">Pref: {{ event.transport_preference ?? '--' }}</div><div class="table-meta mt-1">Bitrate: {{ event.selected_bitrate ?? '--' }}</div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div class="font-medium text-slate-950">{{ event.track_id ?? '--' }}</div><div class="table-meta mt-1">Ready: {{ event.ready_state ?? '--' }}</div><div class="table-meta mt-1">Paused: {{ event.paused === null ? '--' : (event.paused ? 'yes' : 'no') }}</div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div class="font-medium text-slate-950">{{ formatDrift(event.drift) }}</div><div class="table-meta mt-1">Rate: {{ event.playback_rate === null ? '--' : event.playback_rate.toFixed(2) }}</div><div class="table-meta mt-1">Waiting: {{ event.waiting === null ? '--' : (event.waiting ? 'yes' : 'no') }}</div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div>{{ formatDate(event.timestamp ?? event.logged_at) }}</div><div class="table-meta mt-1">{{ formatDate(event.logged_at) }}</div></td>
                                </tr>
                                <tr v-if="recentRadioEvents.length === 0"><td colspan="6" class="px-4 py-8 text-center text-sm text-slate-400">No radio telemetry available yet.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <section class="surface-card !rounded-[1.75rem] !p-5">
                <h2 class="text-lg font-semibold text-slate-950">Training queue drift</h2>
                <p class="mt-1 text-sm text-slate-500">Explicit comparison between PostgreSQL pending tasks and Redis queue state.</p>
                <div class="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4"><div class="text-xs font-semibold uppercase tracking-wide text-slate-500">Pending in DB</div><div class="mt-2 text-2xl font-bold text-slate-950">{{ overview.training_queue_drift.pending_db_count }}</div></div>
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4"><div class="text-xs font-semibold uppercase tracking-wide text-slate-500">Queued in Redis</div><div class="mt-2 text-2xl font-bold text-slate-950">{{ overview.training_queue_drift.queued_redis_count }}</div></div>
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4"><div class="text-xs font-semibold uppercase tracking-wide text-slate-500">Missing from Redis</div><div class="mt-2 text-2xl font-bold" :class="overview.training_queue_drift.missing_from_redis_count === 0 ? 'text-emerald-600' : 'text-amber-700'">{{ overview.training_queue_drift.missing_from_redis_count }}</div></div>
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4"><div class="text-xs font-semibold uppercase tracking-wide text-slate-500">Stale in Redis</div><div class="mt-2 text-2xl font-bold" :class="overview.training_queue_drift.stale_in_redis_count === 0 ? 'text-emerald-600' : 'text-amber-700'">{{ overview.training_queue_drift.stale_in_redis_count }}</div></div>
                </div>
                <div class="mt-4 grid gap-4 xl:grid-cols-2">
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4 text-sm text-slate-700"><div class="font-medium text-slate-950">Pending DB ids</div><div class="mt-2 break-all">{{ formatTaskIds(overview.training_queue_drift.pending_db_task_ids) }}</div></div>
                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4 text-sm text-slate-700"><div class="font-medium text-slate-950">Queued Redis ids</div><div class="mt-2 break-all">{{ formatTaskIds(overview.training_queue_drift.queued_redis_task_ids) }}</div></div>
                </div>
            </section>

            <section class="surface-card !rounded-[1.75rem] !p-5">
                <h2 class="text-lg font-semibold text-slate-950">User model state</h2>
                <p class="mt-1 text-sm text-slate-500">Per-user model status, signals, queue pressure, and latest run output.</p>
                <div class="mt-4 overflow-x-auto">
                    <div class="table-shell min-w-full">
                        <table class="min-w-full text-sm">
                            <thead class="table-header"><tr><th class="px-4 py-3 text-left">User</th><th class="px-4 py-3 text-left">Model</th><th class="px-4 py-3 text-left">Signals</th><th class="px-4 py-3 text-left">Tasks</th><th class="px-4 py-3 text-left">Latest run</th></tr></thead>
                            <tbody class="divide-y divide-slate-200/80">
                                <tr v-for="userModel in overview.user_models" :key="userModel.user_id" class="table-row">
                                    <td class="px-4 py-4 align-top text-slate-800"><div class="font-medium text-slate-950">{{ userModel.name }}</div><div class="table-meta mt-1">{{ userModel.email }}</div><div class="mt-2"><span class="badge-soft">{{ userModel.role }}</span></div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div><span :class="['status-pill', statusPillClass(userModel.model_status)]">{{ userModel.model_status }}</span></div><div class="table-meta mt-2">{{ formatDate(userModel.model_trained_at) }}</div><div class="table-meta mt-2 break-all">{{ userModel.model_path ?? '--' }}</div><button class="action-secondary mt-3 !rounded-xl !px-3 !py-2 !text-xs" :disabled="activeActionKey !== '' || userModel.model_status === 'training'" @click="retrainUser(userModel.user_id)">{{ activeActionKey === `retrain-${userModel.user_id}` ? 'Queueing...' : 'Retrain' }}</button></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div>Likes: {{ userModel.liked_tracks }}</div><div>Liked fragments: {{ userModel.liked_fragments }}</div><div>Disliked fragments: {{ userModel.disliked_fragments }}</div><div>Frequent tracks: {{ userModel.frequently_played_tracks }}</div><div>Rated segments: {{ userModel.rated_segments }}</div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><div>Pending: {{ userModel.training_tasks_pending }}</div><div>Processing: {{ userModel.training_tasks_processing }}</div><div>Completed: {{ userModel.training_tasks_completed }}</div><div>Failed: {{ userModel.training_tasks_failed }}</div></td>
                                    <td class="px-4 py-4 align-top text-slate-700"><template v-if="userModel.latest_training_run"><div class="font-medium text-slate-950">Run #{{ userModel.latest_training_run.id }}</div><div class="table-meta mt-1">Task: {{ userModel.latest_training_run.queue_task_id ?? '--' }} | {{ userModel.latest_training_run.queue_task_status ?? '--' }}</div><div class="mt-2">Loss: {{ formatLoss(userModel.latest_training_run.final_loss) }}</div><div>Positives: {{ userModel.latest_training_run.positives_total }}</div><div>Negatives: {{ userModel.latest_training_run.explicit_negatives_total }}</div></template><span v-else class="table-meta">No runs</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <div class="grid gap-6 xl:grid-cols-2">
                <section v-for="scope in trainingScopes" :key="`${scope.key}-tasks`" class="surface-card !rounded-[1.75rem] !p-5">
                    <h2 class="text-lg font-semibold text-slate-950">{{ scope.title }} training tasks</h2>
                    <div class="mt-4 overflow-x-auto">
                        <div class="table-shell min-w-full">
                            <table class="min-w-full text-sm">
                                <thead class="table-header"><tr><th class="px-4 py-3 text-left">Task</th><th class="px-4 py-3 text-left">User</th><th class="px-4 py-3 text-left">Status</th><th class="px-4 py-3 text-left">Created</th></tr></thead>
                                <tbody class="divide-y divide-slate-200/80">
                                    <tr v-for="task in scopeTasks(scope.key)" :key="task.id" class="table-row">
                                        <td class="px-4 py-4 align-top text-slate-800"><div class="font-medium text-slate-950">#{{ task.id }}</div><div v-if="task.error_message" class="mt-2 rounded-[0.95rem] border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">{{ task.error_message }}</div><div class="mt-3 flex flex-wrap gap-2"><button v-if="task.status === 'failed'" class="action-secondary !rounded-xl !px-3 !py-2 !text-xs" :disabled="activeActionKey !== ''" @click="retryTrainingTask(task.id)">{{ activeActionKey === `retry-${task.id}` ? 'Retrying...' : 'Retry' }}</button><button v-if="task.status === 'pending'" class="action-danger !rounded-xl !px-3 !py-2 !text-xs" :disabled="activeActionKey !== ''" @click="cancelTrainingTask(task.id)">{{ activeActionKey === `cancel-${task.id}` ? 'Cancelling...' : 'Cancel pending' }}</button></div></td>
                                        <td class="px-4 py-4 align-top text-slate-700"><div>{{ task.user_name ?? 'Global/legacy' }}</div><div class="table-meta mt-1">{{ task.user_email ?? '--' }}</div></td>
                                        <td class="px-4 py-4 align-top text-slate-700"><span :class="['status-pill', statusPillClass(task.status)]">{{ task.status }}</span></td>
                                        <td class="px-4 py-4 align-top text-slate-700">{{ formatDate(task.created_at) }}</td>
                                    </tr>
                                    <tr v-if="scopeTasks(scope.key).length === 0"><td colspan="4" class="px-4 py-8 text-center text-sm text-slate-400">No recent {{ scope.title.toLowerCase() }} tasks.</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>

                <section v-for="scope in trainingScopes" :key="`${scope.key}-runs`" class="surface-card !rounded-[1.75rem] !p-5">
                    <h2 class="text-lg font-semibold text-slate-950">{{ scope.title }} training runs</h2>
                    <div class="mt-4 overflow-x-auto">
                        <div class="table-shell min-w-full">
                            <table class="min-w-full text-sm">
                                <thead class="table-header"><tr><th class="px-4 py-3 text-left">Run</th><th class="px-4 py-3 text-left">User</th><th class="px-4 py-3 text-left">Loss</th><th class="px-4 py-3 text-left">Signals</th><th class="px-4 py-3 text-left">Finished</th></tr></thead>
                                <tbody class="divide-y divide-slate-200/80">
                                    <tr v-for="run in scopeRuns(scope.key)" :key="run.id" class="table-row">
                                        <td class="px-4 py-4 align-top text-slate-800"><div class="font-medium text-slate-950">#{{ run.id }}</div><div class="table-meta mt-1">Task: {{ run.queue_task_id ?? '--' }} | {{ run.queue_task_status ?? '--' }}</div></td>
                                        <td class="px-4 py-4 align-top text-slate-700"><div>{{ run.user_name ?? 'Global/legacy' }}</div><div class="table-meta mt-1">{{ run.user_email ?? '--' }}</div></td>
                                        <td class="px-4 py-4 align-top text-slate-700">{{ formatLoss(run.final_loss) }}</td>
                                        <td class="px-4 py-4 align-top text-slate-700"><div>Total+: {{ run.positives_total }}</div><div>Tracks: {{ run.positives_tracks }} | Fragments: {{ run.positives_fragments }} | Segments: {{ run.positives_segments }}</div><div>Negatives: {{ run.explicit_negatives_total }}</div></td>
                                        <td class="px-4 py-4 align-top text-slate-700">{{ formatDate(run.finished_at ?? run.created_at) }}</td>
                                    </tr>
                                    <tr v-if="scopeRuns(scope.key).length === 0"><td colspan="5" class="px-4 py-8 text-center text-sm text-slate-400">No recent {{ scope.title.toLowerCase() }} runs.</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>
            </div>
        </template>
    </div>
</template>
