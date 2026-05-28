<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '@/api/client'
import type { Pattern, PatternExample, PatternMatch, Track } from '@/types'

const route = useRoute()
const router = useRouter()

const pattern = ref<Pattern | null>(null)
const matches = ref<PatternMatch[]>([])
const loading = ref(true)
const searching = ref(false)
const searchMessage = ref('')

const showAddExample = ref(false)
const tracks = ref<Track[]>([])
const selectedTrack = ref<Track | null>(null)
const exampleStartTime = ref(0)
const exampleEndTime = ref(10)

const patternId = computed(() => parseInt(route.params.id as string, 10))

const loadPattern = async () => {
    loading.value = true
    try {
        pattern.value = await api.getPattern(patternId.value)
        matches.value = await api.getPatternMatches(patternId.value)
    } catch (error) {
        console.error('Failed to load pattern:', error)
        router.push({ name: 'patterns' })
    } finally {
        loading.value = false
    }
}

const loadTracks = async () => {
    try {
        const response = await api.getTracks({ status: 'completed', per_page: 100 })
        tracks.value = response.data
    } catch (error) {
        console.error('Failed to load tracks:', error)
    }
}

const searchPattern = async () => {
    if (!pattern.value) return

    searching.value = true
    searchMessage.value = 'Pattern search started...'

    try {
        const result = await api.searchPattern(pattern.value.id)
        searchMessage.value = `Search started (task_id: ${result.task_id}). Results will refresh automatically.`

        setTimeout(async () => {
            matches.value = await api.getPatternMatches(patternId.value)
            searchMessage.value = `Found ${matches.value.length} match(es).`
            searching.value = false
        }, 5000)
    } catch (error: any) {
        searchMessage.value = error.response?.data?.error || 'Search failed.'
        searching.value = false
    }
}

const addExample = async () => {
    if (!pattern.value || !selectedTrack.value) return

    try {
        const example = await api.addPatternExample(pattern.value.id, {
            track_id: selectedTrack.value.id,
            start_time: exampleStartTime.value,
            end_time: exampleEndTime.value,
            is_positive: true,
        })

        if (!pattern.value.examples) {
            pattern.value.examples = []
        }
        example.track = selectedTrack.value
        pattern.value.examples.push(example)

        showAddExample.value = false
        selectedTrack.value = null
        exampleStartTime.value = 0
        exampleEndTime.value = 10
    } catch (error) {
        console.error('Failed to add example:', error)
    }
}

const removeExample = async (example: PatternExample) => {
    if (!pattern.value) return
    if (!confirm('Delete this example?')) return

    try {
        await api.removePatternExample(pattern.value.id, example.id)
        pattern.value.examples = pattern.value.examples?.filter((entry) => entry.id !== example.id)
    } catch (error) {
        console.error('Failed to remove example:', error)
    }
}

const confirmMatch = async (match: PatternMatch, confirmed: boolean) => {
    if (!pattern.value) return

    try {
        const updated = await api.confirmPatternMatch(pattern.value.id, match.id, confirmed)
        const index = matches.value.findIndex((entry) => entry.id === match.id)
        if (index !== -1) {
            matches.value[index] = updated
        }
    } catch (error) {
        console.error('Failed to confirm match:', error)
    }
}

const openTrack = (trackId: number) => {
    router.push({ name: 'track', params: { id: trackId } })
}

const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

const getRepTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
        chroma: 'Chroma',
        pitch_contour: 'Pitch contour',
        midi: 'MIDI',
    }
    return labels[type] || type
}

onMounted(() => {
    loadPattern()
    loadTracks()
})
</script>

<template>
    <div v-if="loading" class="surface-card text-center">
        <p class="text-lg font-semibold text-slate-900">Loading pattern...</p>
    </div>

    <div v-else-if="pattern" class="space-y-8">
        <section class="surface-card">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <router-link to="/patterns" class="badge-soft !normal-case !tracking-normal">Back to patterns</router-link>
                    <h1 class="page-title">{{ pattern.name }}</h1>
                    <p v-if="pattern.description" class="page-copy">{{ pattern.description }}</p>
                </div>
                <button
                    type="button"
                    class="action-primary"
                    :disabled="searching || !pattern.examples?.length"
                    @click="searchPattern"
                >
                    {{ searching ? 'Searching...' : 'Search library' }}
                </button>
            </div>
        </section>

        <section class="surface-card !rounded-[1.75rem] !p-5">
            <h2 class="text-xl font-semibold text-slate-950">Settings</h2>
            <div class="mt-4 flex flex-wrap gap-2">
                <span class="badge-soft !border-purple-200 !bg-purple-50 !text-purple-700">{{ getRepTypeLabel(pattern.representation_type) }}</span>
                <span v-if="pattern.transpose_invariant" class="badge-soft !border-sky-200 !bg-sky-50 !text-sky-700">Transpose aware</span>
                <span v-if="pattern.tempo_invariant" class="badge-soft !border-emerald-200 !bg-emerald-50 !text-emerald-700">Tempo flexible</span>
                <span class="badge-soft">Threshold {{ (pattern.match_threshold * 100).toFixed(0) }}%</span>
                <span class="badge-soft">Importance {{ pattern.importance }}</span>
            </div>
        </section>

        <section class="grid gap-6 xl:grid-cols-[1fr_1fr]">
            <div class="surface-card !rounded-[1.75rem] !p-5">
                <div class="flex items-center justify-between gap-4">
                    <div>
                        <h2 class="text-xl font-semibold text-slate-950">Examples</h2>
                        <p class="mt-1 text-sm text-slate-500">{{ pattern.examples?.length || 0 }} reference fragment(s)</p>
                    </div>
                    <button type="button" class="action-secondary" @click="showAddExample = true">
                        Add example
                    </button>
                </div>

                <div v-if="!pattern.examples?.length" class="empty-state mt-5">
                    <p class="text-lg font-semibold text-slate-900">No examples yet</p>
                    <p class="mt-2 text-sm text-slate-500">Add at least one reference fragment before running a library search.</p>
                </div>

                <div v-else class="mt-5 space-y-3">
                    <div
                        v-for="example in pattern.examples"
                        :key="example.id"
                        class="rounded-[1.3rem] border border-slate-200 bg-slate-50/90 p-4"
                    >
                        <div class="flex items-start justify-between gap-3">
                            <div>
                                <button type="button" class="text-left text-base font-semibold text-slate-950 hover:text-blue-700" @click="openTrack(example.track_id)">
                                    {{ example.track?.title || `Track #${example.track_id}` }}
                                </button>
                                <p class="mt-1 text-sm text-slate-500">{{ formatTime(example.start_time) }} - {{ formatTime(example.end_time) }}</p>
                            </div>
                            <button type="button" class="action-danger !rounded-xl !px-3 !py-2 !text-xs" @click="removeExample(example)">
                                Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            <div class="surface-card !rounded-[1.75rem] !p-5">
                <h2 class="text-xl font-semibold text-slate-950">Matches</h2>
                <p class="mt-1 text-sm text-slate-500">{{ matches.length }} match(es) currently stored.</p>

                <div v-if="searchMessage" class="mt-5 rounded-[1.3rem] border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-medium text-sky-700">
                    {{ searchMessage }}
                </div>

                <div v-if="matches.length === 0" class="empty-state mt-5">
                    <p class="text-lg font-semibold text-slate-900">No matches yet</p>
                    <p class="mt-2 text-sm text-slate-500">Run a search after adding examples to populate this list.</p>
                </div>

                <div v-else class="mt-5 space-y-3">
                    <div
                        v-for="match in matches"
                        :key="match.id"
                        class="rounded-[1.3rem] border p-4"
                        :class="{
                            'border-emerald-200 bg-emerald-50': match.confirmed === true,
                            'border-rose-200 bg-rose-50': match.confirmed === false,
                            'border-slate-200 bg-slate-50/90': match.confirmed === null,
                        }"
                    >
                        <div class="flex items-start justify-between gap-4">
                            <div class="min-w-0">
                                <div class="text-sm font-semibold uppercase tracking-[0.18em] text-slate-500">{{ (match.score * 100).toFixed(0) }}% match</div>
                                <button type="button" class="mt-2 text-left text-base font-semibold text-slate-950 hover:text-blue-700" @click="openTrack(match.track_id)">
                                    {{ match.track?.title || `Track #${match.track_id}` }}
                                </button>
                                <p class="mt-1 text-sm text-slate-500">
                                    {{ match.track?.artist || 'Unknown artist' }} • {{ formatTime(match.start_time) }} - {{ formatTime(match.end_time) }}
                                </p>
                            </div>
                            <div class="flex gap-2">
                                <button type="button" :class="match.confirmed === true ? 'action-primary !rounded-xl !px-3 !py-2 !text-xs' : 'action-secondary !rounded-xl !px-3 !py-2 !text-xs'" @click="confirmMatch(match, true)">
                                    Confirm
                                </button>
                                <button type="button" :class="match.confirmed === false ? 'action-danger !rounded-xl !px-3 !py-2 !text-xs' : 'action-secondary !rounded-xl !px-3 !py-2 !text-xs'" @click="confirmMatch(match, false)">
                                    Reject
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>

        <div v-if="showAddExample" class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 px-4">
            <div class="surface-card w-full max-w-2xl !rounded-[1.8rem] !p-6">
                <h2 class="text-2xl font-semibold text-slate-950">Add example</h2>

                <div class="mt-6 space-y-4">
                    <div>
                        <label class="field-label" for="pattern-example-track">Track</label>
                        <select id="pattern-example-track" v-model="selectedTrack" class="field-select">
                            <option :value="null">Select a completed track...</option>
                            <option v-for="track in tracks" :key="track.id" :value="track">
                                {{ track.title }} - {{ track.artist || 'Unknown' }}
                            </option>
                        </select>
                    </div>

                    <div class="grid gap-4 md:grid-cols-2">
                        <div>
                            <label class="field-label" for="pattern-example-start">Start (seconds)</label>
                            <input id="pattern-example-start" v-model.number="exampleStartTime" type="number" min="0" step="0.1" class="field-control">
                        </div>
                        <div>
                            <label class="field-label" for="pattern-example-end">End (seconds)</label>
                            <input id="pattern-example-end" v-model.number="exampleEndTime" type="number" :min="exampleStartTime + 1" step="0.1" class="field-control">
                        </div>
                    </div>

                    <div class="rounded-[1.3rem] border border-slate-200 bg-slate-50 px-4 py-3 text-sm leading-6 text-slate-600">
                        Tip: if you already marked the fragment on the track page, mirror the same start and end points here to keep the example and the stored waveform context aligned.
                    </div>
                </div>

                <div class="mt-6 flex justify-end gap-3">
                    <button type="button" class="action-secondary" @click="showAddExample = false">
                        Cancel
                    </button>
                    <button type="button" class="action-primary" :disabled="!selectedTrack" @click="addExample">
                        Add
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>
