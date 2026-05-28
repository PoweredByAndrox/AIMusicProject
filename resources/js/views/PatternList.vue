<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import api from '@/api/client'
import type { Pattern } from '@/types'

const router = useRouter()
const patterns = ref<Pattern[]>([])
const loading = ref(false)
const showCreateDialog = ref(false)

const newPattern = ref({
    name: '',
    description: '',
    representation_type: 'chroma' as 'chroma' | 'pitch_contour' | 'midi',
    transpose_invariant: true,
    tempo_invariant: true,
    match_threshold: 0.7,
    importance: 1,
})

const patternStats = computed(() => ([
    {
        label: 'Patterns',
        value: patterns.value.length.toString(),
        detail: 'saved search definitions',
    },
    {
        label: 'Examples',
        value: patterns.value.reduce((sum, pattern) => sum + (pattern.examples_count || 0), 0).toString(),
        detail: 'reference fragments across all patterns',
    },
    {
        label: 'Matches',
        value: patterns.value.reduce((sum, pattern) => sum + (pattern.matches_count || 0), 0).toString(),
        detail: 'stored pattern hits in the library',
    },
]))

const loadPatterns = async () => {
    loading.value = true
    try {
        patterns.value = await api.getPatterns()
    } catch (error) {
        console.error('Failed to load patterns:', error)
    } finally {
        loading.value = false
    }
}

const createPattern = async () => {
    if (!newPattern.value.name.trim()) return

    try {
        const pattern = await api.createPattern(newPattern.value)
        patterns.value.unshift(pattern)
        showCreateDialog.value = false
        newPattern.value = {
            name: '',
            description: '',
            representation_type: 'chroma',
            transpose_invariant: true,
            tempo_invariant: true,
            match_threshold: 0.7,
            importance: 1,
        }
        router.push({ name: 'pattern-detail', params: { id: pattern.id } })
    } catch (error) {
        console.error('Failed to create pattern:', error)
    }
}

const deletePattern = async (pattern: Pattern) => {
    if (!confirm(`Delete pattern "${pattern.name}"?`)) return

    try {
        await api.deletePattern(pattern.id)
        patterns.value = patterns.value.filter((entry) => entry.id !== pattern.id)
    } catch (error) {
        console.error('Failed to delete pattern:', error)
    }
}

const openPattern = (pattern: Pattern) => {
    router.push({ name: 'pattern-detail', params: { id: pattern.id } })
}

const getRepTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
        chroma: 'Chroma harmony',
        pitch_contour: 'Pitch contour',
        midi: 'MIDI notes',
    }
    return labels[type] || type
}

onMounted(() => {
    loadPatterns()
})
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p class="section-kicker">Patterns</p>
                    <h1 class="page-title">Pattern library</h1>
                    <p class="page-copy">
                        Build reusable harmonic or melodic fingerprints from track fragments, then scan the library for related passages.
                    </p>
                </div>
                <button type="button" class="action-primary" @click="showCreateDialog = true">
                    Create pattern
                </button>
            </div>
        </section>

        <section class="grid gap-4 md:grid-cols-3">
            <article v-for="stat in patternStats" :key="stat.label" class="metric-card">
                <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{{ stat.label }}</p>
                <p class="mt-3 text-3xl font-semibold text-slate-950">{{ stat.value }}</p>
                <p class="mt-2 text-sm text-slate-500">{{ stat.detail }}</p>
            </article>
        </section>

        <section v-if="loading" class="surface-card text-center">
            <p class="text-lg font-semibold text-slate-900">Loading patterns...</p>
            <p class="mt-2 text-sm text-slate-500">Fetching saved search definitions and their usage counts.</p>
        </section>

        <section v-else-if="patterns.length === 0" class="surface-card text-center">
            <p class="text-lg font-semibold text-slate-900">No patterns yet</p>
            <p class="mt-2 text-sm text-slate-500">Create your first pattern to search for repeating ideas across the library.</p>
            <div class="mt-5">
                <button type="button" class="action-primary" @click="showCreateDialog = true">
                    Create the first pattern
                </button>
            </div>
        </section>

        <section v-else class="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            <article
                v-for="pattern in patterns"
                :key="pattern.id"
                class="surface-card !rounded-[1.7rem] !p-5 cursor-pointer transition hover:-translate-y-1"
                @click="openPattern(pattern)"
            >
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <h2 class="truncate text-xl font-semibold text-slate-950">{{ pattern.name }}</h2>
                        <p v-if="pattern.description" class="mt-2 line-clamp-2 text-sm leading-6 text-slate-500">
                            {{ pattern.description }}
                        </p>
                    </div>
                    <button
                        type="button"
                        class="action-danger !rounded-xl !px-3 !py-2 !text-xs"
                        title="Delete pattern"
                        @click.stop="deletePattern(pattern)"
                    >
                        Delete
                    </button>
                </div>

                <div class="mt-5 flex flex-wrap gap-2">
                    <span class="badge-soft !border-purple-200 !bg-purple-50 !text-purple-700">
                        {{ getRepTypeLabel(pattern.representation_type) }}
                    </span>
                    <span v-if="pattern.transpose_invariant" class="badge-soft !border-sky-200 !bg-sky-50 !text-sky-700">
                        Transpose-aware
                    </span>
                    <span v-if="pattern.tempo_invariant" class="badge-soft !border-emerald-200 !bg-emerald-50 !text-emerald-700">
                        Tempo-flexible
                    </span>
                    <span class="badge-soft">
                        Importance {{ pattern.importance }}
                    </span>
                </div>

                <div class="mt-5 grid gap-3 md:grid-cols-2">
                    <div class="rounded-2xl bg-slate-50 p-4">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Examples</div>
                        <div class="mt-2 text-2xl font-semibold text-slate-950">{{ pattern.examples_count || 0 }}</div>
                    </div>
                    <div class="rounded-2xl bg-slate-50 p-4">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Matches</div>
                        <div class="mt-2 text-2xl font-semibold text-slate-950">{{ pattern.matches_count || 0 }}</div>
                    </div>
                </div>
            </article>
        </section>

        <div v-if="showCreateDialog" class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 px-4">
            <div class="surface-card w-full max-w-xl !rounded-[1.8rem] !p-6">
                <h2 class="text-2xl font-semibold text-slate-950">Create pattern</h2>

                <div class="mt-6 space-y-4">
                    <div>
                        <label class="field-label" for="pattern-name">Name</label>
                        <input id="pattern-name" v-model="newPattern.name" type="text" class="field-control" placeholder="Dubstep wobble bass">
                    </div>

                    <div>
                        <label class="field-label" for="pattern-description">Description</label>
                        <textarea id="pattern-description" v-model="newPattern.description" class="field-textarea" rows="3" placeholder="What makes this pattern distinct?"></textarea>
                    </div>

                    <div>
                        <label class="field-label" for="pattern-type">Representation</label>
                        <select id="pattern-type" v-model="newPattern.representation_type" class="field-select">
                            <option value="chroma">Chroma: harmonic profile</option>
                            <option value="pitch_contour">Pitch contour: melodic shape</option>
                            <option value="midi">MIDI: note-level pattern</option>
                        </select>
                    </div>

                    <div class="grid gap-4 md:grid-cols-2">
                        <label class="flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-medium text-slate-700">
                            <input v-model="newPattern.transpose_invariant" type="checkbox" class="field-checkbox">
                            Allow transposition
                        </label>
                        <label class="flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm font-medium text-slate-700">
                            <input v-model="newPattern.tempo_invariant" type="checkbox" class="field-checkbox">
                            Allow tempo variation
                        </label>
                    </div>

                    <div>
                        <label class="field-label">Match threshold: {{ (newPattern.match_threshold * 100).toFixed(0) }}%</label>
                        <input v-model.number="newPattern.match_threshold" type="range" min="0.5" max="0.95" step="0.05" class="w-full accent-blue-600">
                    </div>

                    <div>
                        <label class="field-label">Importance</label>
                        <div class="flex flex-wrap gap-2">
                            <button
                                v-for="level in 4"
                                :key="level"
                                type="button"
                                :class="newPattern.importance === level - 1 ? 'tab-button tab-button--active' : 'tab-button tab-button--idle'"
                                @click="newPattern.importance = level - 1"
                            >
                                {{ level - 1 }}
                            </button>
                        </div>
                    </div>
                </div>

                <div class="mt-6 flex justify-end gap-3">
                    <button type="button" class="action-secondary" @click="showCreateDialog = false">
                        Cancel
                    </button>
                    <button type="button" class="action-primary" :disabled="!newPattern.name.trim()" @click="createPattern">
                        Create
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>
