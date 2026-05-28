<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { toast } from 'vue3-toastify'
import api from '@/api/client'
import ArtistCreditsInline from '@/components/ArtistCreditsInline.vue'
import WaveformPlayer from '@/components/WaveformPlayer.vue'
import { usePlayerStore } from '@/stores/playerStore'
import type { Track, Fragment, SimilarTrack } from '@/types'

/**
 * TrackDetail View
 * Requirements: 5.1, 6.1, 6.3, 10.3, 14.1
 */

const route = useRoute()
const router = useRouter()
const playerStore = usePlayerStore()

const track = ref<Track | null>(null)
const fragments = ref<Fragment[]>([])
const similarTracks = ref<SimilarTrack[]>([])
const loading = ref(true)
const editing = ref(false)
const editForm = ref({
    genre: '',
    bpm: 0,
    key: '',
    mood: '',
})

const trackId = computed(() => parseInt(route.params.id as string))
const audioUrl = computed(() => track.value ? `/storage/${track.value.file_path}` : '')
const tuningLabUrl = computed(() => track.value ? `/track/${track.value.id}/tuning-lab` : '#')
const waveformRef = ref<InstanceType<typeof WaveformPlayer> | null>(null)
const waveformLocalTime = ref(0)
const isTrackInGlobalPlayer = computed(() => playerStore.currentTrack?.id === trackId.value)

const genres = [
    'Rock', 'Pop', 'Jazz', 'Classical', 'Hip-Hop', 'Metal', 'Folk',
    'R&B', 'Country', 'Blues', 'Reggae', 'Punk', 'Soul', 'Indie',
    'House', 'Techno', 'Trance', 'Dubstep', 'Drum & Bass',
    'Ambient', 'EDM', 'Electro', 'Trap', 'Future Bass', 'Synthwave',
    'Lo-Fi', 'Deep House', 'Tech House', 'Hardstyle', 'Downtempo'
]
const moods = ['Happy', 'Sad', 'Energetic', 'Calm', 'Aggressive', 'Romantic']
const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

const loadTrack = async () => {
    loading.value = true
    try {
        track.value = await api.getTrack(trackId.value)
        fragments.value = await api.getFragments(trackId.value)

        if (track.value.analysis_status === 'completed') {
            try {
                const response = await api.getSimilarTracks(trackId.value, 5)
                similarTracks.value = response.similar_tracks || response || []
            } catch (e) {
                // Similar tracks may not be available
            }
        }

        editForm.value = {
            genre: track.value.genre || '',
            bpm: track.value.bpm || 0,
            key: track.value.key || '',
            mood: track.value.mood || '',
        }
    } catch (error) {
        console.error('Failed to load track:', error)
        router.push({ name: 'library' })
    } finally {
        loading.value = false
    }
}

const toggleLike = async () => {
    if (!track.value) return
    try {
        if (track.value.is_liked) {
            const response = await api.unlikeTrack(track.value.id)
            track.value.is_liked = false
            const effectiveRating = typeof response.feedback?.track_rating === 'number'
                ? response.feedback.track_rating
                : (typeof response.effective_rating === 'number' ? response.effective_rating : null)
            playerStore.rememberTrackRating(track.value.id, effectiveRating)
            void playerStore.enforceSkipDislikedTracks(track.value.id)
            toast.info('Removed from liked')
        } else {
            const response = await api.likeTrack(track.value.id)
            track.value.is_liked = true
            const effectiveRating = typeof response.feedback?.track_rating === 'number'
                ? response.feedback.track_rating
                : (typeof response.effective_rating === 'number' ? response.effective_rating : null)
            playerStore.rememberTrackRating(track.value.id, effectiveRating)
            void playerStore.enforceSkipDislikedTracks(track.value.id)
            toast.success('Added to liked!')
        }
    } catch (error) {
        console.error('Failed to toggle like:', error)
        toast.error('Failed to update like status')
    }
}

const saveEdits = async () => {
    if (!track.value) return
    try {
        track.value = await api.updateTrack(track.value.id, editForm.value)
        editing.value = false
        toast.success('Track updated!')
    } catch (error) {
        console.error('Failed to save edits:', error)
        toast.error('Failed to save changes')
    }
}

const createFragment = async (data: { start_time: number; end_time: number; is_liked?: boolean; comment?: string }) => {
    if (!track.value) return
    try {
        if (data.is_liked !== undefined) {
            const response = await fetch(`/api/tracks/${track.value.id}/fragments/rate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    start_time: data.start_time,
                    end_time: data.end_time,
                    is_liked: data.is_liked,
                    comment: data.comment || null,
                }),
            })
            const result = await response.json()

            if (!response.ok) {
                throw new Error(result.message || 'Failed to create fragment')
            }

            fragments.value.push(result.fragment)
            toast.success(data.is_liked ? 'Fragment marked as liked' : 'Fragment marked as disliked')
        } else {
            const fragment = await api.createFragment(track.value.id, {
                ...data,
                name: `Fragment ${fragments.value.length + 1}`,
            })
            fragments.value.push(fragment)
            toast.success('Fragment created!')
        }
    } catch (error) {
        console.error('Failed to create fragment:', error)
        toast.error('Failed to create fragment')
    }
}

const showPatternDialog = ref(false)
const patternSelection = ref<{ start_time: number; end_time: number } | null>(null)
const patternName = ref('')
const patternType = ref<'chroma' | 'pitch_contour'>('chroma')

const saveAsPattern = (data: { start_time: number; end_time: number }) => {
    patternSelection.value = data
    patternName.value = `${track.value?.title || 'Track'} - ${formatTime(data.start_time)}`
    showPatternDialog.value = true
}

const createPattern = async () => {
    if (!track.value || !patternSelection.value || !patternName.value.trim()) return

    try {
        const pattern = await api.createPattern({
            name: patternName.value,
            representation_type: patternType.value,
            transpose_invariant: true,
            tempo_invariant: true,
        })

        await api.addPatternExample(pattern.id, {
            track_id: track.value.id,
            start_time: patternSelection.value.start_time,
            end_time: patternSelection.value.end_time,
            is_positive: true,
        })

        toast.success('Pattern created!')
        showPatternDialog.value = false
        patternSelection.value = null
        patternName.value = ''

        router.push({ name: 'pattern-detail', params: { id: pattern.id } })
    } catch (error) {
        console.error('Failed to create pattern:', error)
        toast.error('Failed to create pattern')
    }
}

const deleteFragment = async (fragment: Fragment) => {
    if (!confirm('Delete this fragment?')) return
    try {
        await api.deleteFragment(fragment.id)
        fragments.value = fragments.value.filter((f) => f.id !== fragment.id)
        toast.info('Fragment deleted')
    } catch (error) {
        console.error('Failed to delete fragment:', error)
        toast.error('Failed to delete fragment')
    }
}

const toggleFragmentLike = async (fragment: Fragment) => {
    try {
        if (fragment.is_liked) {
            await api.unlikeFragment(fragment.id)
            fragment.is_liked = false
        } else {
            await api.likeFragment(fragment.id)
            fragment.is_liked = true
        }
    } catch (error) {
        console.error('Failed to toggle fragment like:', error)
        toast.error('Failed to update like')
    }
}

const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

const formatLikedAt = (value: string | null | undefined): string => {
    if (!value) {
        return 'Liked recently'
    }

    const timestamp = new Date(value)
    if (Number.isNaN(timestamp.getTime())) {
        return 'Liked recently'
    }

    return timestamp.toLocaleString([], {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    })
}

const trackStatusClass = (status: string | null | undefined) => {
    switch (status) {
        case 'completed':
            return 'status-pill--ready'
        case 'processing':
            return 'status-pill--working'
        case 'failed':
            return 'status-pill--failed'
        default:
            return 'status-pill--queued'
    }
}

const onWaveformPlayRequested = () => {
    if (!track.value) return

    if (isTrackInGlobalPlayer.value) {
        playerStore.play()
        return
    }

    playerStore.playTrack(track.value, false)
}

const onWaveformPauseRequested = () => {
    playerStore.pause()
}

const onWaveformSeekRequested = (time: number) => {
    if (!track.value) return

    if (isTrackInGlobalPlayer.value) {
        playerStore.seek(time)
        return
    }

    playerStore.playTrack(track.value, false)
    setTimeout(() => {
        if (playerStore.currentTrack?.id !== track.value?.id) return
        playerStore.seek(time)
    }, 100)
}

const onWaveformTimeUpdate = (time: number) => {
    waveformLocalTime.value = time
}

onMounted(() => {
    loadTrack()
})

watch(trackId, (newId, oldId) => {
    if (newId !== oldId) {
        similarTracks.value = []
        fragments.value = []
        editing.value = false
        loadTrack()
    }
})

watch(() => playerStore.currentTime, (time) => {
    if (!isTrackInGlobalPlayer.value || !waveformRef.value) return
    if (Math.abs(waveformLocalTime.value - time) <= 0.5) return
    waveformRef.value.syncSeekTo(time)
})

watch(() => playerStore.isPlaying, (playing) => {
    if (!isTrackInGlobalPlayer.value || !waveformRef.value) return
    if (playing) {
        waveformRef.value.syncPlay()
    } else {
        waveformRef.value.syncPause()
    }
})

watch(() => playerStore.currentTrack?.id, (currentTrackId) => {
    if (!waveformRef.value) return
    if (currentTrackId !== trackId.value) {
        waveformRef.value.syncPause()
    }
})
</script>

<template>
    <div v-if="loading" class="surface-card text-center text-sm text-slate-500">Loading...</div>

    <div v-else-if="track" class="space-y-6 pb-10">
        <section class="surface-card !p-6">
            <div class="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                <div>
                    <p class="section-kicker">Track Detail</p>
                    <h1 class="mt-3 text-3xl font-semibold text-slate-950">{{ track.title }}</h1>
                    <p class="mt-2 text-sm text-slate-500">
                        <ArtistCreditsInline
                            :credits="track.artist_credits"
                            :fallback="track.artist"
                            link-class="transition hover:text-sky-700"
                        />
                    </p>
                    <div class="mt-4 flex flex-wrap gap-2">
                        <span class="badge-soft">Duration: {{ formatTime(track.duration) }}</span>
                        <span class="badge-soft">Source: {{ track.source }}</span>
                        <span :class="['status-pill', trackStatusClass(track.analysis_status)]">
                            {{ track.analysis_status }}
                        </span>
                        <span v-if="track.manually_edited" class="status-pill status-pill--working">Manually edited</span>
                    </div>
                </div>

                <div class="flex flex-wrap gap-3">
                    <button
                        @click="toggleLike"
                        :class="[track.is_liked ? 'action-primary' : 'action-secondary', '!rounded-[1.2rem] !px-4 !py-3']"
                    >
                        {{ track.is_liked ? 'Liked' : 'Like' }}
                    </button>
                    <router-link
                        :to="{ name: 'segment-labeler', params: { id: track.id } }"
                        class="action-secondary !rounded-[1.2rem] !px-4 !py-3"
                    >
                        Label Segments
                    </router-link>
                    <a
                        :href="tuningLabUrl"
                        class="action-secondary !rounded-[1.2rem] !px-4 !py-3"
                    >
                        Tuning Lab
                    </a>
                    <button @click="editing = !editing" class="action-secondary !rounded-[1.2rem] !px-4 !py-3">
                        {{ editing ? 'Cancel' : 'Edit' }}
                    </button>
                </div>
            </div>

            <div class="mt-5 rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Friend likes</div>
                <div v-if="track.friend_likes && track.friend_likes.length > 0" class="mt-3 flex flex-wrap gap-3">
                    <div
                        v-for="friendLike in track.friend_likes"
                        :key="friendLike.id"
                        class="rounded-[1rem] border border-emerald-200 bg-emerald-50/80 px-3 py-2"
                    >
                        <div class="text-sm font-semibold text-emerald-900">{{ friendLike.name }}</div>
                        <div class="mt-1 text-xs text-emerald-700">{{ formatLikedAt(friendLike.liked_at) }}</div>
                    </div>
                </div>
                <div v-else class="mt-3 text-sm text-slate-500">
                    None of your accepted friends liked this track yet.
                </div>
            </div>
        </section>

        <WaveformPlayer
            ref="waveformRef"
            :audio-url="audioUrl"
            :fragments="fragments"
            :duration="track.duration"
            :external-audio="true"
            @fragment-created="createFragment"
            @save-as-pattern="saveAsPattern"
            @time-update="onWaveformTimeUpdate"
            @play-requested="onWaveformPlayRequested"
            @pause-requested="onWaveformPauseRequested"
            @seek="onWaveformSeekRequested"
        />

        <section class="surface-card">
            <div>
                <h2 class="text-xl font-semibold text-slate-950">Track Information</h2>
                <p class="mt-1 text-sm text-slate-500">Metadata, mood tags, and manual overrides for this track.</p>
            </div>

            <div v-if="editing" class="mt-6 grid gap-4 md:grid-cols-2">
                <div>
                    <label class="field-label">Genre</label>
                    <select v-model="editForm.genre" class="field-select">
                        <option value="">Select genre</option>
                        <option v-for="g in genres" :key="g" :value="g">{{ g }}</option>
                    </select>
                </div>
                <div>
                    <label class="field-label">BPM</label>
                    <input v-model.number="editForm.bpm" type="number" class="field-control" />
                </div>
                <div>
                    <label class="field-label">Key</label>
                    <select v-model="editForm.key" class="field-select">
                        <option value="">Select key</option>
                        <option v-for="k in keys" :key="k" :value="k">{{ k }}</option>
                    </select>
                </div>
                <div>
                    <label class="field-label">Mood</label>
                    <select v-model="editForm.mood" class="field-select">
                        <option value="">Select mood</option>
                        <option v-for="m in moods" :key="m" :value="m">{{ m }}</option>
                    </select>
                </div>
                <div class="md:col-span-2">
                    <button @click="saveEdits" class="action-primary">Save Changes</button>
                </div>
            </div>

            <div v-else class="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-4 text-sm">
                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Genre</div>
                    <div v-if="track.genre_tags && track.genre_tags.length > 0" class="mt-3 flex flex-wrap gap-2">
                        <span
                            v-for="(tag, idx) in track.genre_tags"
                            :key="idx"
                            class="badge-soft !tracking-[0.12em] !text-[0.7rem]"
                        >
                            {{ tag.tag }} {{ ((tag.prob || tag.score || 0) * 100).toFixed(0) }}%
                        </span>
                    </div>
                    <div v-else class="mt-3 font-medium text-slate-950">{{ track.genre || '-' }}</div>
                </div>

                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">BPM</div>
                    <div class="mt-3 text-lg font-semibold text-slate-950">{{ track.bpm ? Math.round(track.bpm) : '-' }}</div>
                </div>

                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Key</div>
                    <div class="mt-3 text-lg font-semibold text-slate-950">{{ track.key || '-' }}</div>
                </div>

                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Mood</div>
                    <div v-if="track.mood_tags && track.mood_tags.length > 0" class="mt-3 flex flex-wrap gap-2">
                        <span
                            v-for="(tag, idx) in track.mood_tags"
                            :key="idx"
                            class="badge-soft !tracking-[0.12em] !text-[0.7rem]"
                        >
                            {{ tag.tag }} {{ ((tag.prob || tag.score || 0) * 100).toFixed(0) }}%
                        </span>
                    </div>
                    <div v-else class="mt-3 font-medium text-slate-950">{{ track.mood || '-' }}</div>
                </div>

                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Duration</div>
                    <div class="mt-3 text-lg font-semibold text-slate-950">{{ formatTime(track.duration) }}</div>
                </div>

                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Source</div>
                    <div class="mt-3 font-medium text-slate-950">{{ track.source }}</div>
                </div>

                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Status</div>
                    <div class="mt-3">
                        <span :class="['status-pill', trackStatusClass(track.analysis_status)]">{{ track.analysis_status }}</span>
                    </div>
                </div>

                <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                    <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">Notes</div>
                    <div class="mt-3 font-medium text-slate-950">
                        {{ track.manually_edited ? 'Manual edits applied' : 'Auto analysis only' }}
                    </div>
                </div>
            </div>
        </section>

        <section class="surface-card">
            <div>
                <h2 class="text-xl font-semibold text-slate-950">Saved Fragments ({{ fragments.length }})</h2>
                <p class="mt-1 text-sm text-slate-500">Saved reactions and notes from the waveform review.</p>
            </div>

            <div v-if="fragments.length === 0" class="empty-state mt-6 text-sm text-slate-500">
                No saved fragments yet. Select a region on the waveform to rate or annotate it.
            </div>

            <div v-else class="mt-6 space-y-3">
                <div
                    v-for="fragment in fragments"
                    :key="fragment.id"
                    class="rounded-[1.35rem] border p-4"
                    :class="fragment.is_liked ? 'border-emerald-200 bg-emerald-50/80' : fragment.is_liked === false ? 'border-rose-200 bg-rose-50/80' : 'border-slate-200 bg-slate-50/80'"
                >
                    <div class="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                        <div class="min-w-0">
                            <div class="flex flex-wrap items-center gap-3">
                                <span
                                    :class="[
                                        'status-pill',
                                        fragment.is_liked === true
                                            ? 'status-pill--ready'
                                            : fragment.is_liked === false
                                                ? 'status-pill--failed'
                                                : 'status-pill--queued',
                                    ]"
                                >
                                    {{ fragment.is_liked === true ? 'Liked' : fragment.is_liked === false ? 'Disliked' : 'Unrated' }}
                                </span>
                                <span class="font-semibold text-slate-950">
                                    {{ formatTime(fragment.start_time) }} - {{ formatTime(fragment.end_time) }}
                                </span>
                                <span class="badge-soft !tracking-[0.12em] !text-[0.7rem]">
                                    {{ (fragment.end_time - fragment.start_time).toFixed(1) }}s
                                </span>
                            </div>
                            <div v-if="fragment.comment" class="mt-3 rounded-[1rem] border border-white/80 bg-white/70 px-3 py-2 text-sm italic text-slate-600">
                                "{{ fragment.comment }}"
                            </div>
                        </div>
                        <button @click="deleteFragment(fragment)" class="action-danger !rounded-xl !px-3 !py-2 !text-xs">
                            Delete
                        </button>
                    </div>
                </div>
            </div>
        </section>

        <section v-if="similarTracks.length > 0" class="surface-card">
            <div>
                <h2 class="text-xl font-semibold text-slate-950">Similar Tracks</h2>
                <p class="mt-1 text-sm text-slate-500">Nearest matches based on the current track representation.</p>
            </div>

            <div class="mt-6 space-y-3">
                <router-link
                    v-for="similar in similarTracks"
                    :key="similar.track.id"
                    :to="{ name: 'track', params: { id: similar.track.id } }"
                    class="flex items-center justify-between gap-4 rounded-[1.35rem] border border-slate-200 bg-white/80 px-4 py-3 transition hover:-translate-y-[1px] hover:border-sky-200 hover:bg-sky-50/60"
                >
                    <div class="min-w-0">
                        <div class="font-medium text-slate-950">{{ similar.track.title }}</div>
                        <div class="mt-1 text-sm text-slate-500">{{ similar.track.artist }}</div>
                    </div>
                    <span class="badge-soft">{{ (similar.similarity * 100).toFixed(0) }}% similar</span>
                </router-link>
            </div>
        </section>

        <div v-if="showPatternDialog" class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/45 px-4 backdrop-blur-sm">
            <div class="surface-card !w-full !max-w-md !rounded-[1.75rem] !p-6">
                <h2 class="text-2xl font-semibold text-slate-950">Create Pattern</h2>
                <p class="mt-2 text-sm text-slate-500">Save the selected region as a reusable harmonic or melodic pattern.</p>

                <div class="mt-6 space-y-4">
                    <div>
                        <label class="field-label">Pattern Name</label>
                        <input
                            v-model="patternName"
                            type="text"
                            class="field-control"
                            placeholder="Example: Dubstep wobble bass"
                        />
                    </div>

                    <div>
                        <label class="field-label">Analysis Type</label>
                        <select v-model="patternType" class="field-select">
                            <option value="chroma">Chroma - Harmonic (chords, tonality)</option>
                            <option value="pitch_contour">Pitch - Melodic (vocal, lead)</option>
                        </select>
                    </div>

                    <div class="rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4 text-sm text-slate-600">
                        <p class="font-medium text-slate-950">Selected Region</p>
                        <p class="mt-1">{{ formatTime(patternSelection?.start_time || 0) }} - {{ formatTime(patternSelection?.end_time || 0) }}</p>
                    </div>
                </div>

                <div class="mt-6 flex justify-end gap-3">
                    <button
                        @click="showPatternDialog = false; patternSelection = null"
                        class="action-secondary"
                    >
                        Cancel
                    </button>
                    <button
                        @click="createPattern"
                        :disabled="!patternName.trim()"
                        class="action-primary disabled:opacity-50 disabled:shadow-none"
                    >
                        Create
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>
