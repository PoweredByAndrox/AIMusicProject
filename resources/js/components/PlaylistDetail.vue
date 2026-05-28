<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { api } from '@/api/client'
import { usePlayerStore } from '@/stores/playerStore'
import type { Track } from '@/types'

const route = useRoute()
const router = useRouter()
const playerStore = usePlayerStore()

interface Playlist {
    id: number
    name: string
    description: string | null
    is_ai_generated: boolean
    track_count: number
    total_duration: number
}

const playlistId = computed(() => Number(route.params.id))

const playlist = ref<Playlist | null>(null)
const tracks = ref<Track[]>([])
const isLoading = ref(false)
const error = ref<string | null>(null)
const draggedIndex = ref<number | null>(null)
const dragOverIndex = ref<number | null>(null)

const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

const formatTotalDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    if (hours > 0) return `${hours}h ${mins}m`
    return `${mins} min`
}

const goBack = () => {
    router.push('/playlists')
}

const fetchPlaylist = async () => {
    if (!playlistId.value) return

    isLoading.value = true
    error.value = null
    try {
        const response = await api.getPlaylist(playlistId.value)
        playlist.value = response
        tracks.value = response.tracks || []
    } catch (errorValue) {
        error.value = 'Failed to load playlist'
        console.error('Error fetching playlist:', errorValue)
    } finally {
        isLoading.value = false
    }
}

const reorderTracks = async (fromIndex: number, toIndex: number) => {
    if (!playlistId.value) return

    const [movedTrack] = tracks.value.splice(fromIndex, 1)
    tracks.value.splice(toIndex, 0, movedTrack)

    const trackOrder = tracks.value.map((track, index) => ({ track_id: track.id, position: index }))

    try {
        await api.reorderPlaylistTracks(playlistId.value, trackOrder)
    } catch (errorValue) {
        fetchPlaylist()
        console.error('Error reordering tracks:', errorValue)
    }
}

const removeTrack = async (trackId: number) => {
    if (!playlistId.value) return

    try {
        await api.removeTrackFromPlaylist(playlistId.value, trackId)
        tracks.value = tracks.value.filter((track) => track.id !== trackId)
        if (playlist.value) {
            playlist.value.track_count--
        }
    } catch (errorValue) {
        error.value = 'Failed to remove track'
        console.error('Error removing track:', errorValue)
    }
}

const onDragStart = (index: number) => {
    draggedIndex.value = index
}

const onDragOver = (event: DragEvent, index: number) => {
    event.preventDefault()
    dragOverIndex.value = index
}

const onDragLeave = () => {
    dragOverIndex.value = null
}

const onDrop = (toIndex: number) => {
    if (draggedIndex.value !== null && draggedIndex.value !== toIndex) {
        reorderTracks(draggedIndex.value, toIndex)
    }
    draggedIndex.value = null
    dragOverIndex.value = null
}

const onDragEnd = () => {
    draggedIndex.value = null
    dragOverIndex.value = null
}

const playPlaylist = (startIndex = 0) => {
    if (tracks.value.length > 0) {
        playerStore.playPlaylist(tracks.value, startIndex)
    }
}

const playTrack = (_track: Track, index: number) => {
    playPlaylist(index)
}

onMounted(() => {
    fetchPlaylist()
})

watch(playlistId, () => {
    fetchPlaylist()
})
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <button type="button" class="badge-soft !normal-case !tracking-normal" @click="goBack">
                        Back to playlists
                    </button>
                    <template v-if="playlist">
                        <h1 class="page-title">{{ playlist.name }}</h1>
                        <p v-if="playlist.description" class="page-copy">{{ playlist.description }}</p>
                        <div class="mt-4 flex flex-wrap gap-2">
                            <span class="badge-soft">{{ playlist.track_count }} tracks</span>
                            <span class="badge-soft">{{ formatTotalDuration(playlist.total_duration) }}</span>
                            <span v-if="playlist.is_ai_generated" class="badge-soft !border-purple-200 !bg-purple-50 !text-purple-700">AI generated</span>
                        </div>
                    </template>
                </div>

                <button v-if="tracks.length > 0" type="button" class="action-primary" @click="playPlaylist()">
                    Play playlist
                </button>
            </div>
        </section>

        <div v-if="error" class="rounded-[1.4rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-700">
            {{ error }}
        </div>

        <section v-if="isLoading" class="surface-card text-center">
            <p class="text-lg font-semibold text-slate-900">Loading playlist...</p>
        </section>

        <section v-else class="surface-card !rounded-[1.75rem] !p-5">
            <div class="flex items-center justify-between gap-4">
                <div>
                    <p class="section-kicker">Track order</p>
                    <h2 class="mt-2 text-2xl font-semibold text-slate-950">Playlist tracks</h2>
                </div>
                <span class="badge-soft">Drag to reorder</span>
            </div>

            <div v-if="tracks.length === 0" class="empty-state mt-5">
                <p class="text-lg font-semibold text-slate-900">No tracks in this playlist</p>
                <p class="mt-2 text-sm text-slate-500">Add tracks from the library to start playback from this collection.</p>
            </div>

            <div v-else class="mt-5 space-y-3">
                <div
                    v-for="(track, index) in tracks"
                    :key="track.id"
                    :draggable="true"
                    class="rounded-[1.3rem] border border-slate-200 bg-slate-50/90 p-4 transition"
                    :class="{
                        '!border-blue-400 bg-blue-50': playerStore.currentTrack?.id === track.id,
                        '!border-emerald-400 shadow-[0_0_0_3px_rgba(16,185,129,0.12)]': dragOverIndex === index,
                    }"
                    @dragstart="onDragStart(index)"
                    @dragover="(event) => onDragOver(event, index)"
                    @dragleave="onDragLeave"
                    @drop="onDrop(index)"
                    @dragend="onDragEnd"
                    @dblclick="playTrack(track, index)"
                >
                    <div class="flex items-center gap-4">
                        <div class="w-10 text-center text-sm font-semibold text-slate-400">
                            {{ index + 1 }}
                        </div>

                        <button type="button" class="action-secondary !rounded-xl !px-3 !py-2" @click="playTrack(track, index)">
                            Play
                        </button>

                        <div class="min-w-0 flex-1">
                            <p class="truncate text-base font-semibold text-slate-950">{{ track.title }}</p>
                            <p class="mt-1 truncate text-sm text-slate-500">{{ track.artist || 'Unknown artist' }}</p>
                        </div>

                        <div class="text-sm text-slate-500">
                            {{ formatDuration(track.duration) }}
                        </div>

                        <button type="button" class="action-danger !rounded-xl !px-3 !py-2 !text-xs" @click="removeTrack(track.id)">
                            Remove
                        </button>
                    </div>
                </div>
            </div>
        </section>
    </div>
</template>
