<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { api } from '@/api/client'
import { usePlayerStore } from '@/stores/playerStore'

const router = useRouter()
const playerStore = usePlayerStore()

interface Playlist {
    id: number
    name: string
    description: string | null
    is_ai_generated: boolean
    track_count: number
    total_duration: number
    created_at: string
}

const playlists = ref<Playlist[]>([])
const isLoading = ref(false)
const showCreateDialog = ref(false)
const newPlaylistName = ref('')
const newPlaylistDescription = ref('')
const isCreating = ref(false)
const error = ref<string | null>(null)

const playlistStats = computed(() => ([
    {
        label: 'Playlists',
        value: playlists.value.length.toString(),
        detail: 'saved collections in this account',
    },
    {
        label: 'AI generated',
        value: playlists.value.filter((playlist) => playlist.is_ai_generated).length.toString(),
        detail: 'built by the recommendation stack',
    },
    {
        label: 'Tracks staged',
        value: playlists.value.reduce((sum, playlist) => sum + playlist.track_count, 0).toString(),
        detail: 'tracks across visible playlists',
    },
]))

const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    if (hours > 0) {
        return `${hours}h ${mins}m`
    }
    return `${mins} min`
}

const selectPlaylist = (playlist: Playlist) => {
    router.push(`/playlists/${playlist.id}`)
}

const playPlaylist = async (playlist: Playlist) => {
    try {
        const response = await api.getPlaylist(playlist.id)
        if (response.tracks && response.tracks.length > 0) {
            playerStore.playPlaylist(response.tracks, 0)
        }
    } catch (error) {
        console.error('Error playing playlist:', error)
    }
}

const fetchPlaylists = async () => {
    isLoading.value = true
    error.value = null
    try {
        const response = await api.getPlaylists()
        playlists.value = response.data || []
    } catch (errorValue) {
        error.value = 'Failed to load playlists'
        console.error('Error fetching playlists:', errorValue)
    } finally {
        isLoading.value = false
    }
}

const createPlaylist = async () => {
    if (!newPlaylistName.value.trim()) return

    isCreating.value = true
    try {
        const response = await api.createPlaylist(
            newPlaylistName.value.trim(),
            newPlaylistDescription.value.trim() || null,
        )
        playlists.value.unshift(response as never)
        showCreateDialog.value = false
        newPlaylistName.value = ''
        newPlaylistDescription.value = ''
    } catch (errorValue) {
        error.value = 'Failed to create playlist'
        console.error('Error creating playlist:', errorValue)
    } finally {
        isCreating.value = false
    }
}

const deletePlaylist = async (playlist: Playlist) => {
    if (!confirm(`Delete "${playlist.name}"?`)) return

    try {
        await api.deletePlaylist(playlist.id)
        playlists.value = playlists.value.filter((entry) => entry.id !== playlist.id)
    } catch (errorValue) {
        error.value = 'Failed to delete playlist'
        console.error('Error deleting playlist:', errorValue)
    }
}

onMounted(() => {
    fetchPlaylists()
})
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p class="section-kicker">Playlists</p>
                    <h1 class="page-title">Playlist workspace</h1>
                    <p class="page-copy">Organize manual playlists, review AI-generated collections, and jump straight into playback.</p>
                </div>
                <button type="button" class="action-primary" @click="showCreateDialog = true">
                    New playlist
                </button>
            </div>
        </section>

        <section class="grid gap-4 md:grid-cols-3">
            <article v-for="stat in playlistStats" :key="stat.label" class="metric-card">
                <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{{ stat.label }}</p>
                <p class="mt-3 text-3xl font-semibold text-slate-950">{{ stat.value }}</p>
                <p class="mt-2 text-sm text-slate-500">{{ stat.detail }}</p>
            </article>
        </section>

        <div v-if="error" class="rounded-[1.4rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-700">
            {{ error }}
        </div>

        <section v-if="isLoading" class="surface-card text-center">
            <p class="text-lg font-semibold text-slate-900">Loading playlists...</p>
        </section>

        <section v-else-if="playlists.length === 0" class="surface-card text-center">
            <p class="text-lg font-semibold text-slate-900">No playlists yet</p>
            <p class="mt-2 text-sm text-slate-500">Create the first collection to start grouping tracks by mood, use case, or AI suggestions.</p>
        </section>

        <section v-else class="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            <article
                v-for="playlist in playlists"
                :key="playlist.id"
                class="surface-card !rounded-[1.75rem] !p-5 transition hover:-translate-y-1"
            >
                <div class="flex items-start justify-between gap-3">
                    <div class="min-w-0">
                        <button type="button" class="text-left text-xl font-semibold text-slate-950 hover:text-blue-700" @click="selectPlaylist(playlist)">
                            {{ playlist.name }}
                        </button>
                        <p v-if="playlist.description" class="mt-2 line-clamp-2 text-sm leading-6 text-slate-500">
                            {{ playlist.description }}
                        </p>
                    </div>
                    <span v-if="playlist.is_ai_generated" class="badge-soft !border-purple-200 !bg-purple-50 !text-purple-700">
                        AI
                    </span>
                </div>

                <div class="mt-5 rounded-[1.5rem] bg-gradient-to-br from-slate-950 via-sky-900 to-blue-700 p-6 text-white">
                    <div class="text-xs font-semibold uppercase tracking-[0.2em] text-sky-100/80">Collection</div>
                    <div class="mt-4 text-4xl font-semibold">{{ playlist.track_count }}</div>
                    <div class="mt-2 text-sm text-sky-100/80">tracks • {{ formatDuration(playlist.total_duration) }}</div>
                </div>

                <div class="mt-5 flex gap-2">
                    <button type="button" class="action-primary flex-1" @click="playPlaylist(playlist)">
                        Play
                    </button>
                    <button type="button" class="action-secondary flex-1" @click="selectPlaylist(playlist)">
                        Open
                    </button>
                    <button type="button" class="action-danger" @click="deletePlaylist(playlist)">
                        Delete
                    </button>
                </div>
            </article>
        </section>

        <Teleport to="body">
            <Transition name="fade">
                <div v-if="showCreateDialog" class="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/60 px-4" @click.self="showCreateDialog = false">
                    <div class="surface-card w-full max-w-xl !rounded-[1.8rem] !p-6">
                        <h2 class="text-2xl font-semibold text-slate-950">Create playlist</h2>

                        <div class="mt-6 space-y-4">
                            <div>
                                <label class="field-label" for="playlist-name">Name</label>
                                <input id="playlist-name" v-model="newPlaylistName" type="text" class="field-control" placeholder="Late-night references" @keyup.enter="createPlaylist">
                            </div>

                            <div>
                                <label class="field-label" for="playlist-description">Description</label>
                                <textarea id="playlist-description" v-model="newPlaylistDescription" rows="3" class="field-textarea" placeholder="Add a short note about the playlist."></textarea>
                            </div>
                        </div>

                        <div class="mt-6 flex justify-end gap-3">
                            <button type="button" class="action-secondary" @click="showCreateDialog = false">
                                Cancel
                            </button>
                            <button type="button" class="action-primary" :disabled="!newPlaylistName.trim() || isCreating" @click="createPlaylist">
                                {{ isCreating ? 'Creating...' : 'Create' }}
                            </button>
                        </div>
                    </div>
                </div>
            </Transition>
        </Teleport>
    </div>
</template>

<style scoped>
.fade-enter-active,
.fade-leave-active {
    transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
    opacity: 0;
}
</style>
