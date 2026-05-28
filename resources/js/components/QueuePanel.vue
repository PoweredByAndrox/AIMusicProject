<script setup lang="ts">
/**
 * QueuePanel - Up Next queue panel with recommendations
 * Real-time queue updates via reactive store
 */
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { usePlayerStore } from '@/stores/playerStore'
import { api } from '@/api/client'
import type { Track } from '@/types'

const playerStore = usePlayerStore()

const isOpen = ref(false)
const recommendations = ref<Track[]>([])
const isLoadingRecs = ref(false)
const activeTab = ref<'queue' | 'recommended'>('queue')

const upNextTracks = computed(() => {
  const idx = playerStore.currentIndex
  if (idx < 0) return [...playerStore.queue]
  return playerStore.queue.slice(idx + 1)
})

const currentTrack = computed(() => playerStore.currentTrack)
let refreshInterval: number | null = null

onMounted(() => {
  if (playerStore.smartRadioEnabled || playerStore.aiDjEnabled) {
    startAutoRefresh()
  }
})

onUnmounted(() => {
  stopAutoRefresh()
})

watch([() => playerStore.smartRadioEnabled, () => playerStore.aiDjEnabled], ([radio, dj]) => {
  if (radio || dj) {
    startAutoRefresh()
  } else {
    stopAutoRefresh()
  }
})

function startAutoRefresh() {
  if (refreshInterval) return
  refreshInterval = window.setInterval(() => {
    if (currentTrack.value?.id) {
      loadRecommendations(currentTrack.value.id)
    }
  }, 15000)
}

function stopAutoRefresh() {
  if (refreshInterval) {
    clearInterval(refreshInterval)
    refreshInterval = null
  }
}

watch(currentTrack, async (track) => {
  if (track?.id && track.analysis_status === 'completed') {
    await loadRecommendations(track.id)
  }
}, { immediate: true })

async function loadRecommendations(trackId: number) {
  isLoadingRecs.value = true
  try {
    const response = await api.getCompatibleTracks(trackId)
    recommendations.value = response.map((item) => item.track).filter(Boolean)
  } catch (e) {
    console.error('Error loading recommendations:', e)
    recommendations.value = []
  } finally {
    isLoadingRecs.value = false
  }
}

function togglePanel() {
  isOpen.value = !isOpen.value
}

function playTrack(track: Track) {
  playerStore.playTrack(track)
}

function addToQueue(track: Track) {
  playerStore.addToQueue(track, 'next', 'ai_dj')
}

function addAllRecommended() {
  recommendations.value.slice(0, 10).forEach((track) => {
    playerStore.addToQueue(track, 'end', 'ai_dj')
  })
}

function removeFromQueue(itemId: string) {
  playerStore.removeFromQueue(itemId)
}

function clearQueue() {
  playerStore.clearQueue()
}

function formatDuration(seconds: number): string {
  if (!seconds) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}
</script>

<template>
  <div class="queue-panel-container relative">
    <button
      @click="togglePanel"
      class="queue-toggle"
      :class="{ 'queue-toggle--active': isOpen }"
      :title="isOpen ? 'Hide queue' : 'Show queue'"
    >
      <span class="queue-toggle__label">Queue</span>
      <span class="queue-toggle__count">{{ upNextTracks.length }}</span>
    </button>

    <Teleport to="body">
      <Transition name="slide-up">
        <div
          v-if="isOpen"
          class="queue-panel-overlay fixed bottom-[10.5rem] left-2 right-2 z-[70] overflow-hidden rounded-[1.8rem] text-slate-100 sm:bottom-32 sm:left-auto sm:right-4 sm:w-[min(28rem,calc(100vw-1.5rem))]"
        >
          <div class="queue-panel-header px-5 py-4">
            <div class="flex items-center justify-between">
              <div>
                <div class="text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-slate-400">Playback queue</div>
                <h3 class="mt-2 text-xl font-semibold tracking-tight text-white">Up next</h3>
              </div>
              <button @click="togglePanel" class="panel-close">
                Close
              </button>
            </div>

            <div class="mt-4 flex gap-2">
              <button
                @click="activeTab = 'queue'"
                class="tab-chip"
                :class="{ 'tab-chip--active': activeTab === 'queue' }"
              >
                Queue {{ upNextTracks.length }}
              </button>
              <button
                @click="activeTab = 'recommended'"
                class="tab-chip"
                :class="{ 'tab-chip--active': activeTab === 'recommended' }"
              >
                Recommended {{ recommendations.length }}
              </button>
            </div>
          </div>

          <div class="max-h-[58vh] overflow-y-auto px-4 py-4">
            <div v-if="activeTab === 'queue'">
              <div v-if="currentTrack" class="panel-card mb-3">
                <div class="text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-sky-200/75">Now playing</div>
                <div class="mt-3 flex items-center gap-3">
                  <div class="flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-sky-500 via-blue-600 to-indigo-700 text-sm font-bold text-white">
                    {{ currentTrack.title.slice(0, 2).toUpperCase() }}
                  </div>
                  <div class="min-w-0 flex-1">
                    <div class="truncate text-sm font-semibold text-white">{{ currentTrack.title }}</div>
                    <div class="mt-1 truncate text-xs text-slate-300/75">{{ currentTrack.artist || 'Unknown artist' }}</div>
                  </div>
                </div>
              </div>

              <div v-if="upNextTracks.length > 0" class="space-y-2">
                <div
                  v-for="(item, index) in upNextTracks"
                  :key="item.id"
                  class="queue-row group"
                >
                  <div class="queue-row__index">{{ index + 1 }}</div>
                  <div class="min-w-0 flex-1">
                    <div class="truncate text-sm font-semibold text-white">{{ item.track.title }}</div>
                    <div class="mt-1 truncate text-xs text-slate-300/70">{{ item.track.artist || 'Unknown artist' }}</div>
                  </div>
                  <div class="text-xs font-semibold uppercase tracking-[0.12em] text-slate-400">
                    {{ formatDuration(item.track.duration) }}
                  </div>
                  <button
                    @click="removeFromQueue(item.id)"
                    class="queue-row__action"
                  >
                    Remove
                  </button>
                </div>
              </div>

              <div v-else class="empty-panel-state">
                <div class="text-lg font-semibold text-white">Queue is empty</div>
                <div class="mt-2 text-sm text-slate-300/75">Add tracks from the recommended tab or start Smart Radio.</div>
              </div>

              <button
                v-if="upNextTracks.length > 0"
                @click="clearQueue"
                class="panel-secondary mt-3 w-full"
              >
                Clear queue
              </button>
            </div>

            <div v-if="activeTab === 'recommended'">
              <div v-if="isLoadingRecs" class="empty-panel-state">
                <div class="text-lg font-semibold text-white">Loading recommendations</div>
                <div class="mt-2 text-sm text-slate-300/75">Checking the active track for compatible options.</div>
              </div>

              <div v-else-if="recommendations.length > 0">
                <button
                  @click="addAllRecommended"
                  class="panel-primary mb-3 w-full"
                >
                  Add top 10 to queue
                </button>

                <div class="space-y-2">
                  <div
                    v-for="track in recommendations"
                    :key="track.id"
                    class="queue-row cursor-pointer"
                    @click="playTrack(track)"
                  >
                    <div class="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/5 text-xs font-bold uppercase tracking-[0.16em] text-slate-200">
                      {{ track.title.slice(0, 2).toUpperCase() }}
                    </div>
                    <div class="min-w-0 flex-1">
                      <div class="truncate text-sm font-semibold text-white">{{ track.title }}</div>
                      <div class="mt-1 truncate text-xs text-slate-300/70">
                        {{ track.artist || 'Unknown artist' }}
                        <span v-if="track.bpm"> | {{ Math.round(track.bpm) }} BPM</span>
                      </div>
                    </div>
                    <button
                      @click.stop="addToQueue(track)"
                      class="queue-row__action"
                    >
                      Add
                    </button>
                  </div>
                </div>
              </div>

              <div v-else class="empty-panel-state">
                <div class="text-lg font-semibold text-white">No recommendations yet</div>
                <div class="mt-2 text-sm text-slate-300/75">Play a completed track to load recommended follow-ups.</div>
              </div>
            </div>
          </div>
        </div>
      </Transition>
    </Teleport>
  </div>
</template>

<style scoped>
.queue-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.7rem;
  min-height: 2.85rem;
  border-radius: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.05);
  padding: 0.72rem 0.95rem;
  color: #e2e8f0;
  transition:
    transform 0.18s ease,
    border-color 0.18s ease,
    background-color 0.18s ease;
}

.queue-toggle:hover {
  transform: translateY(-1px);
  border-color: rgba(125, 211, 252, 0.3);
  background: rgba(255, 255, 255, 0.08);
}

.queue-toggle--active {
  border-color: rgba(56, 189, 248, 0.28);
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(37, 99, 235, 0.18));
}

.queue-panel-overlay {
  border: 1px solid rgba(15, 23, 42, 0.9);
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.14), transparent 24%),
    linear-gradient(180deg, rgba(2, 6, 23, 0.985) 0%, rgba(15, 23, 42, 0.975) 100%);
  box-shadow:
    0 28px 90px rgba(2, 6, 23, 0.62),
    0 0 0 1px rgba(148, 163, 184, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.06);
  backdrop-filter: blur(8px);
}

.queue-panel-header {
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
  background: linear-gradient(180deg, rgba(15, 23, 42, 0.42) 0%, rgba(15, 23, 42, 0.12) 100%);
}

.queue-toggle__label {
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}

.queue-toggle__count {
  display: inline-flex;
  min-width: 1.9rem;
  justify-content: center;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.12);
  padding: 0.22rem 0.55rem;
  font-size: 0.8rem;
  font-weight: 800;
}

.panel-close,
.tab-chip,
.panel-secondary,
.panel-primary,
.queue-row__action {
  transition:
    transform 0.18s ease,
    border-color 0.18s ease,
    background-color 0.18s ease,
    color 0.18s ease;
}

.panel-close {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(15, 23, 42, 0.88);
  padding: 0.65rem 0.9rem;
  color: #e2e8f0;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.tab-chip {
  flex: 1;
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.2);
  background: rgba(15, 23, 42, 0.88);
  padding: 0.7rem 0.9rem;
  color: #e2e8f0;
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.tab-chip--active {
  border-color: rgba(56, 189, 248, 0.52);
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.38), rgba(37, 99, 235, 0.34));
  color: #f8fafc;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08);
}

.panel-card,
.queue-row,
.empty-panel-state {
  border-radius: 1.3rem;
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: rgba(15, 23, 42, 0.86);
}

.panel-card {
  padding: 1rem;
}

.queue-row {
  display: flex;
  align-items: center;
  gap: 0.85rem;
  padding: 0.9rem 0.95rem;
}

.queue-row:hover {
  background: rgba(30, 41, 59, 0.96);
}

.queue-row__index {
  width: 1.5rem;
  flex-shrink: 0;
  color: #94a3b8;
  font-size: 0.8rem;
  font-weight: 800;
  text-align: center;
}

.queue-row__action {
  border-radius: 999px;
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(30, 41, 59, 0.94);
  padding: 0.55rem 0.75rem;
  color: #e2e8f0;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.queue-row__action:hover {
  transform: translateY(-1px);
  background: rgba(51, 65, 85, 0.98);
}

.empty-panel-state {
  padding: 2.25rem 1.25rem;
  text-align: center;
}

.panel-primary,
.panel-secondary {
  border-radius: 1rem;
  padding: 0.9rem 1rem;
  font-size: 0.78rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.panel-primary {
  border: 1px solid rgba(96, 165, 250, 0.28);
  background: linear-gradient(135deg, rgba(37, 99, 235, 0.92), rgba(14, 165, 233, 0.92));
  color: #ffffff;
}

.panel-secondary {
  border: 1px solid rgba(148, 163, 184, 0.22);
  background: rgba(15, 23, 42, 0.9);
  color: #e2e8f0;
}

.panel-primary:hover,
.panel-secondary:hover {
  transform: translateY(-1px);
}

.slide-up-enter-active,
.slide-up-leave-active {
  transition: all 0.24s ease;
}

.slide-up-enter-from,
.slide-up-leave-to {
  opacity: 0;
  transform: translateY(14px);
}

@media (max-width: 639px) {
  .queue-toggle {
    min-height: 2.55rem;
    gap: 0.55rem;
    padding: 0.62rem 0.78rem;
  }

  .queue-toggle__label {
    font-size: 0.66rem;
    letter-spacing: 0.14em;
  }

  .queue-toggle__count {
    min-width: 1.7rem;
    padding-inline: 0.45rem;
    font-size: 0.72rem;
  }

  .queue-panel-overlay {
    border-radius: 1.35rem;
  }

  .tab-chip {
    padding: 0.64rem 0.72rem;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
  }

  .queue-row {
    gap: 0.65rem;
    padding: 0.78rem 0.8rem;
  }

  .queue-row__action {
    padding: 0.48rem 0.62rem;
    font-size: 0.62rem;
    letter-spacing: 0.1em;
  }
}
</style>
