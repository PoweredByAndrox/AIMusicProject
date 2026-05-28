<script setup lang="ts">
/**
 * AIDJToggle - AI DJ Live Mode controls
 * Requirements: 32.1, 32.2, 32.6
 */
import { ref, computed, watch } from 'vue'
import { usePlayerStore } from '@/stores/playerStore'
import { api } from '@/api/client'

const playerStore = usePlayerStore()

const isLoading = ref(false)
const nextTrackInfo = ref<{ title: string; artist: string; compatibility: number } | null>(null)

const isEnabled = computed(() => playerStore.aiDjEnabled)
const currentTrack = computed(() => playerStore.currentTrack)

const toggleAIDJ = async () => {
  isLoading.value = true
  try {
    playerStore.setAiDj(!isEnabled.value)
    if (isEnabled.value && currentTrack.value) {
      await fetchNextCompatibleTrack()
    }
  } catch (e) {
    console.error('Error toggling AI DJ:', e)
  } finally {
    isLoading.value = false
  }
}

const fetchNextCompatibleTrack = async () => {
  if (!currentTrack.value) return

  try {
    const tracks = await api.getCompatibleTracks(currentTrack.value.id)
    if (tracks && tracks.length > 0) {
      const next = tracks[0]
      nextTrackInfo.value = {
        title: next.track?.title || 'Unknown',
        artist: next.track?.artist || 'Unknown Artist',
        compatibility: next.total_score || 0,
      }
    } else {
      nextTrackInfo.value = null
    }
  } catch (e) {
    console.error('Error fetching compatible tracks:', e)
  }
}

watch(currentTrack, () => {
  if (isEnabled.value && currentTrack.value) {
    fetchNextCompatibleTrack()
  }
})
</script>

<template>
  <div class="ai-dj-controls">
    <button
      @click="toggleAIDJ"
      :disabled="isLoading"
      class="feature-toggle"
      :class="{ 'feature-toggle--active': isEnabled }"
      title="AI DJ Mode"
    >
      <span class="feature-toggle__kicker">AI DJ</span>
      <span class="feature-toggle__label">
        {{ isLoading ? 'Working' : (isEnabled ? 'Enabled' : 'Assist') }}
      </span>
    </button>

    <Transition name="slide">
      <div
        v-if="isEnabled && nextTrackInfo"
        class="mt-2 rounded-[1.15rem] border border-fuchsia-400/20 bg-white/5 px-4 py-3 text-slate-100 shadow-[0_16px_32px_rgba(15,23,42,0.18)]"
      >
        <div class="text-[0.68rem] font-semibold uppercase tracking-[0.22em] text-fuchsia-200/80">
          Up next
        </div>
        <div class="mt-2 truncate text-sm font-semibold text-white">{{ nextTrackInfo.title }}</div>
        <div class="mt-1 truncate text-xs text-slate-300/80">{{ nextTrackInfo.artist }}</div>
        <div class="mt-3 flex items-center gap-3">
          <div class="h-2 flex-1 overflow-hidden rounded-full bg-white/10">
            <div
              class="h-full rounded-full bg-gradient-to-r from-fuchsia-400 via-violet-400 to-sky-400 transition-all"
              :style="{ width: `${nextTrackInfo.compatibility}%` }"
            />
          </div>
          <span class="text-xs font-semibold uppercase tracking-[0.16em] text-fuchsia-100">
            {{ Math.round(nextTrackInfo.compatibility) }}%
          </span>
        </div>
      </div>
    </Transition>
  </div>
</template>

<style scoped>
.feature-toggle {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  min-height: 2.85rem;
  border-radius: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.05);
  padding: 0.75rem 0.95rem;
  color: #e2e8f0;
  transition:
    transform 0.18s ease,
    border-color 0.18s ease,
    background-color 0.18s ease;
}

.feature-toggle:hover:not(:disabled) {
  transform: translateY(-1px);
  border-color: rgba(217, 70, 239, 0.3);
  background: rgba(255, 255, 255, 0.08);
}

.feature-toggle:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.feature-toggle--active {
  border-color: rgba(217, 70, 239, 0.28);
  background: linear-gradient(135deg, rgba(168, 85, 247, 0.18), rgba(217, 70, 239, 0.18));
}

.feature-toggle__kicker {
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.2em;
  text-transform: uppercase;
}

.feature-toggle__label {
  font-size: 0.82rem;
  font-weight: 700;
}

.slide-enter-active,
.slide-leave-active {
  transition: all 0.22s ease;
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
