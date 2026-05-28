<script setup lang="ts">
/**
 * ResponsivePlayer - Responsive design wrapper for player
 * Requirements: 25.1, 25.2, 25.3, 25.4
 */
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { usePlayerStore } from '@/stores/playerStore'

const playerStore = usePlayerStore()

// State
const windowWidth = ref(window.innerWidth)
const isFullscreen = ref(false)
const isMiniPlayer = ref(false)

// Computed
const isMobile = computed(() => windowWidth.value < 768)
const isTablet = computed(() => windowWidth.value >= 768 && windowWidth.value < 1024)
const isDesktop = computed(() => windowWidth.value >= 1024)

const playerMode = computed(() => {
  if (isFullscreen.value) return 'fullscreen'
  if (isMiniPlayer.value) return 'mini'
  if (isMobile.value) return 'compact'
  return 'default'
})

// Methods
const handleResize = () => {
  windowWidth.value = window.innerWidth
}

const toggleFullscreen = () => {
  isFullscreen.value = !isFullscreen.value
  if (isFullscreen.value) {
    isMiniPlayer.value = false
  }
}

const toggleMiniPlayer = () => {
  isMiniPlayer.value = !isMiniPlayer.value
  if (isMiniPlayer.value) {
    isFullscreen.value = false
  }
}

const exitFullscreen = () => {
  isFullscreen.value = false
}

// Lifecycle
onMounted(() => {
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<template>
  <div 
    class="responsive-player"
    :class="{
      'player-mobile': isMobile,
      'player-tablet': isTablet,
      'player-desktop': isDesktop,
      'player-fullscreen': isFullscreen,
      'player-mini': isMiniPlayer
    }"
  >
    <!-- Fullscreen Mode -->
    <Transition name="fullscreen">
      <div 
        v-if="isFullscreen"
        class="fixed inset-0 bg-gray-900 z-50 flex flex-col"
      >
        <!-- Close button -->
        <button
          @click="exitFullscreen"
          class="absolute top-4 right-4 p-3 bg-gray-800 hover:bg-gray-700 rounded-full transition z-10"
        >
          ✕
        </button>

        <!-- Album art (large) -->
        <div class="flex-1 flex items-center justify-center p-8">
          <div class="w-full max-w-lg aspect-square bg-gray-800 rounded-2xl shadow-2xl flex items-center justify-center">
            <span class="text-9xl">🎵</span>
          </div>
        </div>

        <!-- Track info -->
        <div class="text-center px-8 pb-4">
          <h2 class="text-3xl font-bold text-white">
            {{ playerStore.currentTrack?.title || 'No track' }}
          </h2>
          <p class="text-xl text-gray-400 mt-2">
            {{ playerStore.currentTrack?.artist || 'Unknown Artist' }}
          </p>
        </div>

        <!-- Controls -->
        <div class="px-8 pb-8">
          <slot name="controls" />
        </div>
      </div>
    </Transition>

    <!-- Mini Player (Picture-in-Picture style) -->
    <Transition name="mini">
      <div 
        v-if="isMiniPlayer && !isFullscreen"
        class="fixed bottom-4 right-4 w-72 bg-gray-800 rounded-xl shadow-2xl overflow-hidden z-40"
      >
        <div class="flex items-center p-3">
          <!-- Album art (small) -->
          <div class="w-12 h-12 bg-gray-700 rounded flex-shrink-0 flex items-center justify-center">
            <span class="text-xl">🎵</span>
          </div>
          
          <!-- Track info -->
          <div class="flex-1 min-w-0 ml-3">
            <div class="text-sm font-medium text-white truncate">
              {{ playerStore.currentTrack?.title || 'No track' }}
            </div>
            <div class="text-xs text-gray-400 truncate">
              {{ playerStore.currentTrack?.artist || 'Unknown' }}
            </div>
          </div>
          
          <!-- Controls -->
          <div class="flex items-center space-x-1 ml-2">
            <button
              @click="playerStore.isPlaying ? playerStore.pause() : playerStore.play()"
              class="p-2 hover:bg-gray-700 rounded-full transition"
            >
              {{ playerStore.isPlaying ? '⏸️' : '▶️' }}
            </button>
            <button
              @click="toggleMiniPlayer"
              class="p-2 hover:bg-gray-700 rounded-full transition"
              title="Expand"
            >
              ↗️
            </button>
          </div>
        </div>
        
        <!-- Progress bar -->
        <div class="h-1 bg-gray-700">
          <div 
            class="h-full bg-green-500 transition-all"
            :style="{ width: `${playerStore.progress}%` }"
          />
        </div>
      </div>
    </Transition>

    <!-- Default/Compact Player -->
    <div v-if="!isFullscreen && !isMiniPlayer">
      <slot 
        :is-mobile="isMobile"
        :is-tablet="isTablet"
        :is-desktop="isDesktop"
        :toggle-fullscreen="toggleFullscreen"
        :toggle-mini-player="toggleMiniPlayer"
      />
    </div>
  </div>
</template>

<style scoped>
.fullscreen-enter-active,
.fullscreen-leave-active {
  transition: all 0.3s ease;
}

.fullscreen-enter-from,
.fullscreen-leave-to {
  opacity: 0;
  transform: scale(0.95);
}

.mini-enter-active,
.mini-leave-active {
  transition: all 0.3s ease;
}

.mini-enter-from,
.mini-leave-to {
  opacity: 0;
  transform: translateY(100px);
}

/* Mobile-specific styles */
.player-mobile {
  /* Compact layout for mobile */
}

/* Tablet-specific styles */
.player-tablet {
  /* Medium layout for tablet */
}

/* Desktop-specific styles */
.player-desktop {
  /* Full layout for desktop */
}
</style>
