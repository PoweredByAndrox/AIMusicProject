<script setup lang="ts">
/**
 * PlayerComponent - Enhanced player with Queue Panel
 * Requirements: 1.6, 1.7, 2.1, 2.3, 2.4, 2.5, 2.6, 5.7
 */
import { ref, computed, watch, onMounted } from 'vue'
import { usePlayerStore, toastMessage } from '@/stores/playerStore'
import { getAudioEngine } from '@/audio/AudioEngine'

const playerStore = usePlayerStore()
const audioEngine = getAudioEngine()

// UI State
const showQueuePanel = ref(false)
const isSeeking = ref(false)
const seekValue = ref(0)
const draggedIndex = ref<number | null>(null)
const dragOverIndex = ref<number | null>(null)

// Audio element ref (fallback for browsers without Web Audio API)
const audioRef = ref<HTMLAudioElement | null>(null)

// Computed
const audioUrl = computed(() => {
  if (!playerStore.currentTrack) return ''
  const origin = window.location.origin
  const backendUrl = origin.includes(':5173') ? origin.replace(':5173', ':8000') : origin
  return `${backendUrl}/api/tracks/${playerStore.currentTrack.id}/stream`
})

const formattedCurrentTime = computed(() => formatTime(playerStore.currentTime))
const formattedDuration = computed(() => formatTime(playerStore.duration))

const effectiveVolume = computed(() => {
  return playerStore.isMuted ? 0 : playerStore.volume
})

const repeatIcon = computed(() => {
  switch (playerStore.repeatMode) {
    case 'one': return '🔂'
    case 'all': return '🔁'
    default: return '➡️'
  }
})

const repeatLabel = computed(() => {
  switch (playerStore.repeatMode) {
    case 'one': return 'Repeat One'
    case 'all': return 'Repeat All'
    default: return 'Repeat Off'
  }
})

// Format time helper
function formatTime(seconds: number): string {
  if (!seconds || isNaN(seconds)) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// Audio event handlers
function onLoadedMetadata() {
  if (audioRef.value) {
    playerStore.updateDuration(audioRef.value.duration)
  }
}

function onTimeUpdate() {
  if (audioRef.value && !isSeeking.value) {
    playerStore.updateTime(audioRef.value.currentTime)
  }
}

function onEnded() {
  playerStore.onTrackEnded()
}

// Seek handlers
function onSeekStart() {
  isSeeking.value = true
  seekValue.value = playerStore.currentTime
}

function onSeekInput(event: Event) {
  const target = event.target as HTMLInputElement
  seekValue.value = parseFloat(target.value)
}

function onSeekEnd() {
  if (audioRef.value) {
    audioRef.value.currentTime = seekValue.value
    playerStore.seek(seekValue.value)
  }
  isSeeking.value = false
}

// Volume handler
function onVolumeChange(event: Event) {
  const target = event.target as HTMLInputElement
  playerStore.setVolume(parseFloat(target.value))
}

// Queue drag and drop handlers
function onDragStart(index: number) {
  draggedIndex.value = index
}

function onDragOver(event: DragEvent, index: number) {
  event.preventDefault()
  dragOverIndex.value = index
}

function onDragLeave() {
  dragOverIndex.value = null
}

function onDrop(toIndex: number) {
  if (draggedIndex.value !== null && draggedIndex.value !== toIndex) {
    playerStore.reorderQueue(draggedIndex.value, toIndex)
  }
  draggedIndex.value = null
  dragOverIndex.value = null
}

function onDragEnd() {
  draggedIndex.value = null
  dragOverIndex.value = null
}

// Play specific queue item
function playQueueItem(index: number) {
  const item = playerStore.queue[index]
  if (item) {
    playerStore.playTrack(item.track)
  }
}

// Remove from queue
function removeFromQueue(itemId: string) {
  playerStore.removeFromQueue(itemId)
}

// Toggle queue panel
function toggleQueuePanel() {
  showQueuePanel.value = !showQueuePanel.value
}

// Helper to handle play errors
function handlePlayError(e: Error) {
  if (e.name === 'AbortError') return
  console.error('Play error:', e)
}

// Watch for play/pause changes
watch(() => playerStore.isPlaying, (playing) => {
  if (audioRef.value && playerStore.currentTrack) {
    if (playing) {
      if (!audioRef.value.src || !audioRef.value.src.includes(String(playerStore.currentTrack.id))) {
        audioRef.value.src = audioUrl.value
        audioRef.value.load()
      }
      audioRef.value.play().catch(handlePlayError)
    } else {
      audioRef.value.pause()
    }
  }
})

// Watch for track changes
watch(() => playerStore.currentTrack, () => {
  if (audioRef.value && playerStore.currentTrack) {
    audioRef.value.src = audioUrl.value
    audioRef.value.load()
    if (playerStore.isPlaying) {
      audioRef.value.play().catch(handlePlayError)
    }
  }
})

// Watch for volume changes
watch(effectiveVolume, (vol) => {
  if (audioRef.value) {
    audioRef.value.volume = vol
  }
})

// Watch for seek from store
watch(() => playerStore.currentTime, (time) => {
  if (audioRef.value && !isSeeking.value && Math.abs(audioRef.value.currentTime - time) > 1) {
    audioRef.value.currentTime = time
  }
})

// Sync crossfade settings with audio engine
watch(() => playerStore.crossfadeEnabled, (enabled) => {
  audioEngine.setCrossfadeEnabled(enabled)
})

watch(() => playerStore.crossfadeDuration, (duration) => {
  audioEngine.setCrossfadeDuration(duration)
})

onMounted(() => {
  if (audioRef.value) {
    audioRef.value.volume = effectiveVolume.value
    if (playerStore.currentTrack && audioUrl.value) {
      audioRef.value.src = audioUrl.value
      audioRef.value.load()
    }
  }
  
  // Sync initial crossfade settings
  audioEngine.setCrossfadeEnabled(playerStore.crossfadeEnabled)
  audioEngine.setCrossfadeDuration(playerStore.crossfadeDuration)
})
</script>

<template>
  <div class="fixed bottom-0 left-0 right-0 z-50">
    <!-- Toast notification -->
    <Transition name="toast">
      <div 
        v-if="toastMessage"
        class="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-4 py-2 rounded-lg shadow-lg text-sm"
        :class="{
          'bg-yellow-600 text-white': toastMessage.type === 'warning',
          'bg-red-600 text-white': toastMessage.type === 'error',
          'bg-blue-600 text-white': toastMessage.type === 'info'
        }"
      >
        {{ toastMessage.text }}
      </div>
    </Transition>

    <!-- Queue Panel (Slide-out) -->
    <Transition name="slide">
      <div 
        v-if="showQueuePanel"
        class="absolute bottom-full right-0 w-80 max-h-96 bg-gray-800 border border-gray-700 rounded-t-lg shadow-xl overflow-hidden"
      >
        <div class="flex items-center justify-between px-4 py-3 border-b border-gray-700">
          <h3 class="font-semibold text-white">Queue</h3>
          <div class="flex items-center space-x-2">
            <span class="text-sm text-gray-400">{{ playerStore.queueLength }} tracks</span>
            <button 
              @click="playerStore.clearQueue()"
              class="text-xs text-gray-400 hover:text-white transition"
              title="Clear queue"
            >
              Clear
            </button>
          </div>
        </div>
        
        <div class="overflow-y-auto max-h-72">
          <div
            v-for="(item, index) in playerStore.queue"
            :key="item.id"
            :draggable="true"
            @dragstart="onDragStart(index)"
            @dragover="(e) => onDragOver(e, index)"
            @dragleave="onDragLeave"
            @drop="onDrop(index)"
            @dragend="onDragEnd"
            @dblclick="playQueueItem(index)"
            class="flex items-center px-4 py-2 hover:bg-gray-700 cursor-pointer transition group"
            :class="{
              'bg-gray-700': index === playerStore.currentIndex,
              'border-t-2 border-green-500': dragOverIndex === index
            }"
          >
            <!-- Drag handle -->
            <div class="w-6 text-gray-500 cursor-grab">⋮⋮</div>
            
            <!-- Track number / Playing indicator -->
            <div class="w-8 text-center">
              <span v-if="index === playerStore.currentIndex && playerStore.isPlaying" class="text-green-500">▶</span>
              <span v-else class="text-gray-500">{{ index + 1 }}</span>
            </div>
            
            <!-- Track info -->
            <div class="flex-1 min-w-0 ml-2">
              <div class="text-sm text-white truncate">{{ item.track.title }}</div>
              <div class="text-xs text-gray-400 truncate">{{ item.track.artist || 'Unknown Artist' }}</div>
            </div>
            
            <!-- Duration -->
            <div class="text-xs text-gray-400 ml-2">
              {{ formatTime(item.track.duration) }}
            </div>
            
            <!-- Remove button -->
            <button
              @click.stop="removeFromQueue(item.id)"
              class="ml-2 p-1 text-gray-500 hover:text-red-500 opacity-0 group-hover:opacity-100 transition"
              title="Remove from queue"
            >
              ✕
            </button>
          </div>
          
          <div v-if="playerStore.queue.length === 0" class="px-4 py-8 text-center text-gray-500">
            Queue is empty
          </div>
        </div>
      </div>
    </Transition>

    <!-- Main Player Bar -->
    <div class="bg-gray-900 text-white border-t border-gray-700">
      <!-- Hidden audio element -->
      <audio
        v-if="playerStore.currentTrack"
        ref="audioRef"
        :src="audioUrl"
        @loadedmetadata="onLoadedMetadata"
        @timeupdate="onTimeUpdate"
        @ended="onEnded"
        preload="auto"
      />

      <!-- Progress bar -->
      <div class="h-1 bg-gray-700 cursor-pointer group relative">
        <input
          type="range"
          min="0"
          :max="playerStore.duration || 100"
          :value="isSeeking ? seekValue : playerStore.currentTime"
          step="0.1"
          class="w-full h-1 appearance-none bg-transparent cursor-pointer absolute top-0 left-0 right-0 z-10"
          style="margin: 0;"
          @mousedown="onSeekStart"
          @touchstart="onSeekStart"
          @input="onSeekInput"
          @mouseup="onSeekEnd"
          @touchend="onSeekEnd"
        />
        <div 
          class="h-full bg-green-500 transition-all"
          :style="{ width: `${playerStore.progress}%` }"
        />
      </div>

      <div class="flex items-center justify-between px-4 py-3">
        <!-- Track info -->
        <div class="flex items-center space-x-3 w-1/4 min-w-0">
          <div class="w-14 h-14 bg-gray-700 rounded flex-shrink-0 flex items-center justify-center shadow-lg">
            <span class="text-2xl">🎵</span>
          </div>
          <div class="min-w-0">
            <div class="font-medium truncate text-white">
              {{ playerStore.currentTrack?.title || 'No track selected' }}
            </div>
            <div class="text-sm text-gray-400 truncate">
              {{ playerStore.currentTrack?.artist || 'Select a track to play' }}
            </div>
          </div>
        </div>

        <!-- Controls -->
        <div class="flex flex-col items-center space-y-1">
          <div class="flex items-center space-x-4">
            <!-- Shuffle -->
            <button 
              @click="playerStore.toggleShuffle()"
              class="p-2 hover:bg-gray-700 rounded-full transition-all duration-200"
              :class="{ 'text-green-500 bg-gray-800': playerStore.shuffleMode, 'text-gray-400': !playerStore.shuffleMode }"
              title="Shuffle"
            >
              🔀
            </button>

            <!-- Previous -->
            <button 
              @click="playerStore.previous()"
              :disabled="!playerStore.hasPrevious"
              class="p-2 hover:bg-gray-700 rounded-full transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Previous"
            >
              ⏮️
            </button>

            <!-- Play/Pause -->
            <button 
              @click="playerStore.isPlaying ? playerStore.pause() : playerStore.play()"
              class="p-4 bg-white text-black rounded-full hover:scale-110 transition-all duration-200 shadow-lg"
              :disabled="!playerStore.currentTrack"
              title="Play/Pause"
            >
              {{ playerStore.isPlaying ? '⏸️' : '▶️' }}
            </button>

            <!-- Next -->
            <button 
              @click="playerStore.next()"
              :disabled="!playerStore.hasNext"
              class="p-2 hover:bg-gray-700 rounded-full transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              title="Next"
            >
              ⏭️
            </button>

            <!-- Repeat -->
            <button 
              @click="playerStore.cycleRepeatMode()"
              class="p-2 hover:bg-gray-700 rounded-full transition-all duration-200"
              :class="{ 'text-green-500 bg-gray-800': playerStore.repeatMode !== 'off', 'text-gray-400': playerStore.repeatMode === 'off' }"
              :title="repeatLabel"
            >
              {{ repeatIcon }}
            </button>
          </div>

          <!-- Time -->
          <div class="flex items-center space-x-2 text-xs text-gray-400">
            <span>{{ formattedCurrentTime }}</span>
            <span>/</span>
            <span>{{ formattedDuration }}</span>
          </div>
        </div>

        <!-- Right side controls -->
        <div class="flex items-center space-x-4 w-1/4 justify-end">
          <!-- Crossfade toggle -->
          <button
            @click="playerStore.setCrossfade(!playerStore.crossfadeEnabled)"
            class="p-2 hover:bg-gray-700 rounded-full transition-all duration-200 text-sm"
            :class="{ 'text-green-500 bg-gray-800': playerStore.crossfadeEnabled, 'text-gray-400': !playerStore.crossfadeEnabled }"
            title="Crossfade"
          >
            ⚡
          </button>

          <!-- Queue button -->
          <button 
            @click="toggleQueuePanel"
            class="p-2 hover:bg-gray-700 rounded-full transition-all duration-200 flex items-center space-x-1"
            :class="{ 'text-green-500 bg-gray-800': showQueuePanel, 'text-gray-400': !showQueuePanel }"
            title="Queue"
          >
            <span>📋</span>
            <span class="text-xs">{{ playerStore.queueLength }}</span>
          </button>

          <!-- Volume -->
          <div class="flex items-center space-x-2">
            <button 
              @click="playerStore.toggleMute()"
              class="p-1 hover:bg-gray-700 rounded transition-all duration-200"
              title="Mute"
            >
              {{ playerStore.isMuted || playerStore.volume === 0 ? '🔇' : playerStore.volume < 0.5 ? '🔉' : '🔊' }}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              :value="playerStore.volume"
              @input="onVolumeChange"
              class="w-24 h-1 appearance-none bg-gray-600 rounded cursor-pointer"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 12px;
  height: 12px;
  background: #22c55e;
  border-radius: 50%;
  cursor: pointer;
  transition: transform 0.2s;
}

input[type="range"]::-webkit-slider-thumb:hover {
  transform: scale(1.2);
}

input[type="range"]::-moz-range-thumb {
  width: 12px;
  height: 12px;
  background: #22c55e;
  border-radius: 50%;
  cursor: pointer;
  border: none;
}

/* Slide transition for queue panel */
.slide-enter-active,
.slide-leave-active {
  transition: all 0.3s ease;
}

.slide-enter-from,
.slide-leave-to {
  transform: translateY(100%);
  opacity: 0;
}

/* Toast transition */
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}

.toast-enter-from,
.toast-leave-to {
  transform: translate(-50%, 10px);
  opacity: 0;
}
</style>
