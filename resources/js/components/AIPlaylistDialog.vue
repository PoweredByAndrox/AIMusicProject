<script setup lang="ts">
/**
 * AIPlaylistDialog - AI Playlist Generator UI
 * Requirements: 6.1, 6.4
 */
import { ref, computed } from 'vue'
import { api } from '@/api/client'

interface Props {
  visible: boolean
}

interface Emits {
  (e: 'close'): void
  (e: 'created', playlist: any): void
}

const props = defineProps<Props>()
const emit = defineEmits<Emits>()

// Form state
const name = ref('')
const duration = ref(60) // minutes
const genre = ref('')
const mood = ref('')
const minBpm = ref<number | null>(null)
const maxBpm = ref<number | null>(null)
const trackCount = ref<number | null>(null)
const usePreferences = ref(true) // Use liked tracks for recommendations

// UI state
const isGenerating = ref(false)
const error = ref<string | null>(null)

// Options - Electronic music genres that match ML analysis output
const genres = [
  'Dubstep', 'Drum & Bass', 'Grime', 'House', 'Tech House', 'Deep House',
  'Trance', 'Techno', 'Breakbeat', 'Jungle', 'UK Garage', 'Electro',
  'Progressive House', 'Ambient', 'Downtempo', 'IDM'
]
const moods = ['Happy', 'Sad', 'Aggressive', 'Relaxed', 'Energetic', 'Dark', 'Uplifting', 'Melancholic']

// Computed
const isValid = computed(() => name.value.trim().length > 0)

// Methods
const generatePlaylist = async () => {
  if (!isValid.value) return
  
  isGenerating.value = true
  error.value = null
  
  try {
    const params: {
      name: string
      duration: number
      genre?: string
      mood?: string
      bpm_min?: number
      bpm_max?: number
      track_count?: number
      use_preferences?: boolean
    } = {
      name: name.value.trim(),
      duration: duration.value,
      use_preferences: usePreferences.value,
    }
    
    if (genre.value) params.genre = genre.value
    if (mood.value) params.mood = mood.value
    if (minBpm.value) params.bpm_min = minBpm.value
    if (maxBpm.value) params.bpm_max = maxBpm.value
    if (trackCount.value) params.track_count = trackCount.value
    
    const response = await api.generateAIPlaylist(params)
    emit('created', response)
    resetForm()
    emit('close')
  } catch (e: any) {
    error.value = e.response?.data?.message || 'Failed to generate playlist'
    console.error('Error generating playlist:', e)
  } finally {
    isGenerating.value = false
  }
}

const resetForm = () => {
  name.value = ''
  duration.value = 60
  genre.value = ''
  mood.value = ''
  minBpm.value = null
  maxBpm.value = null
  trackCount.value = null
  usePreferences.value = true
  error.value = null
}

const close = () => {
  resetForm()
  emit('close')
}
</script>

<template>
  <Teleport to="body">
    <Transition name="fade">
      <div
        v-if="visible"
        class="fixed inset-0 bg-black/70 flex items-center justify-center z-50"
        @click.self="close"
      >
        <div class="bg-gray-800 rounded-xl p-6 w-full max-w-lg shadow-2xl">
          <!-- Header -->
          <div class="flex items-center justify-between mb-6">
            <div class="flex items-center space-x-3">
              <span class="text-2xl">🤖</span>
              <h3 class="text-xl font-bold text-white">AI Playlist Generator</h3>
            </div>
            <button
              @click="close"
              class="p-2 hover:bg-gray-700 rounded-full transition"
            >
              ✕
            </button>
          </div>

          <!-- Error message -->
          <div v-if="error" class="mb-4 p-3 bg-red-900/50 text-red-300 rounded-lg">
            {{ error }}
          </div>

          <!-- Form -->
          <div class="space-y-4">
            <!-- Name -->
            <div>
              <label class="block text-sm text-gray-400 mb-1">Playlist Name *</label>
              <input
                v-model="name"
                type="text"
                placeholder="My AI Playlist"
                class="w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            <!-- Duration -->
            <div>
              <label class="block text-sm text-gray-400 mb-1">
                Duration: {{ duration }} minutes
              </label>
              <input
                v-model="duration"
                type="range"
                min="15"
                max="180"
                step="15"
                class="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
              />
            </div>

            <!-- Genre & Mood -->
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm text-gray-400 mb-1">Genre</label>
                <select
                  v-model="genre"
                  class="w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="">Any</option>
                  <option v-for="g in genres" :key="g" :value="g">{{ g }}</option>
                </select>
              </div>
              <div>
                <label class="block text-sm text-gray-400 mb-1">Mood</label>
                <select
                  v-model="mood"
                  class="w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                >
                  <option value="">Any</option>
                  <option v-for="m in moods" :key="m" :value="m">{{ m }}</option>
                </select>
              </div>
            </div>

            <!-- BPM Range -->
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label class="block text-sm text-gray-400 mb-1">Min BPM</label>
                <input
                  v-model.number="minBpm"
                  type="number"
                  min="60"
                  max="200"
                  placeholder="60"
                  class="w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>
              <div>
                <label class="block text-sm text-gray-400 mb-1">Max BPM</label>
                <input
                  v-model.number="maxBpm"
                  type="number"
                  min="60"
                  max="200"
                  placeholder="200"
                  class="w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
                />
              </div>
            </div>

            <!-- Track Count -->
            <div>
              <label class="block text-sm text-gray-400 mb-1">Track Count (optional)</label>
              <input
                v-model.number="trackCount"
                type="number"
                min="5"
                max="100"
                placeholder="Auto (based on duration)"
                class="w-full px-4 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              />
            </div>

            <!-- Use Preferences -->
            <div class="flex items-center space-x-3 p-3 bg-gray-700/50 rounded-lg">
              <input
                v-model="usePreferences"
                type="checkbox"
                id="usePreferences"
                class="w-5 h-5 rounded bg-gray-600 border-gray-500 text-purple-600 focus:ring-purple-500"
              />
              <label for="usePreferences" class="text-sm text-gray-300">
                <span class="font-medium">Use my preferences</span>
                <span class="block text-xs text-gray-500">
                  Prioritize tracks similar to ones you've liked
                </span>
              </label>
            </div>
          </div>

          <!-- Actions -->
          <div class="flex justify-end space-x-3 mt-6">
            <button
              @click="close"
              class="px-4 py-2 text-gray-400 hover:text-white transition"
            >
              Cancel
            </button>
            <button
              @click="generatePlaylist"
              :disabled="!isValid || isGenerating"
              class="px-6 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              <span v-if="isGenerating" class="animate-spin">⏳</span>
              <span>{{ isGenerating ? 'Generating...' : 'Generate Playlist' }}</span>
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
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

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 16px;
  height: 16px;
  background: linear-gradient(to right, #9333ea, #ec4899);
  border-radius: 50%;
  cursor: pointer;
}
</style>
