<script setup lang="ts">
/**
 * SmartRadioToggle - Smart Radio UI controls
 * Requirements: 15.1, 15.6, 17.5, 19.5
 */
import { ref, computed } from 'vue'
import { usePlayerStore } from '@/stores/playerStore'
import { api } from '@/api/client'

const playerStore = usePlayerStore()

const isLoading = ref(false)
const feedbackSent = ref<'more' | 'less' | null>(null)

const isEnabled = computed(() => playerStore.smartRadioEnabled)

const toggleSmartRadio = async () => {
  isLoading.value = true
  try {
    if (isEnabled.value) {
      await api.disableSmartRadio()
      playerStore.setSmartRadio(false)
    } else {
      const trackId = playerStore.currentTrack?.id
      if (trackId) {
        await api.enableSmartRadio(trackId)
        playerStore.setSmartRadio(true)
      }
    }
  } catch (e) {
    console.error('Error toggling smart radio:', e)
  } finally {
    isLoading.value = false
  }
}

const sendFeedback = async (type: 'more' | 'less') => {
  const trackId = playerStore.currentTrack?.id
  if (!trackId) return

  try {
    await api.sendRadioFeedback(trackId, type)
    feedbackSent.value = type
    setTimeout(() => {
      feedbackSent.value = null
    }, 2000)
  } catch (e) {
    console.error('Error sending feedback:', e)
  }
}
</script>

<template>
  <div class="flex flex-wrap items-center gap-2">
    <button
      @click="toggleSmartRadio"
      :disabled="isLoading"
      class="feature-toggle"
      :class="{ 'feature-toggle--active': isEnabled }"
      :title="isEnabled ? 'Disable Smart Radio' : 'Enable Smart Radio'"
    >
      <span class="feature-toggle__kicker">Radio</span>
      <span class="feature-toggle__label">
        {{ isLoading ? 'Working' : (isEnabled ? 'Enabled' : 'Smart') }}
      </span>
    </button>

    <Transition name="slide">
      <div v-if="isEnabled" class="flex items-center gap-2">
        <button
          @click="sendFeedback('more')"
          class="feedback-chip"
          :class="{ 'feedback-chip--positive': feedbackSent === 'more' }"
          title="More like this"
        >
          More
        </button>
        <button
          @click="sendFeedback('less')"
          class="feedback-chip"
          :class="{ 'feedback-chip--negative': feedbackSent === 'less' }"
          title="Less like this"
        >
          Less
        </button>
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
  border-color: rgba(125, 211, 252, 0.3);
  background: rgba(255, 255, 255, 0.08);
}

.feature-toggle:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.feature-toggle--active {
  border-color: rgba(56, 189, 248, 0.28);
  background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(37, 99, 235, 0.18));
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

.feedback-chip {
  min-height: 2.5rem;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.05);
  padding: 0.65rem 0.95rem;
  color: #cbd5e1;
  font-size: 0.75rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  transition:
    transform 0.18s ease,
    border-color 0.18s ease,
    background-color 0.18s ease;
}

.feedback-chip:hover {
  transform: translateY(-1px);
}

.feedback-chip--positive {
  border-color: rgba(52, 211, 153, 0.28);
  background: rgba(16, 185, 129, 0.18);
  color: #d1fae5;
}

.feedback-chip--negative {
  border-color: rgba(248, 113, 113, 0.28);
  background: rgba(239, 68, 68, 0.18);
  color: #fee2e2;
}

.slide-enter-active,
.slide-leave-active {
  transition: all 0.22s ease;
}

.slide-enter-from,
.slide-leave-to {
  opacity: 0;
  transform: translateX(-8px);
}
</style>
