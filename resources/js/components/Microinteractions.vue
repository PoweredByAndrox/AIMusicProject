<script setup lang="ts">
/**
 * Microinteractions - Toast notifications, skeleton loaders, pulse effects
 * Requirements: 24.1, 24.2, 24.3, 24.4
 */
import { ref, provide } from 'vue'

interface Toast {
  id: number
  message: string
  type: 'success' | 'error' | 'info' | 'warning'
  duration: number
}

// Toast state
const toasts = ref<Toast[]>([])
let toastId = 0

// Methods
const showToast = (message: string, type: Toast['type'] = 'info', duration = 3000) => {
  const id = ++toastId
  toasts.value.push({ id, message, type, duration })
  
  setTimeout(() => {
    removeToast(id)
  }, duration)
  
  return id
}

const removeToast = (id: number) => {
  const index = toasts.value.findIndex(t => t.id === id)
  if (index !== -1) {
    toasts.value.splice(index, 1)
  }
}

const success = (message: string) => showToast(message, 'success')
const error = (message: string) => showToast(message, 'error')
const info = (message: string) => showToast(message, 'info')
const warning = (message: string) => showToast(message, 'warning')

// Provide toast functions
provide('toast', { showToast, success, error, info, warning })

// Get icon for toast type
const getToastIcon = (type: Toast['type']) => {
  switch (type) {
    case 'success': return '✓'
    case 'error': return '✕'
    case 'warning': return '⚠'
    default: return 'ℹ'
  }
}

const getToastClass = (type: Toast['type']) => {
  switch (type) {
    case 'success': return 'bg-green-600'
    case 'error': return 'bg-red-600'
    case 'warning': return 'bg-yellow-600'
    default: return 'bg-blue-600'
  }
}
</script>

<template>
  <div class="microinteractions">
    <!-- Toast Container -->
    <Teleport to="body">
      <div class="fixed top-4 right-4 z-50 space-y-2">
        <TransitionGroup name="toast">
          <div
            v-for="toast in toasts"
            :key="toast.id"
            class="flex items-center px-4 py-3 rounded-lg shadow-lg text-white min-w-64 max-w-sm"
            :class="getToastClass(toast.type)"
          >
            <span class="w-6 h-6 flex items-center justify-center bg-white/20 rounded-full mr-3 text-sm">
              {{ getToastIcon(toast.type) }}
            </span>
            <span class="flex-1">{{ toast.message }}</span>
            <button
              @click="removeToast(toast.id)"
              class="ml-3 p-1 hover:bg-white/20 rounded transition"
            >
              ✕
            </button>
          </div>
        </TransitionGroup>
      </div>
    </Teleport>

    <!-- Main content -->
    <slot />
  </div>
</template>

<style scoped>
/* Toast animations */
.toast-enter-active {
  animation: toast-in 0.3s ease;
}

.toast-leave-active {
  animation: toast-out 0.3s ease;
}

@keyframes toast-in {
  from {
    opacity: 0;
    transform: translateX(100px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes toast-out {
  from {
    opacity: 1;
    transform: translateX(0);
  }
  to {
    opacity: 0;
    transform: translateX(100px);
  }
}
</style>
