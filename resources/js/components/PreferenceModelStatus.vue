<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import api from '@/api/client'
import { usePlayerStore } from '@/stores/playerStore'
import type { PreferenceModelStatus } from '@/types'

/**
 * PreferenceModelStatus Component
 * Requirements: 8.5, 8.6
 */

const status = ref<PreferenceModelStatus | null>(null)
const loading = ref(false)
const training = ref(false)
const pollInterval = ref<number | null>(null)
const playerStore = usePlayerStore()

const loadStatus = async () => {
    loading.value = true
    try {
        status.value = await api.getPreferenceModelStatus()
    } catch (error) {
        console.error('Failed to load model status:', error)
    } finally {
        loading.value = false
    }
}

const trainModel = async () => {
    training.value = true
    try {
        await api.trainPreferenceModel()
        startPolling()
    } catch (error) {
        console.error('Failed to start training:', error)
        training.value = false
    }
}

const startPolling = () => {
    if (pollInterval.value) return

    pollInterval.value = window.setInterval(async () => {
        await loadStatus()
        if (status.value?.status !== 'training') {
            stopPolling()
            training.value = false
            if (status.value?.status === 'trained') {
                await playerStore.refreshRadioQueue(true)
            }
        }
    }, 2000)
}

const stopPolling = () => {
    if (pollInterval.value) {
        clearInterval(pollInterval.value)
        pollInterval.value = null
    }
}

const getStatusIcon = (s: string) => {
    const icons: Record<string, string> = {
        not_trained: 'NEW',
        training: 'RUN',
        trained: 'OK',
        error: 'ERR',
    }
    return icons[s] || 'NEW'
}

const getStatusText = (s: string) => {
    const texts: Record<string, string> = {
        not_trained: 'Not Trained',
        training: 'Training...',
        trained: 'Ready',
        error: 'Error',
    }
    return texts[s] || s
}

const getStatusClass = (s: string) => {
    switch (s) {
        case 'trained':
            return 'status-pill--ready'
        case 'training':
            return 'status-pill--working'
        case 'error':
            return 'status-pill--failed'
        default:
            return 'status-pill--queued'
    }
}

const canTrain = () => {
    return status.value &&
           status.value.liked_tracks_count >= 10 &&
           status.value.status !== 'training'
}

onMounted(() => {
    loadStatus()
    if (status.value?.status === 'training') {
        training.value = true
        startPolling()
    }
})

onUnmounted(() => {
    stopPolling()
})
</script>

<template>
    <div class="surface-card !rounded-[1.75rem] !p-5">
        <div class="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div class="flex items-start gap-4">
                <div class="flex h-14 w-14 items-center justify-center rounded-[1.35rem] bg-sky-100 text-sm font-semibold tracking-[0.14em] text-sky-900 shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]">
                    {{ getStatusIcon(status?.status || 'not_trained') }}
                </div>
                <div>
                    <p class="section-kicker">Preference Model</p>
                    <div class="mt-2 flex flex-wrap items-center gap-2">
                        <h3 class="text-xl font-semibold text-slate-950">Training readiness</h3>
                        <span :class="['status-pill', getStatusClass(status?.status || 'not_trained')]">
                            {{ getStatusText(status?.status || 'not_trained') }}
                        </span>
                    </div>
                    <p class="mt-2 text-sm text-slate-500">
                        The model becomes trainable after enough likes are collected from the library.
                    </p>
                </div>
            </div>

            <div class="min-w-[12rem] rounded-[1.35rem] border border-slate-200 bg-slate-50/80 p-4 text-sm text-slate-600">
                <div class="flex items-center justify-between gap-3">
                    <span class="font-medium text-slate-500">Likes collected</span>
                    <span class="font-semibold text-slate-950">{{ status?.liked_tracks_count || 0 }} / 10</span>
                </div>
                <div class="mt-3 h-2 rounded-full bg-slate-200">
                    <div
                        class="h-2 rounded-full bg-gradient-to-r from-blue-600 to-sky-500 transition-all"
                        :style="{ width: `${Math.min((status?.liked_tracks_count || 0) / 10 * 100, 100)}%` }"
                    ></div>
                </div>
            </div>
        </div>

        <div class="mt-4 flex flex-wrap gap-3 text-sm text-slate-500">
            <span class="badge-soft">Ready after 10 likes</span>
            <span v-if="status?.last_trained_at" class="badge-soft">
                Last trained: {{ new Date(status.last_trained_at).toLocaleString() }}
            </span>
        </div>

        <div v-if="status?.error_message" class="mt-4 rounded-[1.25rem] border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {{ status.error_message }}
        </div>

        <button
            v-if="!training"
            @click="trainModel"
            :disabled="!canTrain()"
            class="action-primary mt-5 w-full disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
        >
            {{ canTrain() ? 'Train Model' : `Need ${10 - (status?.liked_tracks_count || 0)} more likes` }}
        </button>

        <div v-else class="mt-5 rounded-[1.35rem] border border-sky-200 bg-sky-50/80 px-4 py-5 text-center">
            <div class="inline-block h-6 w-6 animate-spin rounded-full border-2 border-blue-600 border-t-transparent"></div>
            <p class="mt-2 text-sm font-medium text-sky-800">Training in progress...</p>
        </div>
    </div>
</template>
