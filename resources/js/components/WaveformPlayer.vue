<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch, computed } from 'vue'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js'
import type { Fragment } from '@/types'

/**
 * WaveformPlayer Component
 * Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 6.1, 6.3, 6.4, 7.1, 8.1
 * 
 * Supports fragment creation with:
 * - Like/Dislike rating
 * - Text comment describing what user liked/disliked
 */

interface Props {
    audioUrl: string
    fragments?: Fragment[]
    duration?: number
    externalAudio?: boolean
}

interface Emits {
    (e: 'fragment-created', data: { start_time: number; end_time: number; is_liked?: boolean; comment?: string }): void
    (e: 'fragment-clicked', fragment: Fragment): void
    (e: 'time-update', time: number): void
    (e: 'save-as-pattern', data: { start_time: number; end_time: number }): void
    (e: 'seek', time: number): void
    (e: 'play-requested'): void
    (e: 'pause-requested'): void
}

const props = withDefaults(defineProps<Props>(), {
    fragments: () => [],
    duration: 0,
    externalAudio: false,
})

const emit = defineEmits<Emits>()

const waveformRef = ref<HTMLDivElement | null>(null)
const wavesurfer = ref<WaveSurfer | null>(null)
const regions = ref<RegionsPlugin | null>(null)

const isPlaying = ref(false)
const currentTime = ref(0)
// Load volume from localStorage, default to 1
const savedVolume = localStorage.getItem('player_volume')
const volume = ref(savedVolume !== null ? parseFloat(savedVolume) : 1)
const zoomPercent = ref(100)
const isSyncingFromExternal = ref(false)
const MIN_ZOOM_PERCENT = 100
const MAX_ZOOM_PERCENT = 800

// Fragment creation state
const showRatingDialog = ref(false)
const pendingSelection = ref<{ start: number; end: number } | null>(null)
const selectedRating = ref<boolean | null>(null)
const fragmentComment = ref('')
const hoveredFragment = ref<Fragment | null>(null)
const tooltipPosition = ref({ x: 0, y: 0 })

const hasSelection = computed(() => pendingSelection.value !== null)
const selectionDuration = computed(() => {
    if (!pendingSelection.value) return 0
    return Math.abs(pendingSelection.value.end - pendingSelection.value.start)
})

const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

const clamp = (value: number, min: number, max: number): number => (
    Math.max(min, Math.min(max, value))
)

const fitMinPxPerSec = (): number => {
    const duration = wavesurfer.value?.getDuration() || props.duration || 0
    const width = waveformRef.value?.clientWidth ?? wavesurfer.value?.getWrapper().clientWidth ?? 0
    if (duration <= 0 || width <= 0) return 0
    return width / duration
}

const initWaveSurfer = () => {
    if (!waveformRef.value) return

    regions.value = RegionsPlugin.create()

    wavesurfer.value = WaveSurfer.create({
        container: waveformRef.value,
        waveColor: '#4F46E5',
        progressColor: '#818CF8',
        cursorColor: '#EF4444',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 2,
        height: 128,
        barGap: 2,
        plugins: [regions.value],
    })

    wavesurfer.value.load(props.audioUrl)

    wavesurfer.value.on('ready', () => {
        renderFragments()
        wavesurfer.value?.setVolume(props.externalAudio ? 0 : volume.value)
        applyZoomPercent(zoomPercent.value)
    })

    wavesurfer.value.on('play', () => {
        isPlaying.value = true
    })

    wavesurfer.value.on('pause', () => {
        isPlaying.value = false
    })

    wavesurfer.value.on('timeupdate', (time) => {
        currentTime.value = time
        emit('time-update', time)
    })

    wavesurfer.value.on('click', (relativeX) => {
        if (!wavesurfer.value) return
        const duration = wavesurfer.value.getDuration()
        emit('seek', relativeX * duration)
    })

    // Enable region creation by dragging
    regions.value.enableDragSelection({
        color: 'rgba(147, 51, 234, 0.4)', // Purple for selection
    })

    regions.value.on('region-created', (region) => {
        // Only handle user-created regions (not loaded fragments)
        if (!region.id.startsWith('fragment-')) {
            // Store selection and show rating dialog
            pendingSelection.value = {
                start: region.start,
                end: region.end,
            }
            showRatingDialog.value = true
            selectedRating.value = null
            fragmentComment.value = ''
            // Keep region visible
            region.setOptions({ color: 'rgba(147, 51, 234, 0.5)' })
        }
    })

    regions.value.on('region-clicked', (region, e) => {
        e.stopPropagation()
        const fragmentId = parseInt(region.id.replace('fragment-', ''))
        const fragment = props.fragments.find((f) => f.id === fragmentId)
        if (fragment) {
            emit('fragment-clicked', fragment)
            if (props.externalAudio) {
                emit('seek', fragment.start_time)
                emit('play-requested')
            } else {
                region.play()
            }
        }
    })
}

const renderFragments = () => {
    if (!regions.value) return

    // Clear existing fragment regions
    regions.value.getRegions().forEach((region) => {
        if (region.id.startsWith('fragment-')) {
            region.remove()
        }
    })

    // Add fragment regions
    props.fragments.forEach((fragment) => {
        const region = regions.value?.addRegion({
            id: `fragment-${fragment.id}`,
            start: fragment.start_time,
            end: fragment.end_time,
            color: fragment.is_liked
                ? 'rgba(34, 197, 94, 0.3)'
                : 'rgba(239, 68, 68, 0.3)',
            drag: false,
            resize: false,
        })
        
        // Add hover events for tooltip (especially for fragments with comments)
        if (region) {
            const el = region.element
            el.addEventListener('mouseenter', (e) => {
                hoveredFragment.value = fragment
                tooltipPosition.value = { x: e.clientX, y: e.clientY - 10 }
            })
            el.addEventListener('mousemove', (e) => {
                tooltipPosition.value = { x: e.clientX, y: e.clientY - 10 }
            })
            el.addEventListener('mouseleave', () => {
                hoveredFragment.value = null
            })
        }
    })
}

const play = () => {
    if (!isSyncingFromExternal.value) {
        emit('play-requested')
    }
    if (!props.externalAudio) {
        wavesurfer.value?.play()
    }
}
const pause = () => {
    if (!isSyncingFromExternal.value) {
        emit('pause-requested')
    }
    if (!props.externalAudio) {
        wavesurfer.value?.pause()
    }
}
const stop = () => {
    if (props.externalAudio) {
        emit('pause-requested')
        emit('seek', 0)
    } else {
        wavesurfer.value?.stop()
    }
    currentTime.value = 0
    isPlaying.value = false
}

const togglePlay = () => {
    if (isPlaying.value) {
        pause()
    } else {
        play()
    }
}

const setVolume = (value: number) => {
    volume.value = value
    wavesurfer.value?.setVolume(props.externalAudio ? 0 : value)
    // Save to localStorage
    localStorage.setItem('player_volume', value.toString())
}

const applyZoomPercent = (nextZoomPercent: number) => {
    const normalizedPercent = clamp(nextZoomPercent, MIN_ZOOM_PERCENT, MAX_ZOOM_PERCENT)
    zoomPercent.value = normalizedPercent

    if (!wavesurfer.value) return

    if (normalizedPercent === 100) {
        wavesurfer.value.zoom(0)
        return
    }

    const fitZoom = fitMinPxPerSec()
    if (fitZoom <= 0) {
        wavesurfer.value.zoom(0)
        return
    }

    wavesurfer.value.zoom(fitZoom * (normalizedPercent / 100))
}

const setZoom = (value: number) => {
    applyZoomPercent(value)
}

const seekTo = (time: number) => {
    if (props.externalAudio) {
        emit('seek', time)
        return
    }
    if (wavesurfer.value && props.duration > 0) {
        wavesurfer.value.seekTo(time / props.duration)
    }
}

const syncPlay = () => {
    isSyncingFromExternal.value = true
    wavesurfer.value?.play()
    setTimeout(() => { isSyncingFromExternal.value = false }, 100)
}

const syncPause = () => {
    isSyncingFromExternal.value = true
    wavesurfer.value?.pause()
    setTimeout(() => { isSyncingFromExternal.value = false }, 100)
}

const syncSeekTo = (time: number) => {
    isSyncingFromExternal.value = true
    if (wavesurfer.value && props.duration > 0) {
        wavesurfer.value.seekTo(time / props.duration)
    }
    setTimeout(() => { isSyncingFromExternal.value = false }, 100)
}

// Fragment creation handlers
const submitFragment = (isLiked: boolean) => {
    if (!pendingSelection.value) return
    
    selectedRating.value = isLiked
    
    emit('fragment-created', {
        start_time: Math.min(pendingSelection.value.start, pendingSelection.value.end),
        end_time: Math.max(pendingSelection.value.start, pendingSelection.value.end),
        is_liked: isLiked,
        comment: fragmentComment.value.trim() || undefined,
    })
    
    clearSelection()
}

const saveAsPattern = () => {
    if (!pendingSelection.value) return
    
    emit('save-as-pattern', {
        start_time: Math.min(pendingSelection.value.start, pendingSelection.value.end),
        end_time: Math.max(pendingSelection.value.start, pendingSelection.value.end),
    })
    
    clearSelection()
}

const clearSelection = () => {
    // Remove temporary selection region
    if (regions.value) {
        regions.value.getRegions().forEach((region) => {
            if (!region.id.startsWith('fragment-')) {
                region.remove()
            }
        })
    }
    pendingSelection.value = null
    showRatingDialog.value = false
    selectedRating.value = null
    fragmentComment.value = ''
}

watch(() => props.audioUrl, () => {
    wavesurfer.value?.load(props.audioUrl)
})

watch(() => props.fragments, () => {
    renderFragments()
}, { deep: true })

watch(() => props.externalAudio, (external) => {
    if (!wavesurfer.value) return
    if (external) {
        wavesurfer.value.pause()
        isPlaying.value = false
    }
    wavesurfer.value.setVolume(external ? 0 : volume.value)
})

onMounted(() => {
    initWaveSurfer()
})

onUnmounted(() => {
    wavesurfer.value?.destroy()
})

defineExpose({
    play,
    pause,
    stop,
    seekTo,
    syncPlay,
    syncPause,
    syncSeekTo,
})
</script>

<template>
    <div class="waveform-container relative">
        <!-- Waveform Display -->
        <div ref="waveformRef" class="mb-4"></div>

        <!-- Fragment Rating Dialog -->
        <Transition name="slide-up">
            <div 
                v-if="showRatingDialog && hasSelection"
                class="absolute top-0 left-0 right-0 bg-gray-800/95 backdrop-blur rounded-lg p-4 shadow-xl border border-gray-700 z-10"
            >
                <div class="flex items-start justify-between mb-3">
                    <div>
                        <h3 class="text-white font-medium">Оцените этот отрывок</h3>
                        <p class="text-gray-400 text-sm">
                            {{ formatTime(pendingSelection?.start || 0) }} - {{ formatTime(pendingSelection?.end || 0) }}
                            ({{ selectionDuration.toFixed(1) }}s)
                        </p>
                    </div>
                    <button 
                        @click="clearSelection"
                        class="text-gray-400 hover:text-white transition"
                    >
                        ✕
                    </button>
                </div>

                <!-- Comment Input -->
                <div class="mb-4">
                    <label class="block text-sm text-gray-300 mb-1">
                        Что именно вам понравилось/не понравилось? (опционально)
                    </label>
                    <textarea
                        v-model="fragmentComment"
                        placeholder="Например: крутой бас-дроп, интересная мелодия, слишком резкий переход..."
                        class="w-full bg-gray-700 text-white rounded-lg px-3 py-2 text-sm placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none"
                        rows="2"
                        maxlength="500"
                    ></textarea>
                    <p class="text-xs text-gray-500 mt-1">{{ fragmentComment.length }}/500</p>
                </div>

                <!-- Rating Buttons -->
                <div class="flex items-center space-x-3">
                    <button
                        @click="submitFragment(true)"
                        class="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition flex items-center justify-center space-x-2"
                    >
                        <span class="text-lg">👍</span>
                        <span>Нравится</span>
                    </button>
                    <button
                        @click="submitFragment(false)"
                        class="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition flex items-center justify-center space-x-2"
                    >
                        <span class="text-lg">👎</span>
                        <span>Не нравится</span>
                    </button>
                </div>

                <!-- Save as Pattern Button -->
                <div class="mt-3 pt-3 border-t border-gray-700">
                    <button
                        @click="saveAsPattern"
                        class="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition flex items-center justify-center space-x-2"
                    >
                        <span class="text-lg">🎵</span>
                        <span>Сохранить как паттерн</span>
                    </button>
                </div>

                <p class="text-xs text-gray-500 mt-3 text-center">
                    💡 Комментарии помогают модели лучше понять ваши предпочтения
                </p>
            </div>
        </Transition>

        <!-- Fragment Comment Tooltip -->
        <Teleport to="body">
            <Transition name="fade">
                <div
                    v-if="hoveredFragment"
                    class="fixed z-50 bg-gray-900 text-white px-4 py-3 rounded-lg shadow-xl text-sm max-w-sm pointer-events-none"
                    :style="{
                        left: `${tooltipPosition.x}px`,
                        top: `${tooltipPosition.y}px`,
                        transform: 'translate(-50%, -100%)'
                    }"
                >
                    <div class="flex items-center space-x-2 mb-1">
                        <span :class="hoveredFragment.is_liked ? 'text-green-400' : 'text-red-400'" class="text-lg">
                            {{ hoveredFragment.is_liked ? '👍' : '👎' }}
                        </span>
                        <span class="font-medium text-gray-200">
                            {{ formatTime(hoveredFragment.start_time) }} - {{ formatTime(hoveredFragment.end_time) }}
                        </span>
                    </div>
                    <div v-if="hoveredFragment.comment" class="text-gray-300 mt-2 border-t border-gray-700 pt-2">
                        <span class="text-gray-500 text-xs block mb-1">💬 Комментарий:</span>
                        <p class="italic">{{ hoveredFragment.comment }}</p>
                    </div>
                </div>
            </Transition>
        </Teleport>

        <!-- Controls -->
        <div class="flex items-center justify-between">
            <!-- Playback Controls -->
            <div class="flex items-center space-x-4">
                <button
                    @click="togglePlay"
                    class="w-12 h-12 rounded-full bg-blue-600 text-white flex items-center justify-center hover:bg-blue-700 transition-colors"
                >
                    <span v-if="isPlaying" class="text-xl">⏸</span>
                    <span v-else class="text-xl">▶</span>
                </button>
                <button
                    @click="stop"
                    class="w-10 h-10 rounded-full bg-gray-600 text-white flex items-center justify-center hover:bg-gray-700 transition-colors"
                >
                    ⏹
                </button>
            </div>

            <!-- Time Display -->
            <div class="text-white font-mono">
                {{ formatTime(currentTime) }} / {{ formatTime(duration || 0) }}
            </div>

            <!-- Volume Control -->
            <div class="flex items-center space-x-2">
                <span class="text-white">🔊</span>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    :value="volume"
                    @input="setVolume(parseFloat(($event.target as HTMLInputElement).value))"
                    class="w-24"
                />
            </div>

            <!-- Zoom Control -->
            <div class="flex items-center space-x-2">
                <span class="text-white">🔍</span>
                <span class="text-white text-xs font-semibold">{{ zoomPercent }}%</span>
                <input
                    type="range"
                    min="100"
                    max="800"
                    step="10"
                    :value="zoomPercent"
                    @input="setZoom(parseInt(($event.target as HTMLInputElement).value))"
                    class="w-24"
                />
            </div>
        </div>

        <!-- Fragment Legend -->
        <div v-if="fragments.length > 0" class="mt-4 flex items-center space-x-4 text-sm text-gray-300">
            <div class="flex items-center">
                <span class="w-4 h-4 bg-green-500/30 rounded mr-2"></span>
                Fragment
            </div>
            <div class="flex items-center">
                <span class="w-4 h-4 bg-red-500/30 rounded mr-2"></span>
                Liked Fragment
            </div>
        </div>

        <!-- Instructions -->
        <p class="mt-2 text-xs text-gray-400">
            🎯 Выделите область на waveform чтобы оценить отрывок и описать что вам понравилось
        </p>
    </div>
</template>

<style scoped>
.slide-up-enter-active,
.slide-up-leave-active {
    transition: all 0.3s ease;
}

.slide-up-enter-from,
.slide-up-leave-to {
    opacity: 0;
    transform: translateY(-10px);
}

.fade-enter-active,
.fade-leave-active {
    transition: opacity 0.15s ease;
}

.fade-enter-from,
.fade-leave-to {
    opacity: 0;
}
</style>
