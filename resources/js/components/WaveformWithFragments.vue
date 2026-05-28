<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js'
import type { Fragment } from '@/types'

interface Props {
  audioUrl: string
  fragments?: Fragment[]
  duration?: number
  trackId?: number
  externalAudio?: boolean
}

interface Emits {
  (e: 'fragment-created', data: { start_time: number; end_time: number; is_liked: boolean }): void
  (e: 'fragment-clicked', fragment: Fragment): void
  (e: 'fragment-rate', data: { fragmentId: number; isLiked: boolean }): void
  (e: 'time-update', time: number): void
  (e: 'seek', time: number): void
  (e: 'play-requested'): void
  (e: 'pause-requested'): void
}

interface WaveformPayload {
  duration: number
  peaks: number[]
}

type WaveformCacheStore = Map<number, WaveformPayload>
type WaveformRequestStore = Map<number, Promise<WaveformPayload>>

const props = withDefaults(defineProps<Props>(), {
  fragments: () => [],
  duration: 0,
  trackId: 0,
  externalAudio: false,
})

const emit = defineEmits<Emits>()

const waveformPayloadCache = ((globalThis as typeof globalThis & {
  __aimWaveformPayloadCache?: WaveformCacheStore
}).__aimWaveformPayloadCache ??= new Map<number, WaveformPayload>())

const waveformPayloadRequests = ((globalThis as typeof globalThis & {
  __aimWaveformPayloadRequests?: WaveformRequestStore
}).__aimWaveformPayloadRequests ??= new Map<number, Promise<WaveformPayload>>())

function parseWaveformPayload(payload: unknown): WaveformPayload {
  const candidate = payload as { duration?: unknown; peaks?: unknown }
  const duration = Number(candidate?.duration || 0)
  const peaks = Array.isArray(candidate?.peaks)
    ? candidate.peaks.map((value) => Number(value)).filter((value) => Number.isFinite(value))
    : []

  if (duration <= 0 || peaks.length === 0) {
    throw new Error('Waveform payload is invalid')
  }

  return {
    duration,
    peaks,
  }
}

async function fetchWaveformPayload(trackId: number): Promise<WaveformPayload> {
  const cached = waveformPayloadCache.get(trackId)
  if (cached) {
    return cached
  }

  const pending = waveformPayloadRequests.get(trackId)
  if (pending) {
    return pending
  }

  const request = fetch(`/api/tracks/${trackId}/waveform`, {
    headers: {
      Accept: 'application/json',
    },
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Waveform request failed (${response.status})`)
      }

      const payload = parseWaveformPayload(await response.json())
      waveformPayloadCache.set(trackId, payload)
      return payload
    })
    .finally(() => {
      waveformPayloadRequests.delete(trackId)
    })

  waveformPayloadRequests.set(trackId, request)
  return request
}

const waveformRef = ref<HTMLDivElement | null>(null)
const wavesurfer = ref<WaveSurfer | null>(null)
const regions = ref<ReturnType<typeof RegionsPlugin.create> | null>(null)

const isPlaying = ref(false)
const currentTime = ref(0)
const savedVolume = localStorage.getItem('player_volume')
const volume = ref(savedVolume !== null ? parseFloat(savedVolume) : 1)
const zoomPercent = ref(100)
const selectionStart = ref<number | null>(null)
const selectionEnd = ref<number | null>(null)
const hoveredFragment = ref<Fragment | null>(null)
const tooltipPosition = ref({ x: 0, y: 0 })
const showRatingButtons = ref(false)
const isSyncingFromExternal = ref(false)
const waveformStatus = ref(props.externalAudio ? 'Loading synced waveform...' : 'Loading waveform...')
const waveformDuration = ref(props.duration || 0)
const MIN_ZOOM_PERCENT = 100
const MAX_ZOOM_PERCENT = 800

let waveformLoadRequestId = 0

const hasSelection = computed(() => (
  selectionStart.value !== null && selectionEnd.value !== null
))

const selectionDuration = computed(() => {
  if (!hasSelection.value) return 0
  return Math.abs((selectionEnd.value || 0) - (selectionStart.value || 0))
})

const effectiveDuration = computed(() => (
  waveformDuration.value > 0 ? waveformDuration.value : props.duration
))

const formatTime = (seconds: number): string => {
  if (!seconds || Number.isNaN(seconds)) return '0:00'
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const clamp = (value: number, min: number, max: number): number => (
  Math.max(min, Math.min(max, value))
)

const fitMinPxPerSec = (): number => {
  const duration = effectiveDuration.value || wavesurfer.value?.getDuration() || 0
  const width = waveformRef.value?.clientWidth ?? wavesurfer.value?.getWrapper().clientWidth ?? 0
  if (duration <= 0 || width <= 0) return 0
  return width / duration
}

const getFragmentColor = (fragment: Fragment): string => {
  if (fragment.is_liked === true) return 'rgba(34, 197, 94, 0.38)'
  if (fragment.is_liked === false) return 'rgba(239, 68, 68, 0.34)'
  return 'rgba(56, 189, 248, 0.28)'
}

async function loadWaveformSource() {
  if (!wavesurfer.value) return

  const requestId = ++waveformLoadRequestId
  waveformStatus.value = props.externalAudio ? 'Loading synced waveform...' : 'Loading waveform...'

  try {
    if (props.externalAudio && Number(props.trackId || 0) > 0) {
      const payload = await fetchWaveformPayload(Number(props.trackId))
      if (requestId !== waveformLoadRequestId || !wavesurfer.value) return

      waveformDuration.value = payload.duration
      await wavesurfer.value.load('', [payload.peaks], payload.duration)
      applyZoomPercent(zoomPercent.value)
      waveformStatus.value = 'Synced with the main player'
      return
    }

    waveformDuration.value = props.duration || 0
    await wavesurfer.value.load(props.audioUrl)
    if (requestId !== waveformLoadRequestId) return
    applyZoomPercent(zoomPercent.value)
    waveformStatus.value = 'Ready'
  } catch (error) {
    if (requestId !== waveformLoadRequestId) return
    waveformStatus.value = error instanceof Error ? error.message : 'Failed to load waveform'
  }
}

function renderFragments() {
  if (!regions.value) return

  for (const region of regions.value.getRegions()) {
    if (region.id.startsWith('fragment-')) {
      region.remove()
    }
  }

  for (const fragment of props.fragments) {
    const region = regions.value.addRegion({
      id: `fragment-${fragment.id}`,
      start: fragment.start_time,
      end: fragment.end_time,
      color: getFragmentColor(fragment),
      drag: false,
      resize: false,
    })

    if (region.element) {
      region.element.addEventListener('mouseenter', (event) => onFragmentHover(fragment, event))
      region.element.addEventListener('mouseleave', onFragmentLeave)
    }
  }
}

function clearSelection() {
  if (regions.value) {
    for (const region of regions.value.getRegions()) {
      if (!region.id.startsWith('fragment-')) {
        region.remove()
      }
    }
  }

  selectionStart.value = null
  selectionEnd.value = null
  showRatingButtons.value = false
}

function rateSelection(isLiked: boolean) {
  if (!hasSelection.value) return

  emit('fragment-created', {
    start_time: Math.min(selectionStart.value!, selectionEnd.value!),
    end_time: Math.max(selectionStart.value!, selectionEnd.value!),
    is_liked: isLiked,
  })

  clearSelection()
}

function onFragmentHover(fragment: Fragment, event: MouseEvent) {
  hoveredFragment.value = fragment
  tooltipPosition.value = {
    x: event.clientX,
    y: event.clientY - 10,
  }
}

function onFragmentLeave() {
  hoveredFragment.value = null
}

function updateVolume(nextVolume: number) {
  volume.value = nextVolume
  localStorage.setItem('player_volume', nextVolume.toString())

  if (!props.externalAudio) {
    wavesurfer.value?.setVolume(nextVolume)
  }
}

function updateZoom(nextZoom: number) {
  applyZoomPercent(nextZoom)
}

function applyZoomPercent(nextZoomPercent: number) {
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

function play() {
  if (!isSyncingFromExternal.value) {
    emit('play-requested')
  }

  if (!props.externalAudio) {
    wavesurfer.value?.play()
  }

  isPlaying.value = true
}

function pause() {
  if (!isSyncingFromExternal.value) {
    emit('pause-requested')
  }

  if (!props.externalAudio) {
    wavesurfer.value?.pause()
  }

  isPlaying.value = false
}

function stop() {
  if (props.externalAudio) {
    emit('pause-requested')
    emit('seek', 0)
    currentTime.value = 0
    emit('time-update', 0)
    isPlaying.value = false
    return
  }

  wavesurfer.value?.stop()
  currentTime.value = 0
  emit('time-update', 0)
  isPlaying.value = false
}

function togglePlay() {
  if (isPlaying.value) {
    pause()
    return
  }

  play()
}

function seekTo(time: number) {
  const duration = effectiveDuration.value || 0
  const normalizedTime = Math.max(0, Math.min(time, duration > 0 ? duration : time))
  currentTime.value = normalizedTime
  emit('time-update', normalizedTime)

  if (duration > 0 && wavesurfer.value) {
    wavesurfer.value.seekTo(normalizedTime / duration)
  }

  if (props.externalAudio && !isSyncingFromExternal.value) {
    emit('seek', normalizedTime)
  }
}

function syncPlay() {
  isSyncingFromExternal.value = true
  if (!props.externalAudio) {
    wavesurfer.value?.play()
  }
  isPlaying.value = true
  window.setTimeout(() => {
    isSyncingFromExternal.value = false
  }, 80)
}

function syncPause() {
  isSyncingFromExternal.value = true
  if (!props.externalAudio) {
    wavesurfer.value?.pause()
  }
  isPlaying.value = false
  window.setTimeout(() => {
    isSyncingFromExternal.value = false
  }, 80)
}

function syncSeekTo(time: number) {
  isSyncingFromExternal.value = true
  seekTo(time)
  window.setTimeout(() => {
    isSyncingFromExternal.value = false
  }, 80)
}

function initWaveSurfer() {
  if (!waveformRef.value) return

  regions.value = RegionsPlugin.create()
  wavesurfer.value = WaveSurfer.create({
    container: waveformRef.value,
    waveColor: '#64748b',
    progressColor: '#38bdf8',
    cursorColor: '#f8fafc',
    barWidth: 2,
    barRadius: 3,
    cursorWidth: 2,
    height: 128,
    barGap: 2,
    normalize: true,
    plugins: [regions.value],
  })

  wavesurfer.value.on('ready', () => {
    waveformStatus.value = props.externalAudio ? 'Synced with the main player' : 'Ready'
    renderFragments()
    if (!props.externalAudio) {
      wavesurfer.value?.setVolume(volume.value)
    }
  })

  wavesurfer.value.on('play', () => {
    if (!props.externalAudio) {
      isPlaying.value = true
    }
  })

  wavesurfer.value.on('pause', () => {
    if (!props.externalAudio) {
      isPlaying.value = false
    }
  })

  wavesurfer.value.on('timeupdate', (time) => {
    if (props.externalAudio) return
    currentTime.value = time
    emit('time-update', time)
  })

  wavesurfer.value.on('click', (relativeX) => {
    const duration = effectiveDuration.value || wavesurfer.value?.getDuration() || 0
    const time = relativeX * duration
    seekTo(time)
  })

  regions.value.enableDragSelection({
    color: 'rgba(56, 189, 248, 0.22)',
  })

  regions.value.on('region-created', (region) => {
    if (region.id.startsWith('fragment-')) {
      return
    }

    selectionStart.value = region.start
    selectionEnd.value = region.end
    showRatingButtons.value = true
    region.setOptions({ color: 'rgba(56, 189, 248, 0.3)' })
  })

  regions.value.on('region-clicked', (region, event) => {
    event.stopPropagation()
    const fragmentId = Number(region.id.replace('fragment-', ''))
    const fragment = props.fragments.find((candidate) => candidate.id === fragmentId)
    if (!fragment) {
      return
    }

    emit('fragment-clicked', fragment)
    if (props.externalAudio) {
      seekTo(fragment.start_time)
      emit('play-requested')
      return
    }

    region.play()
  })

  void loadWaveformSource()
}

watch(() => props.fragments, () => {
  renderFragments()
}, { deep: true })

watch(() => props.duration, (nextDuration) => {
  if (!props.externalAudio && nextDuration > 0) {
    waveformDuration.value = nextDuration
  }
})

watch(
  () => [props.audioUrl, props.trackId, props.externalAudio] as const,
  () => {
    clearSelection()
    void loadWaveformSource()
  },
)

watch(() => props.externalAudio, (external) => {
  if (!wavesurfer.value) return

  if (external) {
    wavesurfer.value.pause()
    isPlaying.value = false
    return
  }

  wavesurfer.value.setVolume(volume.value)
})

onMounted(() => {
  initWaveSurfer()
})

onUnmounted(() => {
  try {
    wavesurfer.value?.destroy()
  } catch {
    // Ignore teardown races while the component unmounts during waveform loading.
  }
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
  <div class="relative min-w-0 w-full max-w-full overflow-hidden rounded-[1.2rem] border border-white/10 bg-slate-900/95 p-4 shadow-[0_18px_45px_rgba(15,23,42,0.22)]">
    <div class="flex flex-col gap-3 border-b border-white/10 pb-3 md:flex-row md:items-center md:justify-between">
      <div>
        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Waveform Review</div>
        <div class="mt-1 text-sm text-slate-300">{{ waveformStatus }}</div>
      </div>

      <div class="flex flex-wrap items-center gap-2">
        <span
          class="rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em]"
          :class="externalAudio ? 'border-sky-300/30 bg-sky-400/10 text-sky-200' : 'border-slate-600 bg-slate-800 text-slate-300'"
        >
          {{ externalAudio ? 'Synced with player' : 'Local playback' }}
        </span>
        <span class="rounded-full border border-slate-600 bg-slate-800 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300">
          {{ fragments.length }} fragment{{ fragments.length === 1 ? '' : 's' }}
        </span>
      </div>
    </div>

    <div ref="waveformRef" class="waveform-scroll-shell mt-4 min-w-0 max-w-full overflow-x-auto overflow-y-hidden rounded-[1rem] border border-white/10 bg-slate-950/80"></div>

    <Transition name="fade">
      <div
        v-if="showRatingButtons && hasSelection"
        class="absolute right-4 top-4 flex flex-wrap items-center gap-2 rounded-[1rem] border border-sky-300/30 bg-slate-950/95 px-3 py-2 shadow-xl"
      >
        <span class="text-xs font-semibold uppercase tracking-[0.14em] text-slate-300">
          {{ formatTime(selectionDuration) }} selected
        </span>
        <button
          type="button"
          class="rounded-full border border-emerald-300/30 bg-emerald-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-emerald-200 transition hover:border-emerald-200 hover:bg-emerald-500/20"
          @click="rateSelection(true)"
        >
          Keep
        </button>
        <button
          type="button"
          class="rounded-full border border-rose-300/30 bg-rose-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-rose-200 transition hover:border-rose-200 hover:bg-rose-500/20"
          @click="rateSelection(false)"
        >
          Reject
        </button>
        <button
          type="button"
          class="rounded-full border border-slate-600 bg-slate-800 px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-slate-300 transition hover:border-slate-500 hover:text-white"
          @click="clearSelection"
        >
          Cancel
        </button>
      </div>
    </Transition>

    <Teleport to="body">
      <Transition name="fade">
        <div
          v-if="hoveredFragment"
          class="pointer-events-none fixed z-50 max-w-sm rounded-[1rem] border border-slate-700 bg-slate-950 px-4 py-3 text-sm text-white shadow-2xl"
          :style="{
            left: `${tooltipPosition.x}px`,
            top: `${tooltipPosition.y}px`,
            transform: 'translate(-50%, -100%)',
          }"
        >
          <div class="flex items-center gap-2">
            <span :class="hoveredFragment.is_liked ? 'text-emerald-300' : 'text-rose-300'" class="text-lg">
              {{ hoveredFragment.is_liked ? '↑' : '↓' }}
            </span>
            <span class="font-semibold text-slate-100">
              {{ formatTime(hoveredFragment.start_time) }} - {{ formatTime(hoveredFragment.end_time) }}
            </span>
          </div>
          <p v-if="hoveredFragment.comment" class="mt-2 text-slate-300">
            {{ hoveredFragment.comment }}
          </p>
          <p v-else-if="hoveredFragment.name" class="mt-2 text-slate-400">
            {{ hoveredFragment.name }}
          </p>
        </div>
      </Transition>
    </Teleport>

    <div class="mt-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
      <div class="flex flex-wrap items-center gap-3">
        <button
          type="button"
          class="inline-flex h-11 w-11 items-center justify-center rounded-full bg-sky-500 text-white transition hover:bg-sky-400"
          @click="togglePlay"
        >
          {{ isPlaying ? 'II' : '>' }}
        </button>
        <button
          type="button"
          class="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-600 bg-slate-800 text-slate-200 transition hover:border-slate-500 hover:text-white"
          @click="stop"
        >
          []
        </button>
        <div class="rounded-full border border-slate-600 bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200">
          {{ formatTime(currentTime) }} / {{ formatTime(effectiveDuration || 0) }}
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-4">
        <label v-if="!externalAudio" class="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
          <span>Volume</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="volume"
            class="h-1.5 w-24 cursor-pointer appearance-none rounded bg-slate-700"
            @input="updateVolume(parseFloat(($event.target as HTMLInputElement).value))"
          >
        </label>
        <label class="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
          <span>Zoom {{ zoomPercent }}%</span>
          <input
            type="range"
            min="100"
            max="800"
            step="10"
            :value="zoomPercent"
            class="h-1.5 w-24 cursor-pointer appearance-none rounded bg-slate-700"
            @input="updateZoom(parseInt(($event.target as HTMLInputElement).value))"
          >
        </label>
      </div>
    </div>

    <div class="mt-4 flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-400">
      <span class="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800 px-3 py-1">
        <span class="h-2 w-2 rounded-full bg-emerald-400/80"></span>
        Positive fragment
      </span>
      <span class="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800 px-3 py-1">
        <span class="h-2 w-2 rounded-full bg-rose-400/80"></span>
        Negative fragment
      </span>
      <span class="inline-flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800 px-3 py-1">
        <span class="h-2 w-2 rounded-full bg-sky-400/80"></span>
        Drag to mark a section
      </span>
    </div>

    <p class="mt-3 text-xs leading-6 text-slate-400">
      Drag to mark a section and rate it. Click a saved fragment to jump playback to that exact point.
    </p>
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

.waveform-scroll-shell {
  overscroll-behavior-x: contain;
  scrollbar-width: thin;
}

input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 12px;
  height: 12px;
  background: #38bdf8;
  border-radius: 9999px;
  cursor: pointer;
}

input[type="range"]::-moz-range-thumb {
  width: 12px;
  height: 12px;
  background: #38bdf8;
  border-radius: 9999px;
  cursor: pointer;
  border: none;
}
</style>
