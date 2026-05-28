<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js'
import type { StemTuningFocusWindow } from '@/types'

interface Props {
    sourceUrl: string
    waveformUrl: string
    sourceLabel: string
    monitorVolume?: number
    duration?: number
    focusWindows?: StemTuningFocusWindow[]
    loopWindow?: {
        enabled: boolean
        startSec: number
        endSec: number
    }
}

interface Emits {
    (e: 'media-ready', media: HTMLAudioElement): void
    (e: 'time-update', time: number): void
    (e: 'analysis-frame', payload: { waveform: number[]; spectrum: number[] }): void
}

interface WaveformPayload {
    duration: number
    peaks: number[]
}

const props = withDefaults(defineProps<Props>(), {
    monitorVolume: 1,
    duration: 0,
    focusWindows: () => [],
    loopWindow: () => ({
        enabled: false,
        startSec: 0,
        endSec: 0,
    }),
})

const emit = defineEmits<Emits>()

const waveformRef = ref<HTMLDivElement | null>(null)
const wavesurfer = ref<WaveSurfer | null>(null)
const regions = ref<ReturnType<typeof RegionsPlugin.create> | null>(null)
const currentTime = ref(0)
const isPlaying = ref(false)
const zoomPercent = ref(100)
const status = ref('Loading waveform...')
const resolvedDuration = ref(props.duration || 0)
const loopWindowArmed = ref(false)
const MIN_ZOOM_PERCENT = 100
const MAX_ZOOM_PERCENT = 800
const ANALYSIS_WAVEFORM_POINTS = 96
const ANALYSIS_SPECTRUM_BINS = 32
let lastRequestedSeekTime: number | null = null
let lastPlayRequestAt = 0
let lastPlayRequestTime = -1
let analysisFrameHandle: number | null = null
let audioContext: AudioContext | null = null
let mediaSourceNode: MediaElementAudioSourceNode | null = null
let analyserNode: AnalyserNode | null = null
let gainNode: GainNode | null = null
let mediaElementEventCleanups: Array<() => void> = []
let audioGraphOperation: Promise<void> = Promise.resolve()

const waveformPayloadCache = ((globalThis as typeof globalThis & {
    __aimStemTuningWaveformCache?: Map<string, WaveformPayload>
}).__aimStemTuningWaveformCache ??= new Map<string, WaveformPayload>())

const waveformPayloadRequests = ((globalThis as typeof globalThis & {
    __aimStemTuningWaveformRequests?: Map<string, Promise<WaveformPayload>>
}).__aimStemTuningWaveformRequests ??= new Map<string, Promise<WaveformPayload>>())

let mediaElement = createMediaElement()

const safeDuration = computed(() => resolvedDuration.value > 0 ? resolvedDuration.value : props.duration)

function formatTime(seconds: number): string {
    const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
    const mins = Math.floor(safe / 60)
    const secs = Math.floor(safe % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
}

function createMediaElement(): HTMLAudioElement {
    const nextMediaElement = new Audio()
    nextMediaElement.preload = 'auto'
    nextMediaElement.crossOrigin = 'anonymous'
    nextMediaElement.volume = clamp(props.monitorVolume, 0, 1)
    return nextMediaElement
}

function resolveAbsoluteUrl(url: string): string {
    return new URL(url, window.location.origin).toString()
}

function clearMediaElementEventListeners() {
    for (const cleanup of mediaElementEventCleanups) {
        cleanup()
    }
    mediaElementEventCleanups = []
}

function addMediaElementListener<K extends keyof HTMLMediaElementEventMap>(
    target: HTMLAudioElement,
    type: K,
    listener: (event: HTMLMediaElementEventMap[K]) => void,
) {
    target.addEventListener(type, listener as EventListener)
    mediaElementEventCleanups.push(() => {
        target.removeEventListener(type, listener as EventListener)
    })
}

function waveformDebug(message: string, context: Record<string, unknown> = {}): void {
    console.info('[TuningWaveform:v2-transport]', message, {
        sourceLabel: props.sourceLabel,
        sourceUrl: props.sourceUrl,
        waveformUrl: props.waveformUrl,
        mediaCurrentTime: Number(mediaElement.currentTime.toFixed(3)),
        componentCurrentTime: Number(currentTime.value.toFixed(3)),
        duration: Number((safeDuration.value || wavesurfer.value?.getDuration() || 0).toFixed(3)),
        isPlaying: !mediaElement.paused,
        ...context,
    })
}

function emitSilentAnalysisFrame(): void {
    emit('analysis-frame', {
        waveform: Array.from({ length: ANALYSIS_WAVEFORM_POINTS }, () => 0),
        spectrum: Array.from({ length: ANALYSIS_SPECTRUM_BINS }, () => 0),
    })
}

function downsampleWaveform(values: Uint8Array): number[] {
    const points: number[] = []
    const stride = Math.max(1, Math.floor(values.length / ANALYSIS_WAVEFORM_POINTS))
    for (let index = 0; index < values.length && points.length < ANALYSIS_WAVEFORM_POINTS; index += stride) {
        points.push(((values[index] / 255) * 2) - 1)
    }
    while (points.length < ANALYSIS_WAVEFORM_POINTS) {
        points.push(0)
    }
    return points
}

function downsampleSpectrum(values: Uint8Array): number[] {
    const bins: number[] = []
    const stride = Math.max(1, Math.floor(values.length / ANALYSIS_SPECTRUM_BINS))
    for (let bin = 0; bin < ANALYSIS_SPECTRUM_BINS; bin += 1) {
        const start = bin * stride
        if (start >= values.length) {
            bins.push(0)
            continue
        }
        const end = Math.min(values.length, start + stride)
        let sum = 0
        for (let index = start; index < end; index += 1) {
            sum += values[index]
        }
        bins.push((sum / Math.max(1, end - start)) / 255)
    }
    return bins
}

function startAnalysisLoop() {
    if (analysisFrameHandle !== null) {
        cancelAnimationFrame(analysisFrameHandle)
    }

    const tick = () => {
        if (!analyserNode) {
            emitSilentAnalysisFrame()
            analysisFrameHandle = requestAnimationFrame(tick)
            return
        }

        const timeDomain = new Uint8Array(analyserNode.fftSize)
        const spectrum = new Uint8Array(analyserNode.frequencyBinCount)
        analyserNode.getByteTimeDomainData(timeDomain)
        analyserNode.getByteFrequencyData(spectrum)

        emit('analysis-frame', {
            waveform: downsampleWaveform(timeDomain),
            spectrum: downsampleSpectrum(spectrum),
        })
        analysisFrameHandle = requestAnimationFrame(tick)
    }

    analysisFrameHandle = requestAnimationFrame(tick)
}

async function ensureAudioGraph() {
    const nextOperation = audioGraphOperation.catch(() => undefined).then(async () => {
        if (audioContext === null) {
            const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
            if (!AudioContextCtor) {
                return
            }

            audioContext = new AudioContextCtor()
            analyserNode = audioContext.createAnalyser()
            analyserNode.fftSize = 2048
            analyserNode.smoothingTimeConstant = 0.72
            gainNode = audioContext.createGain()
            gainNode.gain.value = clamp(props.monitorVolume, 0, 1)
            analyserNode.connect(gainNode)
            gainNode.connect(audioContext.destination)
            startAnalysisLoop()
        }

        if (audioContext.state === 'suspended') {
            await audioContext.resume().catch(() => undefined)
        }

        if (mediaSourceNode === null) {
            mediaSourceNode = audioContext.createMediaElementSource(mediaElement)
            mediaSourceNode.connect(analyserNode!)
        }

        mediaElement.volume = 1
    })

    audioGraphOperation = nextOperation
    await nextOperation
}

function fitMinPxPerSec(): number {
    const duration = safeDuration.value || wavesurfer.value?.getDuration() || 0
    const width = waveformRef.value?.clientWidth ?? wavesurfer.value?.getWrapper().clientWidth ?? 0
    if (duration <= 0 || width <= 0) {
        return 0
    }

    return width / duration
}

async function fetchWaveformPayload(url: string): Promise<WaveformPayload> {
    const cached = waveformPayloadCache.get(url)
    if (cached) {
        return cached
    }

    const pending = waveformPayloadRequests.get(url)
    if (pending) {
        return pending
    }

    const request = fetch(url, {
        headers: {
            Accept: 'application/json',
        },
    }).then(async (response) => {
        if (!response.ok) {
            throw new Error(`Waveform request failed (${response.status})`)
        }

        const raw = await response.json() as { duration?: unknown; peaks?: unknown }
        const duration = Number(raw.duration || 0)
        const peaks = Array.isArray(raw.peaks)
            ? raw.peaks.map((value) => Number(value)).filter((value) => Number.isFinite(value))
            : []

        if (duration <= 0 || peaks.length === 0) {
            throw new Error('Waveform payload is invalid')
        }

        const payload = { duration, peaks }
        waveformPayloadCache.set(url, payload)
        return payload
    }).finally(() => {
        waveformPayloadRequests.delete(url)
    })

    waveformPayloadRequests.set(url, request)
    return request
}

function regionColor(kind: string): string {
    switch (kind) {
        case 'problem':
            return 'rgba(248, 113, 113, 0.18)'
        case 'audit':
            return 'rgba(56, 189, 248, 0.18)'
        default:
            return 'rgba(148, 163, 184, 0.18)'
    }
}

function renderFocusWindows() {
    if (!regions.value) {
        return
    }

    for (const region of regions.value.getRegions()) {
        if (region.id.startsWith('focus-')) {
            region.remove()
        }
    }

    for (const windowDef of props.focusWindows) {
        regions.value.addRegion({
            id: `focus-${windowDef.id}`,
            start: windowDef.start_sec,
            end: windowDef.end_sec,
            drag: false,
            resize: false,
            color: regionColor(windowDef.kind),
        })
    }
}

let loadRequestId = 0

function attachMediaElementListeners(target: HTMLAudioElement) {
    clearMediaElementEventListeners()

    addMediaElementListener(target, 'timeupdate', () => {
        if (target !== mediaElement) {
            return
        }
        currentTime.value = target.currentTime
        emit('time-update', target.currentTime)
        enforceLoopWindow()
    })

    addMediaElementListener(target, 'seeking', () => {
        if (target !== mediaElement) {
            return
        }
        waveformDebug('media:seeking', {
            lastRequestedSeekTime: lastRequestedSeekTime !== null ? Number(lastRequestedSeekTime.toFixed(3)) : null,
        })
    })

    addMediaElementListener(target, 'seeked', () => {
        if (target !== mediaElement) {
            return
        }
        waveformDebug('media:seeked', {
            lastRequestedSeekTime: lastRequestedSeekTime !== null ? Number(lastRequestedSeekTime.toFixed(3)) : null,
            finalCurrentTime: Number(target.currentTime.toFixed(3)),
        })
        lastRequestedSeekTime = null
    })

    addMediaElementListener(target, 'play', () => {
        if (target !== mediaElement) {
            return
        }
        isPlaying.value = true
        waveformDebug('media:play')
    })

    addMediaElementListener(target, 'pause', () => {
        if (target !== mediaElement) {
            return
        }
        isPlaying.value = false
        waveformDebug('media:pause')
    })
}

async function loadSource() {
    if (!wavesurfer.value) {
        return
    }

    const requestId = ++loadRequestId
    const absoluteSourceUrl = resolveAbsoluteUrl(props.sourceUrl)
    const currentAbsoluteSourceUrl = mediaElement.currentSrc || mediaElement.src
    const sameSource = currentAbsoluteSourceUrl === absoluteSourceUrl
    const resumeTime = Number.isFinite(mediaElement.currentTime) && mediaElement.currentTime > 0
        ? mediaElement.currentTime
        : currentTime.value
    const resumePlayback = !mediaElement.paused && !mediaElement.ended
    const loadStartedAt = performance.now()
    status.value = 'Preparing source...'
    waveformDebug('loadSource:start', {
        requestId,
        sameSource,
        resumeTime: Number(resumeTime.toFixed(3)),
        resumePlayback,
    })

    try {
        const payload = await fetchWaveformPayload(props.waveformUrl)
        if (requestId !== loadRequestId || !wavesurfer.value) {
            waveformDebug('loadSource:aborted-stale-request', { requestId })
            return
        }

        const elapsedWhilePreparing = resumePlayback ? (performance.now() - loadStartedAt) / 1000 : 0
        const nextTime = Math.min(resumeTime + elapsedWhilePreparing, Math.max(0, payload.duration - 0.05))

        mediaElement.pause()
        resolvedDuration.value = payload.duration
        await wavesurfer.value.load(absoluteSourceUrl, [payload.peaks], payload.duration)
        if (requestId !== loadRequestId) {
            waveformDebug('loadSource:post-load-stale-request', { requestId })
            return
        }

        renderFocusWindows()
        zoomPercent.value = 100
        applyZoomPercent(100)
        if (nextTime > 0) {
            seekTo(nextTime)
        } else {
            currentTime.value = 0
            emit('time-update', 0)
        }

        if (resumePlayback) {
            await mediaElement.play().catch(() => undefined)
        }

        status.value = 'Ready'
        waveformDebug('loadSource:ready', {
            requestId,
            payloadDuration: Number(payload.duration.toFixed(3)),
            resumedTo: Number((nextTime > 0 ? nextTime : 0).toFixed(3)),
            resumedPlayback: resumePlayback,
        })
    } catch (error) {
        status.value = error instanceof Error ? error.message : 'Failed to load source'
        waveformDebug('loadSource:error', {
            requestId,
            error: error instanceof Error ? error.message : String(error),
        })
    }
}

function seekTo(time: number) {
    const duration = safeDuration.value || wavesurfer.value?.getDuration() || mediaElement.duration || 0
    const nextTime = Math.max(0, Math.min(time, duration > 0 ? duration : time))
    lastRequestedSeekTime = nextTime
    waveformDebug('seekTo', {
        requestedTime: Number(time.toFixed(3)),
        nextTime: Number(nextTime.toFixed(3)),
        wavesurferTimeBefore: Number((wavesurfer.value?.getCurrentTime() || 0).toFixed(3)),
    })
    if (wavesurfer.value) {
        wavesurfer.value.setTime(nextTime)
    } else {
        mediaElement.currentTime = nextTime
    }
    currentTime.value = nextTime
    loopWindowArmed.value = nextTime >= props.loopWindow.startSec && nextTime < props.loopWindow.endSec
    emit('time-update', nextTime)
}

async function playFrom(time: number) {
    const now = performance.now()
    if (Math.abs(time - lastPlayRequestTime) < 0.12 && now - lastPlayRequestAt < 220) {
        waveformDebug('playFrom:deduped', {
            requestedTime: Number(time.toFixed(3)),
            deltaMs: Number((now - lastPlayRequestAt).toFixed(1)),
        })
        return
    }

    lastPlayRequestAt = now
    lastPlayRequestTime = time
    waveformDebug('playFrom:start', {
        requestedTime: Number(time.toFixed(3)),
    })
    await ensureAudioGraph()
    seekTo(time)
    await mediaElement.play().catch(() => undefined)
    waveformDebug('playFrom:end', {
        requestedTime: Number(time.toFixed(3)),
        mediaCurrentTimeAfter: Number(mediaElement.currentTime.toFixed(3)),
        wavesurferTimeAfter: Number((wavesurfer.value?.getCurrentTime() || 0).toFixed(3)),
    })
}

function resolvePointerTime(event: MouseEvent): number {
    const duration = safeDuration.value || wavesurfer.value?.getDuration() || mediaElement.duration || 0
    if (duration <= 0) {
        return 0
    }

    const host = waveformRef.value
    const wrapper = wavesurfer.value?.getWrapper()
    const rect = wrapper?.getBoundingClientRect() ?? host?.getBoundingClientRect()
    if (!rect) {
        return 0
    }

    const totalWidth = Math.max(wrapper?.scrollWidth ?? 0, rect.width ?? 0, 1)
    const scrollLeft = wavesurfer.value?.getScroll() ?? wrapper?.scrollLeft ?? 0
    const localX = clamp((event.clientX - rect.left) + scrollLeft, 0, totalWidth)
    return (localX / totalWidth) * duration
}

function handleWaveformPointer(event: MouseEvent) {
    if (event.button !== 0) {
        return
    }

    event.preventDefault()
    event.stopPropagation()
    const resolvedTime = resolvePointerTime(event)
    waveformDebug('pointerClick', {
        clientX: event.clientX,
        resolvedTime: Number(resolvedTime.toFixed(3)),
    })
    void playFrom(resolvedTime)
}

async function playPause() {
    await ensureAudioGraph()
    if (mediaElement.paused) {
        await mediaElement.play().catch(() => undefined)
        return
    }

    mediaElement.pause()
}

function stop() {
    mediaElement.pause()
    seekTo(0)
}

function applyZoomPercent(nextZoomPercent: number) {
    const normalizedPercent = clamp(nextZoomPercent, MIN_ZOOM_PERCENT, MAX_ZOOM_PERCENT)
    zoomPercent.value = normalizedPercent

    if (!wavesurfer.value) {
        return
    }

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

function updateZoom(nextZoomPercent: number) {
    applyZoomPercent(nextZoomPercent)
}

function enforceLoopWindow() {
    if (!props.loopWindow.enabled) {
        loopWindowArmed.value = false
        return
    }

    if (props.loopWindow.endSec <= props.loopWindow.startSec) {
        loopWindowArmed.value = false
        return
    }

    if (!loopWindowArmed.value) {
        if (mediaElement.currentTime >= props.loopWindow.startSec && mediaElement.currentTime < props.loopWindow.endSec) {
            loopWindowArmed.value = true
        }
        return
    }

    if (mediaElement.currentTime >= props.loopWindow.endSec) {
        mediaElement.currentTime = props.loopWindow.startSec
    }
}

function initWaveSurfer() {
    if (!waveformRef.value) {
        return
    }

    regions.value = RegionsPlugin.create()
    wavesurfer.value = WaveSurfer.create({
        container: waveformRef.value,
        media: mediaElement,
        interact: false,
        waveColor: '#64748b',
        progressColor: '#38bdf8',
        cursorColor: '#f8fafc',
        barWidth: 2,
        barGap: 2,
        barRadius: 3,
        cursorWidth: 2,
        height: 140,
        normalize: true,
        plugins: regions.value ? [regions.value as any] : [],
    })

    regions.value.on('region-clicked', (region, event) => {
        event.preventDefault()
        event.stopPropagation()
        const resolvedTime = resolvePointerTime(event)
        waveformDebug('regionClick', {
            regionId: region.id,
            regionStart: Number(region.start.toFixed(3)),
            regionEnd: Number(region.end.toFixed(3)),
            resolvedTime: Number(resolvedTime.toFixed(3)),
        })
        void playFrom(resolvedTime)
    })
    attachMediaElementListeners(mediaElement)
    emit('media-ready', mediaElement)
    waveformDebug('initWaveSurfer:media-ready-emitted')
    void loadSource()
}

watch(
    () => [props.sourceUrl, props.waveformUrl] as const,
    ([nextSourceUrl, nextWaveformUrl], [prevSourceUrl, prevWaveformUrl]) => {
        waveformDebug('watch:source-changed', {
            prevSourceUrl,
            nextSourceUrl,
            prevWaveformUrl,
            nextWaveformUrl,
        })
        void loadSource()
    },
)

watch(() => props.monitorVolume, (nextVolume) => {
    if (gainNode !== null) {
        gainNode.gain.value = clamp(nextVolume, 0, 1)
        mediaElement.volume = 1
        return
    }
    mediaElement.volume = clamp(nextVolume, 0, 1)
})

watch(() => props.focusWindows, () => {
    renderFocusWindows()
}, { deep: true })

watch(
    () => [props.loopWindow.enabled, props.loopWindow.startSec, props.loopWindow.endSec] as const,
    ([enabled, startSec, endSec]) => {
        if (!enabled || endSec <= startSec) {
            loopWindowArmed.value = false
            return
        }

        loopWindowArmed.value = mediaElement.currentTime >= startSec && mediaElement.currentTime < endSec
    },
)

onMounted(() => {
    initWaveSurfer()
    emitSilentAnalysisFrame()
})

onUnmounted(() => {
    clearMediaElementEventListeners()
    mediaElement.pause()
    mediaElement.src = ''
    if (analysisFrameHandle !== null) {
        cancelAnimationFrame(analysisFrameHandle)
        analysisFrameHandle = null
    }
    if (audioContext !== null) {
        void audioContext.close().catch(() => undefined)
    }
    if (mediaSourceNode !== null) {
        mediaSourceNode.disconnect()
        mediaSourceNode = null
    }
    if (analyserNode !== null) {
        analyserNode.disconnect()
        analyserNode = null
    }
    if (gainNode !== null) {
        gainNode.disconnect()
        gainNode = null
    }
    audioContext = null
    emitSilentAnalysisFrame()
    wavesurfer.value?.destroy()
})

defineExpose({
    seekTo,
    playFrom,
    play: () => mediaElement.play(),
    pause: () => mediaElement.pause(),
    get mediaElement() {
        return mediaElement
    },
})
</script>

<template>
    <div class="waveform-container min-w-0">
        <div class="flex flex-col gap-3 border-b border-white/10 pb-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
                <div class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">Tuning Preview</div>
                <div class="mt-1 text-sm text-slate-200">{{ sourceLabel }}</div>
                <div class="mt-1 text-xs text-slate-400">{{ status }}</div>
            </div>

            <div class="flex flex-wrap items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300">
                <span class="rounded-full border border-sky-300/20 bg-sky-400/10 px-3 py-1">
                    {{ formatTime(currentTime) }} / {{ formatTime(safeDuration || 0) }}
                </span>
            </div>
        </div>

        <div
            ref="waveformRef"
            class="mt-4 cursor-pointer overflow-hidden rounded-[1rem] border border-white/10 bg-slate-950/80"
            @click="handleWaveformPointer"
        ></div>

        <div class="mt-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div class="flex flex-wrap items-center gap-3">
                <button type="button" class="action-primary !rounded-full !px-4 !py-3" @click="playPause">
                    {{ isPlaying ? 'Pause' : 'Play' }}
                </button>
                <button type="button" class="action-secondary !rounded-full !px-4 !py-3" @click="stop">
                    Stop
                </button>
            </div>

            <label class="flex items-center gap-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                <span>Zoom {{ zoomPercent }}%</span>
                <input
                    type="range"
                    min="100"
                    max="800"
                    step="10"
                    :value="zoomPercent"
                    class="h-1.5 w-28 cursor-pointer appearance-none rounded bg-slate-700"
                    @input="updateZoom(parseInt(($event.target as HTMLInputElement).value))"
                >
            </label>
        </div>
    </div>
</template>

<style scoped>
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 9999px;
    background: #38bdf8;
    cursor: pointer;
}

input[type="range"]::-moz-range-thumb {
    width: 12px;
    height: 12px;
    border-radius: 9999px;
    background: #38bdf8;
    cursor: pointer;
    border: none;
}
</style>
