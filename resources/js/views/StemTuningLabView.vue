<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { toast } from 'vue3-toastify'
import api from '@/api/client'
import StemTuningLiveMonitor from '@/components/StemTuningLiveMonitor.vue'
import StemTuningWaveform from '@/components/StemTuningWaveformTransport.vue'
import { AUTH_TOKEN_STORAGE_KEY } from '@/constants/auth'
import type {
    StemTuningDetectorSegments,
    StemTuningFocusWindow,
    StemTuningLabSession,
    StemTuningHybridBuildConfig,
    StemTuningLayerState,
    StemTuningProcessorConfig,
    StemTuningPreset,
    StemTuningRenderMode,
    StemTuningRenderTask,
    StemTuningStemRewriteConfig,
    StemTuningSource,
    StemTuningUnsafeBsRoformerInternals,
    StemTuningUnsafeHybridInternals,
    StemTuningUnsafeModelInternals,
    StemTuningVocalsRewriteConfig,
} from '@/types'

type VocalsRewriteCleanupMode = 'none' | 'subtractive' | 'hard_mask'
type VocalsRewriteTaskMode = StemTuningVocalsRewriteConfig['taskMode']
type VocalsRewriteAudiblePreset = 'tight_clean' | 'hard_isolate' | 'artifact_killer' | 'airy_preserve'
type UnsafeFieldKind = 'text' | 'number' | 'checkbox' | 'textarea' | 'string_array'
type JsonDraftStatusTone = 'muted' | 'pending' | 'success' | 'error'

interface StemTuningWaveformExpose {
    seekTo: (time: number) => void
    playFrom: (time: number) => Promise<void>
    mediaElement: HTMLAudioElement
}

interface DetectorReviewCard {
    segmentId: string
    label: string
    startSec: number
    endSec: number
    windowStartSec: number
    windowEndSec: number
    structureLabel: string
    kinds: string[]
    segmentPriority: number
    windowScore: number
    selected: boolean
    notes: string
}

interface UnsafeFieldDefinition {
    key: string
    label: string
    description: string
    kind: UnsafeFieldKind
    placeholder?: string
    min?: number
    max?: number
    step?: number
}

const route = useRoute()
const router = useRouter()

const session = ref<StemTuningLabSession | null>(null)
const loading = ref(true)
const errorMessage = ref('')
const canBuildStems = ref(false)
const selectedSourceKey = ref('')
const editorLayerKey = ref('')
const stemBrowserRef = ref<HTMLDivElement | null>(null)
const preset = ref<StemTuningPreset | null>(null)
const importBuffer = ref('')
const activeModeConfigDraft = ref('')
const unsafeModelInternalsDraft = ref('')
const importBufferApplyTimer = ref<number | null>(null)
const activeModeConfigDraftApplyTimer = ref<number | null>(null)
const unsafeModelInternalsDraftApplyTimer = ref<number | null>(null)
const importBufferStatusTone = ref<JsonDraftStatusTone>('muted')
const importBufferStatusMessage = ref('Paste any preset JSON. Valid JSON auto-applies after a short pause.')
const activeModeConfigDraftStatusTone = ref<JsonDraftStatusTone>('muted')
const activeModeConfigDraftStatusMessage = ref('Valid mode JSON auto-applies after a short pause.')
const unsafeModelInternalsDraftStatusTone = ref<JsonDraftStatusTone>('muted')
const unsafeModelInternalsDraftStatusMessage = ref('Valid unsafe JSON auto-applies after a short pause.')
const currentTime = ref(0)
const monitorVolume = ref(1)
const mediaElementRef = ref<HTMLAudioElement | null>(null)
const waveformTransportRef = ref<StemTuningWaveformExpose | null>(null)
const buildTask = ref<StemTuningRenderTask | null>(null)
const buildPollingTimer = ref<number | null>(null)
const buildSubmitting = ref(false)
const autoRenderEnabled = ref(true)
const autoRenderTimer = ref<number | null>(null)
const autoRenderQueued = ref(false)
const renderTask = ref<StemTuningRenderTask | null>(null)
const renderPollingTimer = ref<number | null>(null)
const renderSubmitting = ref(false)
const renderCancelRequestedTaskId = ref<number | null>(null)

let suppressPresetWatch = false
let suppressSelectedSourcePresetSync = false
let suppressSelectedSourceEditorSync = false
let lastCompletedRenderSignature = ''
const AUTO_RENDER_DELAY_MS = 1800
const RENDER_STATUS_POLL_MS = 750
const JSON_AUTO_APPLY_DELAY_MS = 650
const DEFAULT_ANALYSIS_WAVEFORM = Array.from({ length: 96 }, () => 0)
const DEFAULT_ANALYSIS_SPECTRUM = Array.from({ length: 32 }, () => 0)
const STEM_REWRITE_AFFECTED_BASE_KEYS = ['drums', 'other', 'guitar', 'piano'] as const
const STEM_REWRITE_PRESERVED_BASE_KEYS = ['vocals', 'bass'] as const

let suppressImportBufferWatch = false
let suppressActiveModeConfigDraftWatch = false
let suppressUnsafeModelInternalsDraftWatch = false

const trackId = computed(() => Number(route.params.id))
const track = computed(() => session.value?.track ?? null)
const lab = computed(() => session.value?.lab ?? null)
const sources = computed(() => lab.value?.sources ?? [])
const editableSources = computed(() => sources.value.filter((source) => source.editable))
const detectorSegments = computed<StemTuningDetectorSegments | null>(() => (
    lab.value?.detector_segments ?? null
))
const selectedSource = computed<StemTuningSource | null>(() => (
    sources.value.find((source) => source.key === selectedSourceKey.value) ?? null
))
const selectedSourceStreamUrl = computed(() => withApiToken(selectedSource.value?.stream_url ?? ''))
const selectedSourceWaveformUrl = computed(() => withApiToken(selectedSource.value?.waveform_url ?? ''))
const displayPresetJson = computed(() => (
    preset.value ? JSON.stringify(preset.value, null, 2) : ''
))
const currentLayerStates = computed<Record<string, StemTuningLayerState>>(() => preset.value?.layerStates ?? {})
const enabledLayerCount = computed(() => editableSources.value.filter((source) => isLayerEnabled(source.key)).length)
const layerCards = computed(() => editableSources.value.map((source) => ({
    ...source,
    enabled: isLayerEnabled(source.key),
    gainDb: layerGainDb(source.key),
    transientDb: layerProcessorValue(source.key, 'transientDb'),
    bodyDb: layerProcessorValue(source.key, 'bodyDb'),
    harmonicDuckDb: layerProcessorValue(source.key, 'harmonicDuckDb'),
    bleedDuckDb: layerProcessorValue(source.key, 'bleedDuckDb'),
    adjustmentCount: countLayerAdjustments(source.key),
    primary: source.key === selectedSourceKey.value,
})))
const editorLayerCard = computed(() => (
    layerCards.value.find((card) => card.key === editorLayerKey.value)
    ?? layerCards.value[0]
    ?? null
))
const monitorVolumePercent = computed(() => Math.round(monitorVolume.value * 100))
const buildBusy = computed(() => buildSubmitting.value || ['pending', 'processing'].includes(buildTask.value?.status ?? ''))
const renderBusy = computed(() => renderSubmitting.value || ['pending', 'processing'].includes(renderTask.value?.status ?? ''))
const autoRenderActive = computed(() => autoRenderEnabled.value && Boolean(session.value && preset.value))
const renderProgressPercent = computed(() => {
    const explicit = renderTask.value?.progress
    if (typeof explicit === 'number' && Number.isFinite(explicit)) {
        return clamp(Math.round(explicit), 0, 100)
    }
    if (renderTask.value?.status === 'completed') {
        return 100
    }
    return 0
})
const renderLogEntries = computed(() => renderTask.value?.logs ?? [])
const liveMonitorWaveform = ref<number[]>([...DEFAULT_ANALYSIS_WAVEFORM])
const liveMonitorSpectrum = ref<number[]>([...DEFAULT_ANALYSIS_SPECTRUM])
const detectorSelectedSegmentIds = computed(() => {
    const vocalDrop = detectorSegments.value?.selected?.vocal_drop ?? []
    const fallback = detectorSegments.value?.selected?.drop_like ?? []
    const selected = vocalDrop.length > 0 ? vocalDrop : fallback
    return new Set(selected.map((segment) => segment.segmentId))
})
const detectorReviewCards = computed<DetectorReviewCard[]>(() => {
    const payload = detectorSegments.value
    if (!payload) {
        return []
    }

    const selectedWindows = payload.selected.vocal_drop.length > 0
        ? payload.selected.vocal_drop
        : payload.selected.drop_like
    const selectedOrderById = new Map(
        selectedWindows.map((segment, index) => [segment.segmentId, index + 1]),
    )

    return payload.segments
        .filter((segment) => segment.kinds.includes('drop_like') || segment.kinds.includes('vocal_drop'))
        .sort((left, right) => (
            right.scores.segmentPriority - left.scores.segmentPriority
            || left.startSec - right.startSec
        ))
        .map((segment, index) => {
            const selected = detectorSelectedSegmentIds.value.has(segment.segmentId)
            const kinds = segment.kinds.filter((kind) => (
                kind === 'vocal_drop'
                || kind === 'drop_like'
                || kind === 'has_vocals'
                || kind === 'chorus_like'
            ))

            return {
                segmentId: segment.segmentId,
                label: selected
                    ? `Review window ${selectedOrderById.get(segment.segmentId) ?? index + 1}`
                    : `Detector segment ${index + 1}`,
                startSec: segment.startSec,
                endSec: segment.endSec,
                windowStartSec: segment.window.startSec,
                windowEndSec: segment.window.endSec,
                structureLabel: segment.structureLabel,
                kinds,
                segmentPriority: segment.scores.segmentPriority,
                windowScore: segment.window.score,
                selected,
                notes: `${formatTime(segment.window.startSec)}-${formatTime(segment.window.endSec)} inside ${formatTime(segment.startSec)}-${formatTime(segment.endSec)}`,
            }
        })
})
const detectorWaveformWindows = computed<StemTuningFocusWindow[]>(() => (
    detectorReviewCards.value.flatMap((segment) => ([
        {
            id: `${segment.segmentId}-segment`,
            label: `${segment.label} segment`,
            start_sec: segment.startSec,
            end_sec: segment.endSec,
            kind: 'window',
            notes: `${segment.structureLabel} segment`,
        },
        {
            id: `${segment.segmentId}-window`,
            label: segment.label,
            start_sec: segment.windowStartSec,
            end_sec: segment.windowEndSec,
            kind: segment.selected ? 'audit' : 'problem',
            notes: segment.notes,
        },
    ]))
))
const activeDetectorSegmentId = computed(() => (
    detectorReviewCards.value.find((segment) => currentTime.value >= segment.windowStartSec && currentTime.value <= segment.windowEndSec)?.segmentId
    ?? detectorReviewCards.value.find((segment) => currentTime.value >= segment.startSec && currentTime.value <= segment.endSec)?.segmentId
    ?? ''
))

function clamp(value: number, min: number, max: number): number {
    return Math.max(min, Math.min(max, value))
}

function toNumber(value: unknown, fallback: number): number {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
}

function withApiToken(url: string): string {
    if (!url) {
        return ''
    }

    const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
    if (!token) {
        return url
    }

    const resolved = new URL(url, window.location.origin)
    if (!resolved.searchParams.has('token')) {
        resolved.searchParams.set('token', token)
    }

    if (resolved.origin === window.location.origin) {
        return `${resolved.pathname}${resolved.search}${resolved.hash}`
    }

    return resolved.toString()
}

function tuningLabDebug(message: string, context: Record<string, unknown> = {}): void {
    console.info('[StemTuningLab]', message, {
        trackId: trackId.value,
        selectedSourceKey: selectedSourceKey.value,
        currentTime: Number(currentTime.value.toFixed(3)),
        loading: loading.value,
        renderBusy: renderBusy.value,
        buildBusy: buildBusy.value,
        ...context,
    })
}

function extractApiErrorMessage(error: unknown, fallback: string): string {
    const candidate = error as {
        response?: {
            data?: {
                message?: string
                error?: string
            }
        }
        message?: string
    }

    return candidate?.response?.data?.message
        ?? candidate?.response?.data?.error
        ?? candidate?.message
        ?? fallback
}

function extractApiErrorBoolean(error: unknown, key: string, fallback = false): boolean {
    const candidate = error as {
        response?: {
            data?: Record<string, unknown>
        }
    }

    const value = candidate?.response?.data?.[key]
    return typeof value === 'boolean' ? value : fallback
}

function clonePresetState(value: StemTuningPreset | null): StemTuningPreset | null {
    if (!value) {
        return null
    }

    return JSON.parse(JSON.stringify(value)) as StemTuningPreset
}

function presetSignature(value: StemTuningPreset | null): string {
    return value ? JSON.stringify(value) : ''
}

async function withSuppressedPresetWatch<T>(callback: () => T | Promise<T>): Promise<T> {
    suppressPresetWatch = true
    suppressSelectedSourcePresetSync = true

    try {
        return await callback()
    } finally {
        await nextTick()
        suppressSelectedSourcePresetSync = false
        suppressPresetWatch = false
    }
}

function setSelectedSourceKey(nextKey: string, options?: { syncPreset?: boolean }) {
    suppressSelectedSourcePresetSync = options?.syncPreset === false
    suppressSelectedSourceEditorSync = options?.syncPreset === false
    selectedSourceKey.value = nextKey
}

function jsonDraftStatusClasses(tone: JsonDraftStatusTone): string {
    switch (tone) {
    case 'pending':
        return 'text-amber-600'
    case 'success':
        return 'text-emerald-600'
    case 'error':
        return 'text-rose-600'
    default:
        return 'text-slate-500'
    }
}

function stopImportBufferApplyTimer() {
    if (importBufferApplyTimer.value !== null) {
        window.clearTimeout(importBufferApplyTimer.value)
        importBufferApplyTimer.value = null
    }
}

function stopActiveModeConfigDraftApplyTimer() {
    if (activeModeConfigDraftApplyTimer.value !== null) {
        window.clearTimeout(activeModeConfigDraftApplyTimer.value)
        activeModeConfigDraftApplyTimer.value = null
    }
}

function stopUnsafeModelInternalsDraftApplyTimer() {
    if (unsafeModelInternalsDraftApplyTimer.value !== null) {
        window.clearTimeout(unsafeModelInternalsDraftApplyTimer.value)
        unsafeModelInternalsDraftApplyTimer.value = null
    }
}

function replaceImportBufferSilently(value: string) {
    stopImportBufferApplyTimer()
    suppressImportBufferWatch = true
    importBuffer.value = value
    window.setTimeout(() => {
        suppressImportBufferWatch = false
    }, 0)
}

function replaceActiveModeConfigDraftSilently(value: string) {
    stopActiveModeConfigDraftApplyTimer()
    suppressActiveModeConfigDraftWatch = true
    activeModeConfigDraft.value = value
    window.setTimeout(() => {
        suppressActiveModeConfigDraftWatch = false
    }, 0)
}

function replaceUnsafeModelInternalsDraftSilently(value: string) {
    stopUnsafeModelInternalsDraftApplyTimer()
    suppressUnsafeModelInternalsDraftWatch = true
    unsafeModelInternalsDraft.value = value
    window.setTimeout(() => {
        suppressUnsafeModelInternalsDraftWatch = false
    }, 0)
}

function defaultLayerState(enabled: boolean): StemTuningLayerState {
    return {
        enabled,
        gainDb: 0,
        transientDb: 0,
        bodyDb: 0,
        harmonicDuckDb: 0,
        bleedDuckDb: 0,
    }
}

function defaultProcessorConfig(): StemTuningProcessorConfig {
    return {
        bypass: false,
        harmonicMargin: 1,
        percussiveMargin: 4,
        transientCutoffHz: 2500,
        bodyLowHz: 90,
        bodyHighHz: 320,
        bleedLowHz: 700,
        bleedHighHz: 3500,
    }
}

function defaultHybridBuildConfig(): StemTuningHybridBuildConfig {
    return {
        modelName: 'hybrid_htdemucs_ft_6s',
        sharpenBleed: true,
        includeMelodyStem: false,
        aggressiveMelodyFirst: false,
    }
}

function defaultStemRewriteConfig(): StemTuningStemRewriteConfig {
    return {
        otherMove: 0.45,
        guitarMove: 1,
        pianoMove: 1,
        drumsMaskedMove: 0.12,
        targetMelodyRms: 0.026,
        maxMelodyGain: 1.45,
        stageDrumHarmonicMove: 0.42,
        stageHarmonicStemPercussiveMove: 0.22,
        stageCleanupDrumHarmonicMove: 0.05,
        stageCleanupHarmonicStemPercussiveMove: 0.28,
        harmonicCount: 3,
        confidenceFloor: 0.3,
    }
}

function defaultVocalsRewriteConfig(): StemTuningVocalsRewriteConfig {
    return {
        taskMode: 'custom',
        vocalIsolationIntensity: 0,
        bleedRemovalIntensity: 0,
        airRestore: 0,
        bodyPreserve: 0,
        artifactSuppression: 0,
        centerVocalSuppression: 0,
        candidateMode: 'hybrid_vocals',
        lowBsWeight: 0.75,
        highBsWeight: 0.25,
        inactiveLowBsWeight: 0.97,
        inactiveHighBsWeight: 0.9,
        crossoverHz: 3000,
        activityAbsFloorRatio: 0.08,
        activityRelFloor: 0.35,
        activitySmoothingFrames: 11,
        vocalSharpenAlpha: 0,
        tailGateStrength: 0,
        tailGateFloorRatio: 0.08,
        tailGateSmoothingFrames: 11,
        subtractiveStrength: 0,
        subtractivePower: 1,
        subtractiveResidualFloor: 0.12,
        subtractiveActiveRestore: 0,
        subtractiveTargetRmsRatio: 0,
        subtractiveMaxGain: 1.35,
        hardMaskThreshold: 0,
        hardMaskPower: 3,
        hardMaskResidualFloor: 0.02,
    }
}

function defaultUnsafeHybridInternals(): StemTuningUnsafeHybridInternals {
    return {
        requestedDevice: 'auto',
        stage1Model: 'htdemucs_ft',
        stage2Model: 'htdemucs_6s',
        stage1RequestedDevice: 'auto',
        stage2RequestedDevice: 'auto',
        stage1Split: true,
        stage1Progress: true,
        stage1Shifts: 1,
        stage1Overlap: 0.25,
        stage1Segment: 0,
        stage2Split: true,
        stage2Progress: true,
        stage2Shifts: 1,
        stage2Overlap: 0.25,
        stage2Segment: 0,
    }
}

function defaultUnsafeBsRoformerInternals(): StemTuningUnsafeBsRoformerInternals {
    return {
        requestedDevice: 'auto',
        deviceIds: ['0'],
        forceCpu: false,
        disableDetailedPbar: false,
        repoPath: '/tmp/msst_zfturbo',
        configPath: '/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml',
        checkpointPath: '/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt',
        useTta: true,
        extractInstrumental: true,
        filenameTemplate: '{instr}',
        configOverrideYaml: '',
        extraCliArgs: [],
    }
}

function defaultUnsafeModelInternals(): StemTuningUnsafeModelInternals {
    return {
        hybrid: defaultUnsafeHybridInternals(),
        bsRoformer: defaultUnsafeBsRoformerInternals(),
    }
}

function sanitizeProcessorConfig(candidate: unknown, fallback?: StemTuningProcessorConfig): StemTuningProcessorConfig {
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}
    const defaults = fallback ?? defaultProcessorConfig()
    const bodyLowHz = clamp(toNumber(raw.bodyLowHz, defaults.bodyLowHz), 40, 400)
    const bleedLowHz = clamp(toNumber(raw.bleedLowHz, defaults.bleedLowHz), 120, 2500)

    return {
        bypass: Boolean(raw.bypass ?? defaults.bypass),
        harmonicMargin: clamp(toNumber(raw.harmonicMargin, defaults.harmonicMargin), 0.5, 4),
        percussiveMargin: clamp(toNumber(raw.percussiveMargin, defaults.percussiveMargin), 0.5, 6),
        transientCutoffHz: clamp(toNumber(raw.transientCutoffHz, defaults.transientCutoffHz), 800, 8000),
        bodyLowHz,
        bodyHighHz: clamp(toNumber(raw.bodyHighHz, defaults.bodyHighHz), bodyLowHz + 20, 1200),
        bleedLowHz,
        bleedHighHz: clamp(toNumber(raw.bleedHighHz, defaults.bleedHighHz), bleedLowHz + 100, 9000),
    }
}

function sanitizeRenderMode(candidate: unknown, fallback: StemTuningRenderMode): StemTuningRenderMode {
    return candidate === 'hybrid_build' || candidate === 'stem_rewrite' || candidate === 'vocals_rewrite' || candidate === 'post_dsp'
        ? candidate
        : fallback
}

function sanitizeHybridBuildConfig(candidate: unknown, fallback?: StemTuningHybridBuildConfig): StemTuningHybridBuildConfig {
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}
    const defaults = fallback ?? defaultHybridBuildConfig()

    return {
        modelName: raw.modelName === 'hybrid_htdemucs_ft_bs_roformer_6s' ? 'hybrid_htdemucs_ft_bs_roformer_6s' : defaults.modelName,
        sharpenBleed: Boolean(raw.sharpenBleed ?? defaults.sharpenBleed),
        includeMelodyStem: Boolean(raw.includeMelodyStem ?? defaults.includeMelodyStem),
        aggressiveMelodyFirst: Boolean(raw.aggressiveMelodyFirst ?? defaults.aggressiveMelodyFirst),
    }
}

function sanitizeStemRewriteConfig(candidate: unknown, fallback?: StemTuningStemRewriteConfig): StemTuningStemRewriteConfig {
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}
    const defaults = fallback ?? defaultStemRewriteConfig()

    return {
        otherMove: clamp(toNumber(raw.otherMove, defaults.otherMove), 0, 1.5),
        guitarMove: clamp(toNumber(raw.guitarMove, defaults.guitarMove), 0, 1.5),
        pianoMove: clamp(toNumber(raw.pianoMove, defaults.pianoMove), 0, 1.5),
        drumsMaskedMove: clamp(toNumber(raw.drumsMaskedMove, defaults.drumsMaskedMove), 0, 1),
        targetMelodyRms: clamp(toNumber(raw.targetMelodyRms, defaults.targetMelodyRms), 0.001, 0.1),
        maxMelodyGain: clamp(toNumber(raw.maxMelodyGain, defaults.maxMelodyGain), 1, 3),
        stageDrumHarmonicMove: clamp(toNumber(raw.stageDrumHarmonicMove, defaults.stageDrumHarmonicMove), 0, 1),
        stageHarmonicStemPercussiveMove: clamp(toNumber(raw.stageHarmonicStemPercussiveMove, defaults.stageHarmonicStemPercussiveMove), 0, 1),
        stageCleanupDrumHarmonicMove: clamp(toNumber(raw.stageCleanupDrumHarmonicMove, defaults.stageCleanupDrumHarmonicMove), 0, 1),
        stageCleanupHarmonicStemPercussiveMove: clamp(toNumber(raw.stageCleanupHarmonicStemPercussiveMove, defaults.stageCleanupHarmonicStemPercussiveMove), 0, 1),
        harmonicCount: Math.round(clamp(toNumber(raw.harmonicCount, defaults.harmonicCount), 1, 8)),
        confidenceFloor: clamp(toNumber(raw.confidenceFloor, defaults.confidenceFloor), 0, 1),
    }
}

function sanitizeVocalsRewriteConfig(candidate: unknown, fallback?: StemTuningVocalsRewriteConfig): StemTuningVocalsRewriteConfig {
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}
    const defaults = fallback ?? defaultVocalsRewriteConfig()

    return {
        taskMode: raw.taskMode === 'clean_vocals'
            || raw.taskMode === 'minus_from_vocals'
            || raw.taskMode === 'artifact_repair'
            || raw.taskMode === 'aggressive_vocal_suppression'
            || raw.taskMode === 'detail_preserve'
            || raw.taskMode === 'custom'
            ? raw.taskMode
            : defaults.taskMode,
        vocalIsolationIntensity: clamp(toNumber(raw.vocalIsolationIntensity, defaults.vocalIsolationIntensity), 0, 1),
        bleedRemovalIntensity: clamp(toNumber(raw.bleedRemovalIntensity, defaults.bleedRemovalIntensity), 0, 1),
        airRestore: clamp(toNumber(raw.airRestore, defaults.airRestore), 0, 1),
        bodyPreserve: clamp(toNumber(raw.bodyPreserve, defaults.bodyPreserve), 0, 1),
        artifactSuppression: clamp(toNumber(raw.artifactSuppression, defaults.artifactSuppression), 0, 1),
        centerVocalSuppression: clamp(toNumber(raw.centerVocalSuppression, defaults.centerVocalSuppression), 0, 1),
        candidateMode: raw.candidateMode === 'ft_vocals' || raw.candidateMode === 'bs_vocals' || raw.candidateMode === 'hybrid_vocals'
            ? raw.candidateMode
            : defaults.candidateMode,
        lowBsWeight: clamp(toNumber(raw.lowBsWeight, defaults.lowBsWeight), 0, 1),
        highBsWeight: clamp(toNumber(raw.highBsWeight, defaults.highBsWeight), 0, 1),
        inactiveLowBsWeight: clamp(toNumber(raw.inactiveLowBsWeight, defaults.inactiveLowBsWeight), 0, 1),
        inactiveHighBsWeight: clamp(toNumber(raw.inactiveHighBsWeight, defaults.inactiveHighBsWeight), 0, 1),
        crossoverHz: clamp(toNumber(raw.crossoverHz, defaults.crossoverHz), 200, 12000),
        activityAbsFloorRatio: clamp(toNumber(raw.activityAbsFloorRatio, defaults.activityAbsFloorRatio), 0, 1),
        activityRelFloor: clamp(toNumber(raw.activityRelFloor, defaults.activityRelFloor), 0, 1),
        activitySmoothingFrames: Math.round(clamp(toNumber(raw.activitySmoothingFrames, defaults.activitySmoothingFrames), 1, 101)),
        vocalSharpenAlpha: clamp(toNumber(raw.vocalSharpenAlpha, defaults.vocalSharpenAlpha), 0, 4),
        tailGateStrength: clamp(toNumber(raw.tailGateStrength, defaults.tailGateStrength), 0, 1),
        tailGateFloorRatio: clamp(toNumber(raw.tailGateFloorRatio, defaults.tailGateFloorRatio), 0, 1),
        tailGateSmoothingFrames: Math.round(clamp(toNumber(raw.tailGateSmoothingFrames, defaults.tailGateSmoothingFrames), 1, 101)),
        subtractiveStrength: clamp(toNumber(raw.subtractiveStrength, defaults.subtractiveStrength), 0, 4),
        subtractivePower: clamp(toNumber(raw.subtractivePower, defaults.subtractivePower), 0.1, 8),
        subtractiveResidualFloor: clamp(toNumber(raw.subtractiveResidualFloor, defaults.subtractiveResidualFloor), 0, 1),
        subtractiveActiveRestore: clamp(toNumber(raw.subtractiveActiveRestore, defaults.subtractiveActiveRestore), 0, 1),
        subtractiveTargetRmsRatio: clamp(toNumber(raw.subtractiveTargetRmsRatio, defaults.subtractiveTargetRmsRatio), 0, 4),
        subtractiveMaxGain: clamp(toNumber(raw.subtractiveMaxGain, defaults.subtractiveMaxGain), 1, 4),
        hardMaskThreshold: clamp(toNumber(raw.hardMaskThreshold, defaults.hardMaskThreshold), 0, 1),
        hardMaskPower: clamp(toNumber(raw.hardMaskPower, defaults.hardMaskPower), 1, 12),
        hardMaskResidualFloor: clamp(toNumber(raw.hardMaskResidualFloor, defaults.hardMaskResidualFloor), 0, 1),
    }
}

function sanitizeUnsafeHybridInternals(candidate: unknown, fallback?: StemTuningUnsafeHybridInternals): StemTuningUnsafeHybridInternals {
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}
    const defaults = fallback ?? defaultUnsafeHybridInternals()

    return {
        requestedDevice: String(raw.requestedDevice ?? defaults.requestedDevice ?? 'auto').trim() || 'auto',
        stage1Model: String(raw.stage1Model ?? defaults.stage1Model ?? 'htdemucs_ft').trim() || 'htdemucs_ft',
        stage2Model: String(raw.stage2Model ?? defaults.stage2Model ?? 'htdemucs_6s').trim() || 'htdemucs_6s',
        stage1RequestedDevice: String(raw.stage1RequestedDevice ?? defaults.stage1RequestedDevice ?? 'auto').trim() || 'auto',
        stage2RequestedDevice: String(raw.stage2RequestedDevice ?? defaults.stage2RequestedDevice ?? 'auto').trim() || 'auto',
        stage1Split: Boolean(raw.stage1Split ?? defaults.stage1Split),
        stage1Progress: Boolean(raw.stage1Progress ?? defaults.stage1Progress),
        stage1Shifts: Math.round(clamp(toNumber(raw.stage1Shifts, defaults.stage1Shifts), 1, 8)),
        stage1Overlap: clamp(toNumber(raw.stage1Overlap, defaults.stage1Overlap), 0, 0.99),
        stage1Segment: clamp(toNumber(raw.stage1Segment, defaults.stage1Segment), 0, 600),
        stage2Split: Boolean(raw.stage2Split ?? defaults.stage2Split),
        stage2Progress: Boolean(raw.stage2Progress ?? defaults.stage2Progress),
        stage2Shifts: Math.round(clamp(toNumber(raw.stage2Shifts, defaults.stage2Shifts), 1, 8)),
        stage2Overlap: clamp(toNumber(raw.stage2Overlap, defaults.stage2Overlap), 0, 0.99),
        stage2Segment: clamp(toNumber(raw.stage2Segment, defaults.stage2Segment), 0, 600),
    }
}

function sanitizeUnsafeBsRoformerInternals(candidate: unknown, fallback?: StemTuningUnsafeBsRoformerInternals): StemTuningUnsafeBsRoformerInternals {
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}
    const defaults = fallback ?? defaultUnsafeBsRoformerInternals()

    return {
        requestedDevice: String(raw.requestedDevice ?? defaults.requestedDevice ?? 'auto').trim() || 'auto',
        deviceIds: Array.isArray(raw.deviceIds)
            ? raw.deviceIds.map((value) => String(value).trim()).filter((value) => value.length > 0)
            : defaults.deviceIds,
        forceCpu: Boolean(raw.forceCpu ?? defaults.forceCpu),
        disableDetailedPbar: Boolean(raw.disableDetailedPbar ?? defaults.disableDetailedPbar),
        repoPath: String(raw.repoPath ?? defaults.repoPath ?? '/tmp/msst_zfturbo').trim() || '/tmp/msst_zfturbo',
        configPath: String(raw.configPath ?? defaults.configPath ?? '/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml').trim() || '/tmp/msst_zfturbo_weights/bs_roformer_vocals.yaml',
        checkpointPath: String(raw.checkpointPath ?? defaults.checkpointPath ?? '/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt').trim() || '/tmp/msst_zfturbo_weights/bs_roformer_vocals.ckpt',
        useTta: Boolean(raw.useTta ?? defaults.useTta),
        extractInstrumental: Boolean(raw.extractInstrumental ?? defaults.extractInstrumental),
        filenameTemplate: String(raw.filenameTemplate ?? defaults.filenameTemplate ?? '{instr}').trim() || '{instr}',
        configOverrideYaml: String(raw.configOverrideYaml ?? defaults.configOverrideYaml ?? ''),
        extraCliArgs: Array.isArray(raw.extraCliArgs)
            ? raw.extraCliArgs.map((value) => String(value).trim()).filter((value) => value.length > 0)
            : defaults.extraCliArgs,
    }
}

function vocalsRewriteTaskModePreset(mode: VocalsRewriteTaskMode): StemTuningVocalsRewriteConfig {
    const custom = defaultVocalsRewriteConfig()

    if (mode === 'clean_vocals') {
        return sanitizeVocalsRewriteConfig({
            ...custom,
            taskMode: mode,
            vocalIsolationIntensity: 0.58,
            bleedRemovalIntensity: 0.52,
            airRestore: 0.46,
            bodyPreserve: 0.42,
            artifactSuppression: 0.38,
            centerVocalSuppression: 0.42,
            candidateMode: 'hybrid_vocals',
            lowBsWeight: 0.82,
            highBsWeight: 0.34,
            inactiveLowBsWeight: 0.98,
            inactiveHighBsWeight: 0.93,
            crossoverHz: 3200,
            activityAbsFloorRatio: 0.06,
            activityRelFloor: 0.3,
            activitySmoothingFrames: 9,
            vocalSharpenAlpha: 0.08,
            tailGateStrength: 0.12,
            tailGateFloorRatio: 0.06,
            tailGateSmoothingFrames: 9,
            subtractiveStrength: 0.22,
            subtractivePower: 1.18,
            subtractiveResidualFloor: 0.14,
            subtractiveActiveRestore: 0.04,
            subtractiveTargetRmsRatio: 0.98,
            subtractiveMaxGain: 1.28,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        }, custom)
    }

    if (mode === 'minus_from_vocals') {
        return sanitizeVocalsRewriteConfig({
            ...custom,
            taskMode: mode,
            vocalIsolationIntensity: 0.9,
            bleedRemovalIntensity: 0.85,
            airRestore: 0.2,
            bodyPreserve: 0.2,
            artifactSuppression: 0.62,
            centerVocalSuppression: 0.92,
            candidateMode: 'hybrid_vocals',
            lowBsWeight: 0.88,
            highBsWeight: 0.42,
            inactiveLowBsWeight: 0.99,
            inactiveHighBsWeight: 0.96,
            crossoverHz: 3400,
            activityAbsFloorRatio: 0.04,
            activityRelFloor: 0.24,
            activitySmoothingFrames: 7,
            vocalSharpenAlpha: 0.04,
            tailGateStrength: 0.22,
            tailGateFloorRatio: 0.05,
            tailGateSmoothingFrames: 7,
            subtractiveStrength: 0.42,
            subtractivePower: 1.35,
            subtractiveResidualFloor: 0.08,
            subtractiveActiveRestore: 0.01,
            subtractiveTargetRmsRatio: 0.85,
            subtractiveMaxGain: 1.12,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        }, custom)
    }

    if (mode === 'artifact_repair') {
        return sanitizeVocalsRewriteConfig({
            ...custom,
            taskMode: mode,
            vocalIsolationIntensity: 0.3,
            bleedRemovalIntensity: 0.38,
            airRestore: 0.34,
            bodyPreserve: 0.72,
            artifactSuppression: 0.86,
            centerVocalSuppression: 0.18,
            candidateMode: 'ft_vocals',
            lowBsWeight: 0.58,
            highBsWeight: 0.18,
            inactiveLowBsWeight: 0.88,
            inactiveHighBsWeight: 0.78,
            crossoverHz: 2800,
            activityAbsFloorRatio: 0.09,
            activityRelFloor: 0.42,
            activitySmoothingFrames: 15,
            vocalSharpenAlpha: 0,
            tailGateStrength: 0.05,
            tailGateFloorRatio: 0.08,
            tailGateSmoothingFrames: 15,
            subtractiveStrength: 0.1,
            subtractivePower: 1.05,
            subtractiveResidualFloor: 0.2,
            subtractiveActiveRestore: 0.08,
            subtractiveTargetRmsRatio: 1.05,
            subtractiveMaxGain: 1.4,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        }, custom)
    }

    if (mode === 'aggressive_vocal_suppression') {
        return sanitizeVocalsRewriteConfig({
            ...custom,
            taskMode: mode,
            vocalIsolationIntensity: 0.96,
            bleedRemovalIntensity: 0.96,
            airRestore: 0.12,
            bodyPreserve: 0.1,
            artifactSuppression: 0.72,
            centerVocalSuppression: 0.98,
            candidateMode: 'bs_vocals',
            lowBsWeight: 0.95,
            highBsWeight: 0.62,
            inactiveLowBsWeight: 1,
            inactiveHighBsWeight: 0.98,
            crossoverHz: 3600,
            activityAbsFloorRatio: 0.03,
            activityRelFloor: 0.18,
            activitySmoothingFrames: 5,
            vocalSharpenAlpha: 0.02,
            tailGateStrength: 0.18,
            tailGateFloorRatio: 0.04,
            tailGateSmoothingFrames: 5,
            subtractiveStrength: 0,
            subtractivePower: 1,
            subtractiveResidualFloor: 0.12,
            subtractiveActiveRestore: 0,
            subtractiveTargetRmsRatio: 0,
            subtractiveMaxGain: 1.2,
            hardMaskThreshold: 0.58,
            hardMaskPower: 4.8,
            hardMaskResidualFloor: 0.015,
        }, custom)
    }

    if (mode === 'detail_preserve') {
        return sanitizeVocalsRewriteConfig({
            ...custom,
            taskMode: mode,
            vocalIsolationIntensity: 0.34,
            bleedRemovalIntensity: 0.18,
            airRestore: 0.82,
            bodyPreserve: 0.86,
            artifactSuppression: 0.22,
            centerVocalSuppression: 0.16,
            candidateMode: 'hybrid_vocals',
            lowBsWeight: 0.68,
            highBsWeight: 0.24,
            inactiveLowBsWeight: 0.94,
            inactiveHighBsWeight: 0.88,
            crossoverHz: 2600,
            activityAbsFloorRatio: 0.08,
            activityRelFloor: 0.36,
            activitySmoothingFrames: 13,
            vocalSharpenAlpha: 0.14,
            tailGateStrength: 0,
            tailGateFloorRatio: 0.08,
            tailGateSmoothingFrames: 13,
            subtractiveStrength: 0.08,
            subtractivePower: 1,
            subtractiveResidualFloor: 0.18,
            subtractiveActiveRestore: 0.12,
            subtractiveTargetRmsRatio: 1.08,
            subtractiveMaxGain: 1.55,
            hardMaskThreshold: 0,
            hardMaskPower: 3,
            hardMaskResidualFloor: 0.02,
        }, custom)
    }

    return custom
}

function sanitizeUnsafeModelInternals(candidate: unknown, fallback?: StemTuningUnsafeModelInternals): StemTuningUnsafeModelInternals {
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}
    const defaults = fallback ?? defaultUnsafeModelInternals()

    return {
        hybrid: sanitizeUnsafeHybridInternals(raw.hybrid, defaults.hybrid),
        bsRoformer: sanitizeUnsafeBsRoformerInternals(raw.bsRoformer, defaults.bsRoformer),
    }
}

function defaultLayerStatesForSources(
    availableSources: StemTuningSource[],
    sourceKey: string,
): Record<string, StemTuningLayerState> {
    const editableAvailableSources = availableSources.filter((source) => source.editable)
    if (editableAvailableSources.length === 0) {
        return {}
    }

    const firstSourceKey = editableAvailableSources[0]?.key ?? sourceKey
    const resolvedKey = editableAvailableSources.some((source) => source.key === sourceKey) ? sourceKey : firstSourceKey

    return editableAvailableSources.reduce<Record<string, StemTuningLayerState>>((acc, source) => {
        acc[source.key] = defaultLayerState(source.key === resolvedKey)
        return acc
    }, {})
}

function normalizeLayerStates(
    candidate: unknown,
    availableSources: StemTuningSource[],
    sourceKey: string,
): Record<string, StemTuningLayerState> {
    const editableAvailableSources = availableSources.filter((source) => source.editable)
    const fallback = defaultLayerStatesForSources(editableAvailableSources, sourceKey)
    const raw = typeof candidate === 'object' && candidate !== null ? candidate as Record<string, unknown> : {}

    const states = editableAvailableSources.reduce<Record<string, StemTuningLayerState>>((acc, source) => {
        const entry = raw[source.key]
        if (typeof entry === 'object' && entry !== null) {
            const layer = entry as Record<string, unknown>
            acc[source.key] = {
                enabled: Boolean(layer.enabled ?? fallback[source.key]?.enabled ?? false),
                gainDb: clamp(toNumber(layer.gainDb, fallback[source.key]?.gainDb ?? 0), -48, 12),
                transientDb: clamp(toNumber(layer.transientDb, fallback[source.key]?.transientDb ?? 0), -18, 18),
                bodyDb: clamp(toNumber(layer.bodyDb, fallback[source.key]?.bodyDb ?? 0), -18, 18),
                harmonicDuckDb: clamp(toNumber(layer.harmonicDuckDb, fallback[source.key]?.harmonicDuckDb ?? 0), -18, 0),
                bleedDuckDb: clamp(toNumber(layer.bleedDuckDb, fallback[source.key]?.bleedDuckDb ?? 0), -18, 0),
            }
            return acc
        }

        acc[source.key] = fallback[source.key] ?? defaultLayerState(false)
        return acc
    }, {})

    if (!Object.values(states).some((layer) => layer.enabled) && editableAvailableSources.length > 0) {
        const preferredKey = editableAvailableSources.some((source) => source.key === sourceKey)
            ? sourceKey
            : editableAvailableSources[0].key
        states[preferredKey] = {
            ...(fallback[preferredKey] ?? defaultLayerState(true)),
            enabled: true,
        }
    }

    return states
}

function buildFallbackPreset(
    defaultSourceKey: string,
    availableSources: StemTuningSource[],
): StemTuningPreset {
    const editableAvailableSources = availableSources.filter((source) => source.editable)
    const firstSourceKey = editableAvailableSources[0]?.key ?? defaultSourceKey
    const resolvedSourceKey = editableAvailableSources.some((source) => source.key === defaultSourceKey)
        ? defaultSourceKey
        : firstSourceKey

    return {
        version: 1,
        sourceKey: resolvedSourceKey,
        renderMode: 'post_dsp',
        layerStates: defaultLayerStatesForSources(editableAvailableSources, resolvedSourceKey),
        hybridBuildConfig: defaultHybridBuildConfig(),
        stemRewriteConfig: defaultStemRewriteConfig(),
        vocalsRewriteConfig: defaultVocalsRewriteConfig(),
        unsafeModelInternals: defaultUnsafeModelInternals(),
        processorConfig: defaultProcessorConfig(),
        focusWindow: {
            enabled: false,
            startSec: 0,
            endSec: 0,
            xfadeSec: 0,
            targetLayerKey: resolvedSourceKey,
            layerGainDb: 0,
            otherLayersDuckDb: 0,
        },
    }
}

function sanitizePreset(
    input: unknown,
    fallback: StemTuningPreset,
    availableSources: StemTuningSource[],
): StemTuningPreset {
    const editableAvailableSources = availableSources.filter((source) => source.editable)
    const candidate = typeof input === 'object' && input !== null ? input as Record<string, unknown> : {}
    const sourceKey = typeof candidate.sourceKey === 'string' ? candidate.sourceKey : fallback.sourceKey
    const resolvedSourceKey = editableAvailableSources.some((source) => source.key === sourceKey) ? sourceKey : fallback.sourceKey

    return {
        version: 1,
        sourceKey: resolvedSourceKey,
        renderMode: sanitizeRenderMode(candidate.renderMode, fallback.renderMode),
        layerStates: normalizeLayerStates(candidate.layerStates, editableAvailableSources, resolvedSourceKey),
        hybridBuildConfig: sanitizeHybridBuildConfig(candidate.hybridBuildConfig, fallback.hybridBuildConfig),
        stemRewriteConfig: sanitizeStemRewriteConfig(candidate.stemRewriteConfig, fallback.stemRewriteConfig),
        vocalsRewriteConfig: sanitizeVocalsRewriteConfig(candidate.vocalsRewriteConfig, fallback.vocalsRewriteConfig),
        unsafeModelInternals: sanitizeUnsafeModelInternals(candidate.unsafeModelInternals, fallback.unsafeModelInternals),
        processorConfig: sanitizeProcessorConfig(candidate.processorConfig, fallback.processorConfig),
        focusWindow: {
            enabled: false,
            startSec: 0,
            endSec: 0,
            xfadeSec: 0,
            targetLayerKey: resolvedSourceKey,
            layerGainDb: 0,
            otherLayersDuckDb: 0,
        },
    }
}

function formatTime(seconds: number): string {
    const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0
    const mins = Math.floor(safe / 60)
    const secs = Math.floor(safe % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
}

function formatFileSize(bytes: number): string {
    if (!Number.isFinite(bytes) || bytes <= 0) {
        return '0 B'
    }

    const units = ['B', 'KB', 'MB', 'GB']
    let value = bytes
    let unitIndex = 0
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024
        unitIndex += 1
    }
    return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`
}

function formatDetectorKindLabel(kind: string): string {
    switch (kind) {
        case 'vocal_drop':
            return 'Vocal drop'
        case 'drop_like':
            return 'Drop-like'
        case 'has_vocals':
            return 'Has vocals'
        case 'chorus_like':
            return 'Chorus-like'
        default:
            return kind.replace(/_/g, ' ')
    }
}

async function jumpToDetectorSegment(segmentId: string): Promise<void> {
    const target = detectorReviewCards.value.find((segment) => segment.segmentId === segmentId)
    if (!target) {
        return
    }

    if (waveformTransportRef.value?.playFrom) {
        await waveformTransportRef.value.playFrom(target.windowStartSec)
        return
    }

    if (mediaElementRef.value) {
        mediaElementRef.value.currentTime = target.windowStartSec
        await mediaElementRef.value.play().catch(() => undefined)
        currentTime.value = target.windowStartSec
    }
}

function normalizeSourceLayerKey(key: string): string {
    return key.replace(/^(stem_|render_)/, '')
}

function isLayerEnabled(key: string): boolean {
    return Boolean(currentLayerStates.value[key]?.enabled)
}

function layerGainDb(key: string): number {
    return currentLayerStates.value[key]?.gainDb ?? 0
}

function layerProcessorValue(
    key: string,
    field: 'transientDb' | 'bodyDb' | 'harmonicDuckDb' | 'bleedDuckDb',
): number {
    return currentLayerStates.value[key]?.[field] ?? 0
}

const currentProcessorConfig = computed<StemTuningProcessorConfig>(() => (
    preset.value?.processorConfig ?? defaultProcessorConfig()
))
const postDspBypassed = computed(() => currentProcessorConfig.value.bypass)
const currentRenderMode = computed<StemTuningRenderMode>(() => (
    preset.value?.renderMode ?? 'post_dsp'
))
const stemRewriteAffectedSources = computed(() => (
    editableSources.value.filter((source) => STEM_REWRITE_AFFECTED_BASE_KEYS.includes(
        normalizeSourceLayerKey(source.key) as typeof STEM_REWRITE_AFFECTED_BASE_KEYS[number],
    ))
))
const stemRewriteSelectedSourcePreserved = computed(() => (
    currentRenderMode.value === 'stem_rewrite'
    && STEM_REWRITE_PRESERVED_BASE_KEYS.includes(
        normalizeSourceLayerKey(selectedSourceKey.value) as typeof STEM_REWRITE_PRESERVED_BASE_KEYS[number],
    )
))
const stemRewriteEnabledAffectedCount = computed(() => (
    editableSources.value.filter((source) => (
        isLayerEnabled(source.key)
        && STEM_REWRITE_AFFECTED_BASE_KEYS.includes(
            normalizeSourceLayerKey(source.key) as typeof STEM_REWRITE_AFFECTED_BASE_KEYS[number],
        )
    )).length
))
const stemRewriteMixWillSoundUnchanged = computed(() => (
    currentRenderMode.value === 'stem_rewrite'
    && editableSources.value.some((source) => isLayerEnabled(source.key))
    && stemRewriteEnabledAffectedCount.value === 0
))
const currentHybridBuildConfig = computed<StemTuningHybridBuildConfig>(() => (
    preset.value?.hybridBuildConfig ?? defaultHybridBuildConfig()
))
const currentStemRewriteConfig = computed<StemTuningStemRewriteConfig>(() => (
    preset.value?.stemRewriteConfig ?? defaultStemRewriteConfig()
))
const currentVocalsRewriteConfig = computed<StemTuningVocalsRewriteConfig>(() => (
    preset.value?.vocalsRewriteConfig ?? defaultVocalsRewriteConfig()
))
const currentUnsafeModelInternals = computed<StemTuningUnsafeModelInternals>(() => (
    preset.value?.unsafeModelInternals ?? defaultUnsafeModelInternals()
))
const currentUnsafeHybridInternals = computed<StemTuningUnsafeHybridInternals>(() => (
    currentUnsafeModelInternals.value.hybrid
))
const currentUnsafeBsRoformerInternals = computed<StemTuningUnsafeBsRoformerInternals>(() => (
    currentUnsafeModelInternals.value.bsRoformer
))
const baselinePreset = computed<StemTuningPreset | null>(() => {
    if (!session.value) {
        return null
    }

    const fallbackPreset = buildFallbackPreset(session.value.lab.default_source_key, sources.value)
    return sanitizePreset(session.value.lab.default_preset ?? {}, fallbackPreset, sources.value)
})
const isPostDspMode = computed(() => currentRenderMode.value === 'post_dsp')
const isHybridBuildMode = computed(() => currentRenderMode.value === 'hybrid_build')
const isStemRewriteMode = computed(() => currentRenderMode.value === 'stem_rewrite')
const isVocalsRewriteMode = computed(() => currentRenderMode.value === 'vocals_rewrite')
const selectedPreviewStemKey = computed(() => stemBaseKey(selectedSource.value?.key ?? ''))
const selectedEditorStemKey = computed(() => stemBaseKey(editorLayerCard.value?.key ?? ''))
const vocalsSourceKey = computed(() => (
    editableSources.value.find((source) => stemBaseKey(source.key) === 'vocals')?.key ?? ''
))
const renderVocalsSourceKey = computed(() => (
    sources.value.find((source) => source.key === 'render_vocals')?.key ?? ''
))
const vocalsRewriteEnabledLayerCount = computed(() => (
    editableSources.value.filter((source) => isLayerEnabled(source.key)).length
))
const stemRewritePreviewNoOp = computed(() => (
    isStemRewriteMode.value
    && ['vocals', 'bass'].includes(selectedPreviewStemKey.value)
))
const stemRewriteEditorNoOp = computed(() => (
    isStemRewriteMode.value
    && ['vocals', 'bass'].includes(selectedEditorStemKey.value)
))
const vocalsRewritePreviewNoOp = computed(() => (
    isVocalsRewriteMode.value
    && selectedPreviewStemKey.value !== 'vocals'
))
const vocalsRewriteEditorNoOp = computed(() => (
    isVocalsRewriteMode.value
    && selectedEditorStemKey.value !== 'vocals'
))
const vocalsRewriteMixPreviewTrap = computed(() => (
    isVocalsRewriteMode.value
    && vocalsRewriteEnabledLayerCount.value > 1
))
const currentVocalsRewriteCleanupMode = computed<VocalsRewriteCleanupMode>(() => {
    if (currentVocalsRewriteConfig.value.subtractiveStrength > 1e-6) {
        return 'subtractive'
    }

    if (currentVocalsRewriteConfig.value.hardMaskThreshold > 1e-6) {
        return 'hard_mask'
    }

    return 'none'
})
const activeModeConfigField = computed<'processorConfig' | 'hybridBuildConfig' | 'stemRewriteConfig' | 'vocalsRewriteConfig'>(() => {
    if (isVocalsRewriteMode.value) {
        return 'vocalsRewriteConfig'
    }

    if (isStemRewriteMode.value) {
        return 'stemRewriteConfig'
    }

    if (isHybridBuildMode.value) {
        return 'hybridBuildConfig'
    }

    return 'processorConfig'
})
const activeModeConfigLabel = computed(() => {
    if (isVocalsRewriteMode.value) {
        return 'Vocals Rewrite'
    }

    if (isStemRewriteMode.value) {
        return 'Stem Rewrite'
    }

    if (isHybridBuildMode.value) {
        return 'Hybrid Build'
    }

    return 'Post-DSP'
})
const activeModeConfigValue = computed<Record<string, unknown>>(() => {
    if (isVocalsRewriteMode.value) {
        return { ...currentVocalsRewriteConfig.value }
    }

    if (isStemRewriteMode.value) {
        return { ...currentStemRewriteConfig.value }
    }

    if (isHybridBuildMode.value) {
        return { ...currentHybridBuildConfig.value }
    }

    return { ...currentProcessorConfig.value }
})
const activeModeConfigJson = computed(() => JSON.stringify(activeModeConfigValue.value, null, 2))
const unsafeModelInternalsJson = computed(() => JSON.stringify(currentUnsafeModelInternals.value, null, 2))
const unsafeHybridRootFields: UnsafeFieldDefinition[] = [
    {
        key: 'requestedDevice',
        label: 'Global requested device',
        description: 'Fallback device request for the hybrid pipeline before stage-specific overrides.',
        kind: 'text',
        placeholder: 'auto',
    },
]
const unsafeHybridStageFields: UnsafeFieldDefinition[] = [
    {
        key: 'model',
        label: 'Stage model',
        description: 'Demucs-family model name for this hybrid stage.',
        kind: 'text',
    },
    {
        key: 'requestedDevice',
        label: 'Stage requested device',
        description: 'Per-stage device request. Use auto unless you intentionally want CPU fallback or a forced CUDA path.',
        kind: 'text',
        placeholder: 'auto',
    },
    {
        key: 'split',
        label: 'Split long audio',
        description: 'Maps directly to Demucs apply_model(split=...).',
        kind: 'checkbox',
    },
    {
        key: 'progress',
        label: 'Verbose progress',
        description: 'Maps directly to Demucs apply_model(progress=...).',
        kind: 'checkbox',
    },
    {
        key: 'shifts',
        label: 'Shifts',
        description: 'Extra Demucs stochastic shifts. Higher is slower.',
        kind: 'number',
        min: 1,
        max: 8,
        step: 1,
    },
    {
        key: 'overlap',
        label: 'Overlap',
        description: 'Demucs overlap ratio when split is enabled.',
        kind: 'number',
        min: 0,
        max: 0.99,
        step: 0.01,
    },
    {
        key: 'segment',
        label: 'Segment seconds',
        description: 'Per-stage Demucs segment length. Zero means do not override the model default.',
        kind: 'number',
        min: 0,
        max: 600,
        step: 0.5,
    },
]
const unsafeBsRoformerFields: UnsafeFieldDefinition[] = [
    {
        key: 'requestedDevice',
        label: 'Requested device',
        description: 'Primary BS-RoFormer device request before CPU forcing or explicit GPU ids.',
        kind: 'text',
        placeholder: 'auto',
    },
    {
        key: 'deviceIds',
        label: 'Device IDs',
        description: 'CUDA device ids passed to the wrapper as --device_ids. Comma or newline separated.',
        kind: 'string_array',
        placeholder: '0,1',
    },
    {
        key: 'forceCpu',
        label: 'Force CPU',
        description: 'Adds --force_cpu and ignores CUDA device ids.',
        kind: 'checkbox',
    },
    {
        key: 'disableDetailedPbar',
        label: 'Disable detailed progress bar',
        description: 'Adds --disable_detailed_pbar to the BS-RoFormer wrapper.',
        kind: 'checkbox',
    },
    {
        key: 'useTta',
        label: 'Use TTA',
        description: 'Adds --use_tta.',
        kind: 'checkbox',
    },
    {
        key: 'extractInstrumental',
        label: 'Extract instrumental',
        description: 'Adds --extract_instrumental.',
        kind: 'checkbox',
    },
    {
        key: 'filenameTemplate',
        label: 'Filename template',
        description: 'Wrapper filename template for emitted stems.',
        kind: 'text',
        placeholder: '{instr}',
    },
    {
        key: 'repoPath',
        label: 'Repo path',
        description: 'Container path to the BS-RoFormer wrapper repo.',
        kind: 'text',
    },
    {
        key: 'configPath',
        label: 'Config path',
        description: 'Container path to the vendor YAML config.',
        kind: 'text',
    },
    {
        key: 'checkpointPath',
        label: 'Checkpoint path',
        description: 'Container path to the BS-RoFormer checkpoint.',
        kind: 'text',
    },
    {
        key: 'extraCliArgs',
        label: 'Extra CLI args',
        description: 'Additional wrapper args appended after the first-class flags. Comma or newline separated.',
        kind: 'string_array',
        placeholder: '--disable_detailed_pbar',
    },
    {
        key: 'configOverrideYaml',
        label: 'YAML override',
        description: 'Raw YAML replacement for configPath. Leave empty to use the vendor config file as-is.',
        kind: 'textarea',
        placeholder: 'audio:\\n  chunk_size: 485100',
    },
]
const vocalsRewriteCleanupConflict = computed(() => (
    isVocalsRewriteMode.value
    && currentVocalsRewriteConfig.value.subtractiveStrength > 1e-6
    && currentVocalsRewriteConfig.value.hardMaskThreshold > 1e-6
))
const activeModeAdjustmentCount = computed(() => {
    const baseline = baselinePreset.value
    if (!baseline) {
        return 0
    }

    if (isStemRewriteMode.value) {
        return countStemRewriteConfigAdjustments(currentStemRewriteConfig.value, baseline.stemRewriteConfig)
    }

    if (isVocalsRewriteMode.value) {
        return countVocalsRewriteConfigAdjustments(currentVocalsRewriteConfig.value, baseline.vocalsRewriteConfig)
    }

    if (isHybridBuildMode.value) {
        return countHybridBuildConfigAdjustments(currentHybridBuildConfig.value, baseline.hybridBuildConfig)
    }

    return countProcessorConfigAdjustments(currentProcessorConfig.value, baseline.processorConfig)
})
const editorChangedParamCount = computed(() => (
    (editorLayerCard.value?.adjustmentCount ?? 0) + activeModeAdjustmentCount.value
))

function stemBaseKey(key: string): string {
    if (key.startsWith('render_')) {
        return key.slice('render_'.length)
    }

    if (key.startsWith('stem_')) {
        return key.slice('stem_'.length)
    }

    return key
}

function countNumericAdjustment(current: number, baseline: number, epsilon = 0.0005): number {
    return Math.abs(current - baseline) >= epsilon ? 1 : 0
}

function countProcessorConfigAdjustments(
    current: StemTuningProcessorConfig,
    baseline: StemTuningProcessorConfig,
): number {
    return (
        (current.bypass !== baseline.bypass ? 1 : 0)
        + countNumericAdjustment(current.harmonicMargin, baseline.harmonicMargin)
        + countNumericAdjustment(current.percussiveMargin, baseline.percussiveMargin)
        + countNumericAdjustment(current.transientCutoffHz, baseline.transientCutoffHz)
        + countNumericAdjustment(current.bodyLowHz, baseline.bodyLowHz)
        + countNumericAdjustment(current.bodyHighHz, baseline.bodyHighHz)
        + countNumericAdjustment(current.bleedLowHz, baseline.bleedLowHz)
        + countNumericAdjustment(current.bleedHighHz, baseline.bleedHighHz)
    )
}

function countHybridBuildConfigAdjustments(
    current: StemTuningHybridBuildConfig,
    baseline: StemTuningHybridBuildConfig,
): number {
    return (
        (current.modelName !== baseline.modelName ? 1 : 0)
        + (current.sharpenBleed !== baseline.sharpenBleed ? 1 : 0)
        + (current.includeMelodyStem !== baseline.includeMelodyStem ? 1 : 0)
        + (current.aggressiveMelodyFirst !== baseline.aggressiveMelodyFirst ? 1 : 0)
    )
}

function countStemRewriteConfigAdjustments(
    current: StemTuningStemRewriteConfig,
    baseline: StemTuningStemRewriteConfig,
): number {
    return (
        countNumericAdjustment(current.otherMove, baseline.otherMove)
        + countNumericAdjustment(current.guitarMove, baseline.guitarMove)
        + countNumericAdjustment(current.pianoMove, baseline.pianoMove)
        + countNumericAdjustment(current.drumsMaskedMove, baseline.drumsMaskedMove)
        + countNumericAdjustment(current.targetMelodyRms, baseline.targetMelodyRms)
        + countNumericAdjustment(current.maxMelodyGain, baseline.maxMelodyGain)
        + countNumericAdjustment(current.stageDrumHarmonicMove, baseline.stageDrumHarmonicMove)
        + countNumericAdjustment(current.stageHarmonicStemPercussiveMove, baseline.stageHarmonicStemPercussiveMove)
        + countNumericAdjustment(current.stageCleanupDrumHarmonicMove, baseline.stageCleanupDrumHarmonicMove)
        + countNumericAdjustment(current.stageCleanupHarmonicStemPercussiveMove, baseline.stageCleanupHarmonicStemPercussiveMove)
        + (current.harmonicCount !== baseline.harmonicCount ? 1 : 0)
        + countNumericAdjustment(current.confidenceFloor, baseline.confidenceFloor)
    )
}

function countVocalsRewriteConfigAdjustments(
    current: StemTuningVocalsRewriteConfig,
    baseline: StemTuningVocalsRewriteConfig,
): number {
    return (
        (current.taskMode !== baseline.taskMode ? 1 : 0)
        + countNumericAdjustment(current.vocalIsolationIntensity, baseline.vocalIsolationIntensity)
        + countNumericAdjustment(current.bleedRemovalIntensity, baseline.bleedRemovalIntensity)
        + countNumericAdjustment(current.airRestore, baseline.airRestore)
        + countNumericAdjustment(current.bodyPreserve, baseline.bodyPreserve)
        + countNumericAdjustment(current.artifactSuppression, baseline.artifactSuppression)
        + countNumericAdjustment(current.centerVocalSuppression, baseline.centerVocalSuppression)
        + (current.candidateMode !== baseline.candidateMode ? 1 : 0)
        + countNumericAdjustment(current.lowBsWeight, baseline.lowBsWeight)
        + countNumericAdjustment(current.highBsWeight, baseline.highBsWeight)
        + countNumericAdjustment(current.inactiveLowBsWeight, baseline.inactiveLowBsWeight)
        + countNumericAdjustment(current.inactiveHighBsWeight, baseline.inactiveHighBsWeight)
        + countNumericAdjustment(current.crossoverHz, baseline.crossoverHz)
        + countNumericAdjustment(current.activityAbsFloorRatio, baseline.activityAbsFloorRatio)
        + countNumericAdjustment(current.activityRelFloor, baseline.activityRelFloor)
        + (current.activitySmoothingFrames !== baseline.activitySmoothingFrames ? 1 : 0)
        + countNumericAdjustment(current.vocalSharpenAlpha, baseline.vocalSharpenAlpha)
        + countNumericAdjustment(current.tailGateStrength, baseline.tailGateStrength)
        + countNumericAdjustment(current.tailGateFloorRatio, baseline.tailGateFloorRatio)
        + (current.tailGateSmoothingFrames !== baseline.tailGateSmoothingFrames ? 1 : 0)
        + countNumericAdjustment(current.subtractiveStrength, baseline.subtractiveStrength)
        + countNumericAdjustment(current.subtractivePower, baseline.subtractivePower)
        + countNumericAdjustment(current.subtractiveResidualFloor, baseline.subtractiveResidualFloor)
        + countNumericAdjustment(current.subtractiveActiveRestore, baseline.subtractiveActiveRestore)
        + countNumericAdjustment(current.subtractiveTargetRmsRatio, baseline.subtractiveTargetRmsRatio)
        + countNumericAdjustment(current.subtractiveMaxGain, baseline.subtractiveMaxGain)
        + countNumericAdjustment(current.hardMaskThreshold, baseline.hardMaskThreshold)
        + countNumericAdjustment(current.hardMaskPower, baseline.hardMaskPower)
        + countNumericAdjustment(current.hardMaskResidualFloor, baseline.hardMaskResidualFloor)
    )
}

function setRenderMode(nextMode: StemTuningRenderMode) {
    if (!preset.value) {
        return
    }

    preset.value.renderMode = nextMode
}

function setProcessorConfigValue(
    field: keyof StemTuningProcessorConfig,
    value: number | boolean,
) {
    if (!preset.value) {
        return
    }

    const candidate = {
        ...currentProcessorConfig.value,
        [field]: value,
    }
    preset.value.processorConfig = sanitizeProcessorConfig(candidate, currentProcessorConfig.value)
}

function clearProcessorBypassForActiveEdit() {
    if (!preset.value || !currentProcessorConfig.value.bypass) {
        return
    }

    preset.value.processorConfig = sanitizeProcessorConfig({
        ...currentProcessorConfig.value,
        bypass: false,
    }, currentProcessorConfig.value)
}

function setHybridBuildConfigValue(
    field: keyof StemTuningHybridBuildConfig,
    value: string | boolean,
) {
    if (!preset.value) {
        return
    }

    const candidate = {
        ...currentHybridBuildConfig.value,
        [field]: value,
    }
    preset.value.hybridBuildConfig = sanitizeHybridBuildConfig(candidate, currentHybridBuildConfig.value)
}

function setStemRewriteConfigValue(
    field: keyof StemTuningStemRewriteConfig,
    value: number,
) {
    if (!preset.value) {
        return
    }

    const candidate = {
        ...currentStemRewriteConfig.value,
        [field]: value,
    }
    preset.value.stemRewriteConfig = sanitizeStemRewriteConfig(candidate, currentStemRewriteConfig.value)
}

function setVocalsRewriteConfigValue(
    field: keyof StemTuningVocalsRewriteConfig,
    value: string | number,
) {
    if (!preset.value) {
        return
    }

    const candidate = {
        ...currentVocalsRewriteConfig.value,
        [field]: value,
    }
    preset.value.vocalsRewriteConfig = sanitizeVocalsRewriteConfig(candidate, currentVocalsRewriteConfig.value)
}

function applyVocalsRewriteTaskMode(mode: VocalsRewriteTaskMode) {
    if (!preset.value) {
        return
    }

    preset.value.vocalsRewriteConfig = sanitizeVocalsRewriteConfig(
        vocalsRewriteTaskModePreset(mode),
        currentVocalsRewriteConfig.value,
    )
}

function vocalsRewriteAudiblePresetConfig(presetName: VocalsRewriteAudiblePreset): StemTuningVocalsRewriteConfig {
    if (presetName === 'tight_clean') {
        return sanitizeVocalsRewriteConfig({
            ...vocalsRewriteTaskModePreset('clean_vocals'),
            taskMode: 'clean_vocals',
            vocalIsolationIntensity: 0.74,
            bleedRemovalIntensity: 0.7,
            airRestore: 0.52,
            bodyPreserve: 0.48,
            artifactSuppression: 0.48,
            centerVocalSuppression: 0.58,
            subtractiveStrength: 0.32,
            subtractivePower: 1.26,
            tailGateStrength: 0.16,
            vocalSharpenAlpha: 0.1,
        }, currentVocalsRewriteConfig.value)
    }

    if (presetName === 'hard_isolate') {
        return sanitizeVocalsRewriteConfig({
            ...vocalsRewriteTaskModePreset('minus_from_vocals'),
            taskMode: 'minus_from_vocals',
            vocalIsolationIntensity: 0.97,
            bleedRemovalIntensity: 0.95,
            airRestore: 0.16,
            bodyPreserve: 0.12,
            artifactSuppression: 0.7,
            centerVocalSuppression: 0.99,
            subtractiveStrength: 0.56,
            subtractivePower: 1.42,
            subtractiveResidualFloor: 0.06,
            subtractiveTargetRmsRatio: 0.8,
            tailGateStrength: 0.22,
        }, currentVocalsRewriteConfig.value)
    }

    if (presetName === 'artifact_killer') {
        return sanitizeVocalsRewriteConfig({
            ...vocalsRewriteTaskModePreset('artifact_repair'),
            taskMode: 'artifact_repair',
            vocalIsolationIntensity: 0.42,
            bleedRemovalIntensity: 0.44,
            airRestore: 0.28,
            bodyPreserve: 0.84,
            artifactSuppression: 0.98,
            centerVocalSuppression: 0.24,
            subtractiveStrength: 0.16,
            subtractiveActiveRestore: 0.12,
            subtractiveTargetRmsRatio: 1.08,
            subtractiveMaxGain: 1.48,
            tailGateStrength: 0.04,
        }, currentVocalsRewriteConfig.value)
    }

    return sanitizeVocalsRewriteConfig({
        ...vocalsRewriteTaskModePreset('detail_preserve'),
        taskMode: 'detail_preserve',
        vocalIsolationIntensity: 0.4,
        bleedRemovalIntensity: 0.22,
        airRestore: 0.94,
        bodyPreserve: 0.94,
        artifactSuppression: 0.18,
        centerVocalSuppression: 0.14,
        vocalSharpenAlpha: 0.18,
        subtractiveStrength: 0.05,
        subtractiveActiveRestore: 0.16,
        subtractiveTargetRmsRatio: 1.12,
        subtractiveMaxGain: 1.65,
    }, currentVocalsRewriteConfig.value)
}

function applyVocalsRewriteAudiblePreset(presetName: VocalsRewriteAudiblePreset) {
    if (!preset.value) {
        return
    }

    preset.value.renderMode = 'vocals_rewrite'
    preset.value.vocalsRewriteConfig = vocalsRewriteAudiblePresetConfig(presetName)
}

function setVocalsRewriteCleanupMode(mode: VocalsRewriteCleanupMode) {
    if (!preset.value) {
        return
    }

    const current = currentVocalsRewriteConfig.value
    const candidate: StemTuningVocalsRewriteConfig = {
        ...current,
    }

    if (mode !== 'subtractive') {
        candidate.subtractiveStrength = 0
    }

    if (mode !== 'hard_mask') {
        candidate.hardMaskThreshold = 0
    }

    preset.value.vocalsRewriteConfig = sanitizeVocalsRewriteConfig(candidate, current)
}

function loadActiveModeConfigDraft() {
    replaceActiveModeConfigDraftSilently(activeModeConfigJson.value)
    activeModeConfigDraftStatusTone.value = 'muted'
    activeModeConfigDraftStatusMessage.value = 'Valid mode JSON auto-applies after a short pause.'
}

function loadUnsafeModelInternalsDraft() {
    replaceUnsafeModelInternalsDraftSilently(unsafeModelInternalsJson.value)
    unsafeModelInternalsDraftStatusTone.value = 'muted'
    unsafeModelInternalsDraftStatusMessage.value = 'Valid unsafe JSON auto-applies after a short pause.'
}

async function copyActiveModeConfigJson() {
    await navigator.clipboard.writeText(activeModeConfigJson.value)
    toast.success(`${activeModeConfigLabel.value} config copied`)
}

async function copyUnsafeModelInternalsJson() {
    await navigator.clipboard.writeText(unsafeModelInternalsJson.value)
    toast.success('Unsafe model internals copied')
}

function applyActiveModeConfigDraft(options?: { silent?: boolean }) {
    if (!preset.value) {
        return false
    }

    let parsed: unknown
    try {
        parsed = JSON.parse(activeModeConfigDraft.value || '{}')
    } catch (error) {
        activeModeConfigDraftStatusTone.value = 'error'
        activeModeConfigDraftStatusMessage.value = `Invalid ${activeModeConfigLabel.value} JSON. Nothing applied.`
        if (!options?.silent) {
            toast.error(`Invalid ${activeModeConfigLabel.value} config JSON`)
        }
        return false
    }

    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        activeModeConfigDraftStatusTone.value = 'error'
        activeModeConfigDraftStatusMessage.value = `${activeModeConfigLabel.value} JSON must be an object.`
        if (!options?.silent) {
            toast.error(`${activeModeConfigLabel.value} config must be a JSON object`)
        }
        return false
    }

    let nextConfigJson = ''
    if (activeModeConfigField.value === 'vocalsRewriteConfig') {
        const nextConfig = sanitizeVocalsRewriteConfig(parsed, currentVocalsRewriteConfig.value)
        nextConfigJson = JSON.stringify(nextConfig)
        if (nextConfigJson === JSON.stringify(currentVocalsRewriteConfig.value)) {
            activeModeConfigDraftStatusTone.value = 'success'
            activeModeConfigDraftStatusMessage.value = `${activeModeConfigLabel.value} already matches current state.`
            return true
        }
        preset.value.vocalsRewriteConfig = nextConfig
    } else if (activeModeConfigField.value === 'stemRewriteConfig') {
        const nextConfig = sanitizeStemRewriteConfig(parsed, currentStemRewriteConfig.value)
        nextConfigJson = JSON.stringify(nextConfig)
        if (nextConfigJson === JSON.stringify(currentStemRewriteConfig.value)) {
            activeModeConfigDraftStatusTone.value = 'success'
            activeModeConfigDraftStatusMessage.value = `${activeModeConfigLabel.value} already matches current state.`
            return true
        }
        preset.value.stemRewriteConfig = nextConfig
    } else if (activeModeConfigField.value === 'hybridBuildConfig') {
        const nextConfig = sanitizeHybridBuildConfig(parsed, currentHybridBuildConfig.value)
        nextConfigJson = JSON.stringify(nextConfig)
        if (nextConfigJson === JSON.stringify(currentHybridBuildConfig.value)) {
            activeModeConfigDraftStatusTone.value = 'success'
            activeModeConfigDraftStatusMessage.value = `${activeModeConfigLabel.value} already matches current state.`
            return true
        }
        preset.value.hybridBuildConfig = nextConfig
    } else {
        const nextConfig = sanitizeProcessorConfig(parsed, currentProcessorConfig.value)
        nextConfigJson = JSON.stringify(nextConfig)
        if (nextConfigJson === JSON.stringify(currentProcessorConfig.value)) {
            activeModeConfigDraftStatusTone.value = 'success'
            activeModeConfigDraftStatusMessage.value = `${activeModeConfigLabel.value} already matches current state.`
            return true
        }
        preset.value.processorConfig = nextConfig
    }

    loadActiveModeConfigDraft()
    activeModeConfigDraftStatusTone.value = 'success'
    activeModeConfigDraftStatusMessage.value = `${activeModeConfigLabel.value} JSON applied automatically.`
    if (!options?.silent) {
        toast.success(`${activeModeConfigLabel.value} config applied`)
    }
    return true
}

function applyUnsafeModelInternalsDraft(options?: { silent?: boolean }) {
    if (!preset.value) {
        return false
    }

    let parsed: unknown
    try {
        parsed = JSON.parse(unsafeModelInternalsDraft.value || '{}')
    } catch (error) {
        unsafeModelInternalsDraftStatusTone.value = 'error'
        unsafeModelInternalsDraftStatusMessage.value = 'Invalid unsafe internals JSON. Nothing applied.'
        if (!options?.silent) {
            toast.error('Invalid unsafe model internals JSON')
        }
        return false
    }

    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
        unsafeModelInternalsDraftStatusTone.value = 'error'
        unsafeModelInternalsDraftStatusMessage.value = 'Unsafe internals JSON must be an object.'
        if (!options?.silent) {
            toast.error('Unsafe model internals must be a JSON object')
        }
        return false
    }

    const nextConfig = sanitizeUnsafeModelInternals(parsed, currentUnsafeModelInternals.value)
    if (JSON.stringify(nextConfig) === JSON.stringify(currentUnsafeModelInternals.value)) {
        unsafeModelInternalsDraftStatusTone.value = 'success'
        unsafeModelInternalsDraftStatusMessage.value = 'Unsafe internals already match current state.'
        return true
    }

    preset.value.unsafeModelInternals = nextConfig
    loadUnsafeModelInternalsDraft()
    unsafeModelInternalsDraftStatusTone.value = 'success'
    unsafeModelInternalsDraftStatusMessage.value = 'Unsafe internals JSON applied automatically.'
    if (!options?.silent) {
        toast.success('Unsafe model internals applied')
    }
    return true
}

function scheduleImportBufferAutoApply() {
    stopImportBufferApplyTimer()

    if (!importBuffer.value.trim()) {
        importBufferStatusTone.value = 'muted'
        importBufferStatusMessage.value = 'Paste any preset JSON. Valid JSON auto-applies after a short pause.'
        return
    }

    importBufferStatusTone.value = 'pending'
    importBufferStatusMessage.value = 'Preset JSON looks edited. Auto-applying after a short pause...'
    importBufferApplyTimer.value = window.setTimeout(() => {
        importBufferApplyTimer.value = null
        importPresetFromTextarea({ silent: true })
    }, JSON_AUTO_APPLY_DELAY_MS)
}

function scheduleActiveModeConfigDraftAutoApply() {
    stopActiveModeConfigDraftApplyTimer()

    if (!activeModeConfigDraft.value.trim()) {
        activeModeConfigDraftStatusTone.value = 'muted'
        activeModeConfigDraftStatusMessage.value = 'Valid mode JSON auto-applies after a short pause.'
        return
    }

    activeModeConfigDraftStatusTone.value = 'pending'
    activeModeConfigDraftStatusMessage.value = `${activeModeConfigLabel.value} JSON edited. Auto-applying after a short pause...`
    activeModeConfigDraftApplyTimer.value = window.setTimeout(() => {
        activeModeConfigDraftApplyTimer.value = null
        applyActiveModeConfigDraft({ silent: true })
    }, JSON_AUTO_APPLY_DELAY_MS)
}

function scheduleUnsafeModelInternalsDraftAutoApply() {
    stopUnsafeModelInternalsDraftApplyTimer()

    if (!unsafeModelInternalsDraft.value.trim()) {
        unsafeModelInternalsDraftStatusTone.value = 'muted'
        unsafeModelInternalsDraftStatusMessage.value = 'Valid unsafe JSON auto-applies after a short pause.'
        return
    }

    unsafeModelInternalsDraftStatusTone.value = 'pending'
    unsafeModelInternalsDraftStatusMessage.value = 'Unsafe JSON edited. Auto-applying after a short pause...'
    unsafeModelInternalsDraftApplyTimer.value = window.setTimeout(() => {
        unsafeModelInternalsDraftApplyTimer.value = null
        applyUnsafeModelInternalsDraft({ silent: true })
    }, JSON_AUTO_APPLY_DELAY_MS)
}

function parseUnsafeStringArray(value: string): string[] {
    return value
        .split(/[\n,]/g)
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0)
}

function hybridStageFieldKey(
    stage: number,
    field: string,
): keyof StemTuningUnsafeHybridInternals {
    const prefix = Number(stage) === 2 ? 'stage2' : 'stage1'

    switch (field) {
    case 'model':
        return `${prefix}Model` as keyof StemTuningUnsafeHybridInternals
    case 'requestedDevice':
        return `${prefix}RequestedDevice` as keyof StemTuningUnsafeHybridInternals
    case 'split':
        return `${prefix}Split` as keyof StemTuningUnsafeHybridInternals
    case 'progress':
        return `${prefix}Progress` as keyof StemTuningUnsafeHybridInternals
    case 'shifts':
        return `${prefix}Shifts` as keyof StemTuningUnsafeHybridInternals
    case 'overlap':
        return `${prefix}Overlap` as keyof StemTuningUnsafeHybridInternals
    case 'segment':
        return `${prefix}Segment` as keyof StemTuningUnsafeHybridInternals
    }

    return `${prefix}RequestedDevice` as keyof StemTuningUnsafeHybridInternals
}

function setUnsafeHybridValue(field: string, value: string | number | boolean) {
    if (!preset.value) {
        return
    }

    preset.value.unsafeModelInternals = sanitizeUnsafeModelInternals({
        ...currentUnsafeModelInternals.value,
        hybrid: {
            ...currentUnsafeHybridInternals.value,
            [field as keyof StemTuningUnsafeHybridInternals]: value,
        },
    }, currentUnsafeModelInternals.value)
}

function setUnsafeHybridStageValue(stage: number, field: string, value: string | number | boolean) {
    setUnsafeHybridValue(hybridStageFieldKey(stage, field), value)
}

function setUnsafeBsRoformerValue(
    field: keyof StemTuningUnsafeBsRoformerInternals,
    value: string | string[] | boolean,
) {
    if (!preset.value) {
        return
    }

    preset.value.unsafeModelInternals = sanitizeUnsafeModelInternals({
        ...currentUnsafeModelInternals.value,
        bsRoformer: {
            ...currentUnsafeBsRoformerInternals.value,
            [field]: value,
        },
    }, currentUnsafeModelInternals.value)
}

function unsafeBsFieldValue(field: string): string | boolean {
    const value = currentUnsafeBsRoformerInternals.value[field as keyof StemTuningUnsafeBsRoformerInternals]
    if (Array.isArray(value)) {
        return value.join('\n')
    }

    return value as string | boolean
}

function unsafeBsTextFieldValue(field: string): string {
    const value = unsafeBsFieldValue(field)
    return typeof value === 'string' ? value : String(value)
}

function unsafeHybridFieldValue(field: string): string | number | boolean {
    return currentUnsafeHybridInternals.value[field as keyof StemTuningUnsafeHybridInternals] as string | number | boolean
}

function unsafeHybridStageValue(stage: number, field: string): string | number | boolean {
    return currentUnsafeHybridInternals.value[hybridStageFieldKey(stage, field)] as string | number | boolean
}

function onUnsafeHybridInput(field: string, event: Event, kind: Exclude<UnsafeFieldKind, 'string_array' | 'textarea'> = 'text') {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    if (kind === 'checkbox') {
        setUnsafeHybridValue(field, target.checked)
        return
    }

    if (kind === 'number') {
        setUnsafeHybridValue(field, Number(target.value))
        return
    }

    setUnsafeHybridValue(field, target.value)
}

function onUnsafeHybridStageInput(
    stage: number,
    field: string,
    event: Event,
    kind: Exclude<UnsafeFieldKind, 'string_array' | 'textarea'> = 'text',
) {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    if (kind === 'checkbox') {
        setUnsafeHybridStageValue(stage, field, target.checked)
        return
    }

    if (kind === 'number') {
        setUnsafeHybridStageValue(stage, field, Number(target.value))
        return
    }

    setUnsafeHybridStageValue(stage, field, target.value)
}

function onUnsafeBsRoformerInput(field: string, event: Event, kind: UnsafeFieldKind) {
    const target = event.target
    if (!(target instanceof HTMLInputElement) && !(target instanceof HTMLTextAreaElement)) {
        return
    }

    if (kind === 'checkbox' && target instanceof HTMLInputElement) {
        setUnsafeBsRoformerValue(field as keyof StemTuningUnsafeBsRoformerInternals, target.checked)
        return
    }

    if (kind === 'string_array') {
        setUnsafeBsRoformerValue(field as keyof StemTuningUnsafeBsRoformerInternals, parseUnsafeStringArray(target.value))
        return
    }

    setUnsafeBsRoformerValue(field as keyof StemTuningUnsafeBsRoformerInternals, target.value)
}

function onProcessorConfigInput(
    field: Exclude<keyof StemTuningProcessorConfig, 'bypass'>,
    event: Event,
) {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    clearProcessorBypassForActiveEdit()
    setProcessorConfigValue(field, Number(target.value))
}

function onStemRewriteConfigInput(
    field: keyof StemTuningStemRewriteConfig,
    event: Event,
) {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    setStemRewriteConfigValue(field, Number(target.value))
}

function onVocalsRewriteConfigInput(
    field: Exclude<keyof StemTuningVocalsRewriteConfig, 'candidateMode' | 'taskMode'>,
    event: Event,
) {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    setVocalsRewriteConfigValue(field, Number(target.value))
}

function countLayerAdjustments(key: string): number {
    const layer = currentLayerStates.value[key]
    if (!layer) {
        return 0
    }

    const values = [
        layer.gainDb ?? 0,
        layer.transientDb ?? 0,
        layer.bodyDb ?? 0,
        layer.harmonicDuckDb ?? 0,
        layer.bleedDuckDb ?? 0,
    ]

    return values.filter((value) => Math.abs(value) >= 0.05).length
}

function setEditorLayer(key: string) {
    if (!editableSources.value.some((source) => source.key === key)) {
        return
    }

    editorLayerKey.value = key
}

function setPreviewSourceWithoutPresetSync(key: string) {
    if (!sources.value.some((source) => source.key === key)) {
        return
    }

    setSelectedSourceKey(key, { syncPreset: false })
}

function previewVocalsSource(preferRendered = false) {
    const preferredKey = preferRendered ? renderVocalsSourceKey.value : vocalsSourceKey.value
    const fallbackKey = preferRendered ? vocalsSourceKey.value : renderVocalsSourceKey.value
    const nextKey = preferredKey || fallbackKey
    if (!nextKey) {
        return
    }

    setPreviewSourceWithoutPresetSync(nextKey)
}

function soloVocalsForNextRender() {
    if (!vocalsSourceKey.value) {
        return
    }

    soloLayer(vocalsSourceKey.value)
}

function setLayerEnabled(key: string, enabled: boolean) {
    if (!preset.value) {
        return
    }

    const current = currentLayerStates.value[key] ?? defaultLayerState(enabled)
    const next = {
        ...preset.value.layerStates,
        [key]: {
            ...current,
            enabled,
        },
    }
    preset.value.layerStates = normalizeLayerStates(next, editableSources.value, preset.value.sourceKey)
}

function setLayerGain(key: string, gainDb: number) {
    if (!preset.value) {
        return
    }

    const current = currentLayerStates.value[key] ?? defaultLayerState(isLayerEnabled(key))
    const next = {
        ...preset.value.layerStates,
        [key]: {
            ...current,
            gainDb: clamp(gainDb, -48, 12),
        },
    }
    preset.value.layerStates = normalizeLayerStates(next, editableSources.value, preset.value.sourceKey)
}

function onLayerGainInput(key: string, event: Event) {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    setLayerGain(key, Number(target.value))
}

function setLayerProcessorValue(
    key: string,
    field: 'transientDb' | 'bodyDb' | 'harmonicDuckDb' | 'bleedDuckDb',
    value: number,
) {
    if (!preset.value) {
        return
    }

    clearProcessorBypassForActiveEdit()

    const current = currentLayerStates.value[key] ?? defaultLayerState(isLayerEnabled(key))
    const ranges = {
        transientDb: [-18, 18],
        bodyDb: [-18, 18],
        harmonicDuckDb: [-18, 0],
        bleedDuckDb: [-18, 0],
    } as const
    const [min, max] = ranges[field]

    const next = {
        ...preset.value.layerStates,
        [key]: {
            ...current,
            [field]: clamp(value, min, max),
        },
    }
    preset.value.layerStates = normalizeLayerStates(next, editableSources.value, preset.value.sourceKey)
}

function onLayerProcessorInput(
    key: string,
    field: 'transientDb' | 'bodyDb' | 'harmonicDuckDb' | 'bleedDuckDb',
    event: Event,
) {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    setLayerProcessorValue(key, field, Number(target.value))
}

function toggleLayer(key: string) {
    setLayerEnabled(key, !isLayerEnabled(key))
}

function resetLayerControls(key: string) {
    if (!preset.value) {
        return
    }

    const current = currentLayerStates.value[key] ?? defaultLayerState(isLayerEnabled(key))
    preset.value.layerStates = normalizeLayerStates({
        ...preset.value.layerStates,
        [key]: {
            ...current,
            gainDb: 0,
            transientDb: 0,
            bodyDb: 0,
            harmonicDuckDb: 0,
            bleedDuckDb: 0,
        },
    }, editableSources.value, preset.value.sourceKey)
}

function focusLayer(key: string) {
    if (!preset.value) {
        return
    }

    selectedSourceKey.value = key
    preset.value.sourceKey = key
    const current = currentLayerStates.value[key] ?? defaultLayerState(true)
    preset.value.layerStates = normalizeLayerStates({
        ...preset.value.layerStates,
        [key]: {
            ...current,
            enabled: true,
        },
    }, editableSources.value, key)
}

function soloLayer(key: string) {
    if (!preset.value) {
        return
    }

    selectedSourceKey.value = key
    preset.value.sourceKey = key
    preset.value.layerStates = editableSources.value.reduce<Record<string, StemTuningLayerState>>((acc, source) => {
        const current = currentLayerStates.value[source.key] ?? defaultLayerState(source.key === key)
        acc[source.key] = {
            ...current,
            enabled: source.key === key,
        }
        return acc
    }, {})
}

function applyMonitorVolume() {
    if (mediaElementRef.value) {
        mediaElementRef.value.volume = clamp(monitorVolume.value, 0, 1)
    }
}

function handleMediaReady(media: HTMLAudioElement) {
    mediaElementRef.value = media
    applyMonitorVolume()
    tuningLabDebug('handleMediaReady', {
        mediaSrc: media.currentSrc || media.src,
    })
}

function handleTimeUpdate(timeSec: number) {
    currentTime.value = timeSec
}

function handleAnalysisFrame(payload: { waveform: number[]; spectrum: number[] }) {
    liveMonitorWaveform.value = payload.waveform
    liveMonitorSpectrum.value = payload.spectrum
}

async function loadSession(options?: {
    preservePreset?: StemTuningPreset | null
    preserveSelectedSourceKey?: string | null
    preserveEditorLayerKey?: string | null
    preserveStemBrowserScrollTop?: number | null
    quiet?: boolean
}) {
    const quiet = options?.quiet ?? false
    tuningLabDebug('loadSession:start', {
        quiet,
        preserveSelectedSourceKey: options?.preserveSelectedSourceKey ?? null,
        preserveEditorLayerKey: options?.preserveEditorLayerKey ?? null,
    })

    if (!quiet) {
        loading.value = true
    }
    stopAutoRenderTimer()
    if (!quiet) {
        errorMessage.value = ''
        canBuildStems.value = false
        tuningLabDebug('loadSession:reset-time')
        currentTime.value = 0
    }

    try {
        const previousTrackId = session.value?.track?.id ?? null
        const currentPresetSnapshot = options?.preservePreset !== undefined
            ? options.preservePreset
            : clonePresetState(preset.value)
        const currentSelectedSourceSnapshot = options?.preserveSelectedSourceKey !== undefined
            ? options.preserveSelectedSourceKey
            : selectedSourceKey.value
        const currentEditorLayerSnapshot = options?.preserveEditorLayerKey !== undefined
            ? options.preserveEditorLayerKey
            : editorLayerKey.value
        const currentStemBrowserScrollTop = options?.preserveStemBrowserScrollTop !== undefined
            ? options.preserveStemBrowserScrollTop
            : (stemBrowserRef.value?.scrollTop ?? null)
        const response = await api.getStemTuningLab(trackId.value)
        session.value = response

        const availableSources = response.lab.sources ?? []
        const availableEditableSources = availableSources.filter((source) => source.editable)
        const fallbackPreset = buildFallbackPreset(response.lab.default_source_key, availableSources)
        const basePreset = sanitizePreset(response.lab.default_preset ?? {}, fallbackPreset, availableSources)
        const shouldPreserveCurrentPreset = (
            currentPresetSnapshot
            && (previousTrackId === null || previousTrackId === response.track.id)
        )
        const nextPreset = shouldPreserveCurrentPreset
            ? sanitizePreset(currentPresetSnapshot, basePreset, availableSources)
            : basePreset

        const preferredSelectedSourceKey = currentSelectedSourceSnapshot
        const resolvedSelectedSourceKey = (
            preferredSelectedSourceKey
            && availableSources.some((source) => source.key === preferredSelectedSourceKey)
        )
            ? preferredSelectedSourceKey
            : nextPreset.sourceKey
        const preferredEditorLayerKey = currentEditorLayerSnapshot
        const resolvedEditorLayerKey = (
            preferredEditorLayerKey
            && availableEditableSources.some((source) => source.key === preferredEditorLayerKey)
        )
            ? preferredEditorLayerKey
            : (
                availableEditableSources.some((source) => source.key === nextPreset.sourceKey)
                    ? nextPreset.sourceKey
                    : (availableEditableSources[0]?.key ?? '')
            )

        await withSuppressedPresetWatch(async () => {
            preset.value = nextPreset
            setSelectedSourceKey(resolvedSelectedSourceKey, { syncPreset: false })
            editorLayerKey.value = resolvedEditorLayerKey
            importBuffer.value = JSON.stringify(nextPreset, null, 2)
            loadActiveModeConfigDraft()
            loadUnsafeModelInternalsDraft()
        })
        await nextTick()
        if (stemBrowserRef.value && typeof currentStemBrowserScrollTop === 'number') {
            stemBrowserRef.value.scrollTop = currentStemBrowserScrollTop
        }
        tuningLabDebug('loadSession:ready', {
            quiet,
            sourceCount: availableSources.length,
            resolvedSelectedSourceKey,
            resolvedEditorLayerKey,
        })
    } catch (error) {
        console.error('Failed to load tuning lab:', error)
        if (quiet) {
            toast.error(extractApiErrorMessage(error, 'Failed to refresh tuning lab session'))
            return
        }
        errorMessage.value = extractApiErrorMessage(error, 'Failed to load tuning lab')
        canBuildStems.value = extractApiErrorBoolean(error, 'buildable', false)
        session.value = null
        preset.value = null
        selectedSourceKey.value = ''
        editorLayerKey.value = ''
    } finally {
        if (!quiet) {
            loading.value = false
        }
    }
}

function stopBuildPolling() {
    if (buildPollingTimer.value !== null) {
        window.clearTimeout(buildPollingTimer.value)
        buildPollingTimer.value = null
    }
}

async function pollBuildStatus(taskId: number) {
    try {
        const status = await api.getStemTuningBuildStatus(trackId.value, taskId)
        buildTask.value = status

        if (status.status === 'completed') {
            stopBuildPolling()
            await loadSession()
            if (session.value) {
                toast.success('Stem sources were built in the container. The tuning lab is now ready.')
            } else {
                toast.error('Stem build finished, but the lab still has no stem session.')
            }
            return
        }

        if (status.status === 'failed') {
            stopBuildPolling()
            toast.error(status.error_message || 'Stem source build failed')
            return
        }

        buildPollingTimer.value = window.setTimeout(() => {
            void pollBuildStatus(taskId)
        }, 2500)
    } catch (error) {
        stopBuildPolling()
        console.error('Failed to poll stem source build status:', error)
        toast.error(extractApiErrorMessage(error, 'Failed to poll stem build status'))
    }
}

async function submitStemBuild() {
    if (buildBusy.value) {
        return
    }

    buildSubmitting.value = true
    stopBuildPolling()

    try {
        const task = await api.buildStemTuningSources(trackId.value)
        buildTask.value = task
        toast.info('Stem source build queued in the ML worker container')
        await pollBuildStatus(task.task_id)
    } catch (error) {
        console.error('Failed to queue stem source build:', error)
        toast.error(extractApiErrorMessage(error, 'Failed to queue stem source build'))
    } finally {
        buildSubmitting.value = false
    }
}

function stopRenderPolling() {
    if (renderPollingTimer.value !== null) {
        window.clearTimeout(renderPollingTimer.value)
        renderPollingTimer.value = null
    }
}

function stopAutoRenderTimer() {
    if (autoRenderTimer.value !== null) {
        window.clearTimeout(autoRenderTimer.value)
        autoRenderTimer.value = null
    }
}

function queueAutoRender() {
    if (!autoRenderEnabled.value || !session.value || !preset.value) {
        return
    }

    autoRenderQueued.value = true
    stopAutoRenderTimer()
    if (renderBusy.value) {
        void requestRenderSupersede('Superseded by newer tuning preset changes')
        return
    }
    autoRenderTimer.value = window.setTimeout(() => {
        void flushAutoRenderQueue()
    }, AUTO_RENDER_DELAY_MS)
}

async function requestRenderSupersede(reason: string) {
    const activeTaskId = renderTask.value?.task_id ?? null
    const activeStatus = renderTask.value?.status ?? ''
    if (!activeTaskId || !['pending', 'processing'].includes(activeStatus)) {
        return
    }

    if (renderCancelRequestedTaskId.value === activeTaskId) {
        return
    }

    renderCancelRequestedTaskId.value = activeTaskId
    try {
        const status = await api.cancelStemTuningRender(trackId.value, activeTaskId, reason)
        renderTask.value = status
    } catch (error) {
        renderCancelRequestedTaskId.value = null
        console.error('Failed to request stem tuning render cancellation:', error)
    }
}

async function flushAutoRenderQueue() {
    stopAutoRenderTimer()

    if (!autoRenderEnabled.value || !preset.value || loading.value || buildBusy.value) {
        return
    }

    if (vocalsRewriteCleanupConflict.value) {
        autoRenderQueued.value = false
        return
    }

    const currentSignature = presetSignature(preset.value)
    if (!currentSignature || currentSignature === lastCompletedRenderSignature) {
        autoRenderQueued.value = false
        return
    }

    if (renderBusy.value) {
        autoRenderQueued.value = true
        return
    }

    autoRenderQueued.value = false
    await submitContainerRender('auto')
}

async function pollRenderStatus(taskId: number, options: {
    submittedSignature: string
    manual: boolean
}) {
    try {
        const status = await api.getStemTuningRenderStatus(trackId.value, taskId)
        renderTask.value = status

        if (status.status === 'completed') {
            renderCancelRequestedTaskId.value = null
            stopRenderPolling()
            const latestPreset = clonePresetState(preset.value)
            const latestSignature = presetSignature(latestPreset)
            const supersededByNewerPreset = Boolean(
                autoRenderEnabled.value
                && latestSignature
                && latestSignature !== options.submittedSignature,
            )

            if (supersededByNewerPreset) {
                renderTask.value = {
                    ...status,
                    message: 'This render finished, but newer edits already superseded it. Skipping stale reload and rendering the latest preset next.',
                }
                autoRenderQueued.value = true
                queueAutoRender()
                return
            }

            const preservedEditorLayerKey = editorLayerKey.value
            const preservedStemBrowserScrollTop = stemBrowserRef.value?.scrollTop ?? null
            await loadSession({
                preservePreset: latestPreset,
                preserveSelectedSourceKey: selectedSourceKey.value,
                preserveEditorLayerKey: preservedEditorLayerKey,
                preserveStemBrowserScrollTop: preservedStemBrowserScrollTop,
                quiet: true,
            })
            lastCompletedRenderSignature = options.submittedSignature

            if (options.manual) {
                toast.success('Container render finished. New stem files are now available in the lab.')
            }

            return
        }

        if (status.status === 'cancelled') {
            renderCancelRequestedTaskId.value = null
            stopRenderPolling()
            if (autoRenderEnabled.value && autoRenderQueued.value) {
                queueAutoRender()
            }
            return
        }

        if (status.status === 'failed') {
            renderCancelRequestedTaskId.value = null
            stopRenderPolling()
            autoRenderQueued.value = false
            toast.error(status.error_message || 'Stem render failed')
            return
        }

        renderPollingTimer.value = window.setTimeout(() => {
            void pollRenderStatus(taskId, options)
        }, RENDER_STATUS_POLL_MS)
    } catch (error) {
        stopRenderPolling()
        console.error('Failed to poll stem tuning render status:', error)
        toast.error(extractApiErrorMessage(error, 'Failed to poll render status'))
    }
}

async function submitContainerRender(mode: 'manual' | 'auto' = 'manual') {
    if (!preset.value || renderBusy.value || buildBusy.value) {
        return
    }

    if (vocalsRewriteCleanupConflict.value) {
        if (mode === 'manual') {
            toast.error('Vocals Rewrite cannot run with both Subtractive strength and Hard mask threshold enabled at the same time')
        }
        return
    }

    renderSubmitting.value = true
    stopRenderPolling()

    try {
        renderCancelRequestedTaskId.value = null
        const presetSnapshot = clonePresetState(preset.value)
        const submittedSignature = presetSignature(presetSnapshot)
        const task = await api.renderStemTuningLab(trackId.value, JSON.parse(JSON.stringify(presetSnapshot)) as Record<string, unknown>)
        renderTask.value = task
        if (mode === 'manual') {
            toast.info('Stem render queued in the ML worker container')
        }
        await pollRenderStatus(task.task_id, {
            submittedSignature,
            manual: mode === 'manual',
        })
    } catch (error) {
        console.error('Failed to queue stem tuning render:', error)
        toast.error(extractApiErrorMessage(error, 'Failed to queue container render'))
    } finally {
        renderSubmitting.value = false
        if (autoRenderEnabled.value && autoRenderQueued.value && !buildBusy.value) {
            queueAutoRender()
        }
    }
}

function resetPreset() {
    if (!session.value) {
        return
    }

    const fallbackPreset = buildFallbackPreset(
        session.value.lab.default_source_key,
        session.value.lab.sources,
    )
    const nextPreset = sanitizePreset(session.value.lab.default_preset ?? {}, fallbackPreset, session.value.lab.sources)
    preset.value = nextPreset
    selectedSourceKey.value = nextPreset.sourceKey
    replaceImportBufferSilently(JSON.stringify(nextPreset, null, 2))
    importBufferStatusTone.value = 'muted'
    importBufferStatusMessage.value = 'Preset editor reloaded from session defaults.'
    loadActiveModeConfigDraft()
    loadUnsafeModelInternalsDraft()
    toast.info('Preset reset to session defaults')
}

function loadCurrentJsonIntoEditor() {
    replaceImportBufferSilently(displayPresetJson.value)
    importBufferStatusTone.value = 'muted'
    importBufferStatusMessage.value = 'Editor reloaded from current preset.'
}

function importPresetFromTextarea(options?: { silent?: boolean }) {
    if (!session.value) {
        return false
    }

    try {
        const parsed = JSON.parse(importBuffer.value || '{}') as unknown
        const fallbackPreset = buildFallbackPreset(
            session.value.lab.default_source_key,
            session.value.lab.sources,
        )
        const nextPreset = sanitizePreset(parsed, fallbackPreset, session.value.lab.sources)
        if (presetSignature(nextPreset) === presetSignature(preset.value)) {
            importBufferStatusTone.value = 'success'
            importBufferStatusMessage.value = 'Preset JSON already matches current state.'
            return true
        }
        preset.value = nextPreset
        selectedSourceKey.value = nextPreset.sourceKey
        loadActiveModeConfigDraft()
        loadUnsafeModelInternalsDraft()
        importBufferStatusTone.value = 'success'
        importBufferStatusMessage.value = 'Preset JSON applied automatically.'
        if (!options?.silent) {
            toast.success('Preset JSON applied')
        }
        return true
    } catch (error) {
        console.error('Failed to import tuning preset:', error)
        importBufferStatusTone.value = 'error'
        importBufferStatusMessage.value = 'Preset JSON is invalid. Nothing applied.'
        if (!options?.silent) {
            toast.error('Preset JSON is invalid')
        }
        return false
    }
}

async function copyPresetJson() {
    try {
        await navigator.clipboard.writeText(displayPresetJson.value)
        toast.success('Preset JSON copied')
    } catch (error) {
        console.error('Failed to copy tuning preset:', error)
        toast.error('Clipboard write failed')
    }
}

function downloadPresetJson() {
    if (!displayPresetJson.value) {
        return
    }

    const blob = new Blob([displayPresetJson.value], { type: 'application/json;charset=utf-8' })
    const href = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = href
    anchor.download = `track-${trackId.value}-tuning-preset.json`
    anchor.click()
    URL.revokeObjectURL(href)
}

function goBackToTrack() {
    router.push({ name: 'track', params: { id: trackId.value } })
}

watch(
    () => trackId.value,
    () => {
        void loadSession()
    },
)

watch(selectedSourceKey, (nextKey) => {
    if (!sources.value.some((source) => source.key === nextKey)) {
        return
    }

    tuningLabDebug('watch:selectedSourceKey', {
        nextKey,
        skipPresetSync: suppressSelectedSourcePresetSync,
    })

    const skipPresetSync = suppressSelectedSourcePresetSync
    const skipEditorSync = suppressSelectedSourceEditorSync
    suppressSelectedSourcePresetSync = false
    suppressSelectedSourceEditorSync = false
    const isEditable = editableSources.value.some((source) => source.key === nextKey)

    if (preset.value && !skipPresetSync && isEditable) {
        preset.value.sourceKey = nextKey
        const current = currentLayerStates.value[nextKey] ?? defaultLayerState(true)
        preset.value.layerStates = normalizeLayerStates({
            ...preset.value.layerStates,
            [nextKey]: {
                ...current,
                enabled: true,
            },
        }, editableSources.value, nextKey)
        preset.value.focusWindow.targetLayerKey = nextKey
    }
    if (isEditable && !skipEditorSync) {
        editorLayerKey.value = nextKey
    }
})

watch(monitorVolume, () => {
    applyMonitorVolume()
})

watch(autoRenderEnabled, (enabled) => {
    if (!enabled) {
        stopAutoRenderTimer()
        autoRenderQueued.value = false
        return
    }

    queueAutoRender()
})

watch(currentRenderMode, () => {
    loadActiveModeConfigDraft()
}, { immediate: true })

watch(currentUnsafeModelInternals, () => {
    loadUnsafeModelInternalsDraft()
}, { immediate: true, deep: true })

watch(importBuffer, () => {
    if (suppressImportBufferWatch) {
        return
    }
    scheduleImportBufferAutoApply()
})

watch(activeModeConfigDraft, () => {
    if (suppressActiveModeConfigDraftWatch) {
        return
    }
    scheduleActiveModeConfigDraftAutoApply()
})

watch(unsafeModelInternalsDraft, () => {
    if (suppressUnsafeModelInternalsDraftWatch) {
        return
    }
    scheduleUnsafeModelInternalsDraftAutoApply()
})

watch(preset, () => {
    if (suppressPresetWatch) {
        return
    }
    queueAutoRender()
}, { deep: true })

onMounted(() => {
    void loadSession()
})

onUnmounted(() => {
    stopBuildPolling()
    stopRenderPolling()
    stopAutoRenderTimer()
    stopImportBufferApplyTimer()
    stopActiveModeConfigDraftApplyTimer()
    stopUnsafeModelInternalsDraftApplyTimer()
})
</script>

<template>
    <div class="mx-auto flex w-full max-w-[1560px] flex-col gap-6 px-4 pb-16 pt-6 lg:px-8">
        <div class="flex flex-col gap-4 rounded-[2rem] border border-slate-200 bg-white/90 px-6 py-6 shadow-[0_30px_120px_rgba(15,23,42,0.08)] lg:flex-row lg:items-end lg:justify-between">
            <div class="max-w-4xl">
                <p class="section-kicker">Tuning Lab</p>
                <h1 class="mt-3 text-3xl font-semibold tracking-tight text-slate-950">
                    {{ lab?.title ?? `Track ${trackId} Tuning Lab` }}
                </h1>
                <p class="mt-3 text-sm leading-6 text-slate-600">
                    {{ lab?.description ?? 'Container-first stem preset authoring surface.' }}
                </p>
                <div class="mt-4 flex flex-wrap gap-3 text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                    <span class="rounded-full border border-slate-200 bg-slate-50 px-3 py-2">
                        Track {{ track?.id ?? trackId }}
                    </span>
                    <span class="rounded-full border border-slate-200 bg-slate-50 px-3 py-2">
                        Stem key: {{ lab?.stem_key ?? 'mix' }}
                    </span>
                    <span class="rounded-full border border-slate-200 bg-slate-50 px-3 py-2">
                        Enabled stems: {{ enabledLayerCount }}
                    </span>
                </div>
            </div>

            <div class="flex flex-wrap gap-3">
                <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="goBackToTrack">
                    Back to track
                </button>
                <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="() => loadSession()">
                    Reload lab
                </button>
            </div>
        </div>

        <section v-if="session && preset && selectedSource" class="surface-card !p-6">
            <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                <div class="max-w-4xl">
                    <p class="section-kicker">Container Render</p>
                    <h2 class="mt-3 text-2xl font-semibold tracking-tight text-slate-950">Write new stem files from the current preset</h2>
                    <p class="mt-2 text-sm leading-6 text-slate-600">
                        This action sends the current stem-layer preset to the ML worker container, writes a new rendered stem directory,
                        and feeds those files back into this lab. Raw stem controls below are the only values that drive the next rendered stem files.
                    </p>
                </div>

                <div class="flex flex-wrap gap-3">
                    <button
                        type="button"
                        class="action-primary !rounded-[1.1rem] !px-5 !py-3"
                        :disabled="!preset || renderBusy || vocalsRewriteCleanupConflict"
                        @click="() => submitContainerRender()"
                    >
                        {{ renderBusy ? 'Rendering in container...' : 'Render stem files in container' }}
                    </button>
                </div>
            </div>

            <div class="mt-5 grid gap-4 md:grid-cols-4">
                <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                    <div class="field-label">Last task</div>
                    <div class="mt-2 text-lg font-semibold text-slate-950">
                        {{ renderTask?.task_id ?? 'No task yet' }}
                    </div>
                </div>
                <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                    <div class="field-label">Status</div>
                    <div class="mt-2 text-lg font-semibold text-slate-950">
                        {{ renderTask?.status ?? 'idle' }}
                    </div>
                </div>
                <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                    <div class="field-label">Progress</div>
                    <div class="mt-2 text-lg font-semibold text-slate-950">
                        {{ renderProgressPercent }}%
                    </div>
                </div>
                <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                    <div class="field-label">Render loop</div>
                    <div class="mt-2 text-sm leading-6 text-slate-600">
                        Queue task -> ML worker render -> new stem files -> session reload
                    </div>
                </div>
            </div>

            <div class="mt-5 rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <div class="field-label">Worker status</div>
                        <div class="mt-2 text-sm leading-6 text-slate-700">
                            {{ renderTask?.message ?? 'Waiting for the next container render task.' }}
                        </div>
                    </div>

                    <div class="w-full lg:max-w-md">
                        <div class="h-2 overflow-hidden rounded-full bg-slate-200">
                            <div
                                class="h-full rounded-full bg-sky-500 transition-all duration-300"
                                :style="{ width: `${renderProgressPercent}%` }"
                            ></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mt-5 grid gap-4 md:grid-cols-2">
                <label class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                    <div class="flex items-center justify-between gap-3">
                        <div>
                            <div class="field-label">Debounced auto-render</div>
                            <div class="mt-1 text-sm text-slate-600">
                                After you stop moving stem controls for about {{ (AUTO_RENDER_DELAY_MS / 1000).toFixed(1) }}s,
                                the current preset is sent to the container automatically.
                            </div>
                        </div>
                        <input v-model="autoRenderEnabled" type="checkbox" class="h-4 w-4 rounded border-slate-300 text-sky-500 focus:ring-sky-200">
                    </div>
                </label>

                <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                    <div class="field-label">Auto-render status</div>
                    <div class="mt-2 text-sm leading-6 text-slate-600">
                        <span v-if="!autoRenderActive">Manual render only.</span>
                        <span v-else-if="vocalsRewriteCleanupConflict">Fix the Vocals Rewrite cleanup conflict before the next container render can be queued.</span>
                        <span v-else-if="renderBusy">Container is rendering now.</span>
                        <span v-else-if="autoRenderQueued">Waiting for the debounce window, then it will render.</span>
                        <span v-else>Idle. The next real stem change will queue a render.</span>
                    </div>
                </div>
            </div>

            <div
                v-if="vocalsRewriteCleanupConflict"
                class="mt-5 rounded-[1.2rem] border border-rose-200 bg-rose-50 px-4 py-4 text-sm leading-6 text-rose-700"
            >
                <span class="font-semibold">Vocals Rewrite cleanup conflict:</span>
                the current worker does not allow <span class="font-semibold">Subtractive strength</span> and
                <span class="font-semibold">Hard mask threshold</span> to be active together. Leave one family at zero before rendering.
            </div>

            <p v-if="renderTask?.error_message" class="mt-4 text-sm leading-6 text-rose-600">
                {{ renderTask.error_message }}
            </p>

            <details class="surface-card mt-5 !p-6">
                <summary class="cursor-pointer list-none">
                    <div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <p class="section-kicker">Container Log</p>
                            <h2 class="mt-3 text-2xl font-semibold tracking-tight text-slate-950">Render task progress and worker messages</h2>
                            <p class="mt-2 text-sm leading-6 text-slate-600">
                                This is the structured log stream from the container render task. Use it when you need to see exactly what the worker is doing.
                            </p>
                        </div>
                        <div class="rounded-[1.1rem] border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-600">
                            Hidden by default on purpose. The main workflow is the stem editor above.
                        </div>
                    </div>
                </summary>

                <div class="mt-5 grid gap-6 xl:grid-cols-[260px,minmax(0,1fr)]">
                    <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                        <div class="field-label">Log entries</div>
                        <div class="mt-2 text-lg font-semibold text-slate-950">
                            {{ renderLogEntries.length }}
                        </div>
                        <p class="mt-2 text-sm leading-6 text-slate-600">
                            Live stage-by-stage messages from the current or latest render task.
                        </p>
                    </div>

                    <div class="max-h-72 space-y-2 overflow-y-auto rounded-[1.2rem] border border-slate-200 bg-white p-4 font-mono text-xs leading-6 text-slate-700">
                        <div v-if="renderLogEntries.length === 0" class="rounded-[0.9rem] border border-slate-200 bg-slate-50 px-3 py-2 text-slate-500">
                            No worker logs yet. Queue a render to see stage-by-stage progress here.
                        </div>
                        <div
                            v-for="(entry, index) in renderLogEntries"
                            :key="`${entry.timestamp}-${entry.stage ?? 'stage'}-${index}`"
                            class="rounded-[0.9rem] border border-slate-200 bg-slate-50 px-3 py-2"
                        >
                            <div class="flex flex-wrap items-center gap-x-3 gap-y-1 text-slate-500">
                                <span>{{ entry.timestamp }}</span>
                                <span class="font-semibold text-sky-700">{{ entry.progress }}%</span>
                                <span class="uppercase tracking-[0.14em] text-slate-400">{{ entry.level }}</span>
                                <span v-if="entry.stage" class="uppercase tracking-[0.14em] text-violet-700">{{ entry.stage }}</span>
                            </div>
                            <div class="mt-1 text-slate-900">{{ entry.message }}</div>
                        </div>
                    </div>
                </div>
            </details>
        </section>

        <section v-if="loading" class="surface-card !p-8">
            <div class="text-sm text-slate-600">Loading tuning lab session...</div>
        </section>

        <section v-else-if="errorMessage" class="surface-card !p-8">
            <div class="max-w-3xl">
                <h2 class="text-xl font-semibold text-slate-950">Tuning lab unavailable</h2>
                <p class="mt-3 text-sm leading-6 text-slate-600">{{ errorMessage }}</p>
                <div class="mt-5 flex flex-wrap gap-3">
                    <button
                        v-if="canBuildStems"
                        type="button"
                        class="action-primary !rounded-[1.1rem] !px-4 !py-3"
                        :disabled="buildBusy"
                        @click="submitStemBuild"
                    >
                        {{ buildBusy ? 'Building stems in container...' : 'Build stems in container' }}
                    </button>
                    <button type="button" class="action-primary !rounded-[1.1rem] !px-4 !py-3" @click="() => loadSession()">
                        Retry
                    </button>
                    <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="goBackToTrack">
                        Back to track
                    </button>
                </div>

                <div v-if="canBuildStems" class="mt-6 grid gap-4 md:grid-cols-3">
                    <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                        <div class="field-label">Build task</div>
                        <div class="mt-2 text-lg font-semibold text-slate-950">
                            {{ buildTask?.task_id ?? 'No task yet' }}
                        </div>
                    </div>
                    <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                        <div class="field-label">Status</div>
                        <div class="mt-2 text-lg font-semibold text-slate-950">
                            {{ buildTask?.status ?? 'idle' }}
                        </div>
                    </div>
                    <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                        <div class="field-label">Build loop</div>
                        <div class="mt-2 text-sm leading-6 text-slate-600">
                            Source audio -> 6-stem separation in container -> playback manifest -> stem-aware tuning lab
                        </div>
                    </div>
                </div>

                <p v-if="buildTask?.error_message" class="mt-4 text-sm leading-6 text-rose-600">
                    {{ buildTask.error_message }}
                </p>
            </div>
        </section>

        <template v-else-if="session && preset && selectedSource">
            <div class="grid gap-6 xl:grid-cols-[1.4fr_0.6fr]">
                <section class="surface-card !overflow-hidden !bg-slate-950 !p-6 text-white">
                    <div class="flex flex-col gap-5">
                        <div class="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                            <div>
                                <p class="section-kicker !text-sky-300/90">Waveform Preview</p>
                                <h2 class="mt-3 text-2xl font-semibold tracking-tight text-white">{{ selectedSource.label }}</h2>
                                <p class="mt-2 text-sm leading-6 text-slate-300">
                                    {{ selectedSource.description }}
                                </p>
                            </div>

                            <div class="grid gap-3 sm:grid-cols-2">
                                <div class="rounded-[1.2rem] border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
                                    <div class="field-label !text-slate-400">Waveform source</div>
                                    <div class="mt-2 text-base font-semibold text-white">{{ selectedSource.file_name }}</div>
                                </div>
                                <div class="rounded-[1.2rem] border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-300">
                                    <div class="field-label !text-slate-400">Session duration</div>
                                    <div class="mt-2 text-base font-semibold text-white">{{ formatTime(track?.duration ?? 0) }}</div>
                                </div>
                            </div>
                        </div>

                        <StemTuningWaveform
                            ref="waveformTransportRef"
                            :source-url="selectedSourceStreamUrl"
                            :waveform-url="selectedSourceWaveformUrl"
                            :source-label="selectedSource.label"
                            :monitor-volume="monitorVolume"
                            :duration="track?.duration"
                            :focus-windows="detectorWaveformWindows"
                            @media-ready="handleMediaReady"
                            @time-update="handleTimeUpdate"
                            @analysis-frame="handleAnalysisFrame"
                        />

                        <div class="rounded-[1.2rem] border border-white/10 bg-white/5 px-4 py-4">
                            <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
                                <div>
                                    <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">Player Volume</div>
                                    <div class="mt-2 text-sm text-slate-300">
                                        Controls the actual monitoring loudness of the player. Applies instantly.
                                    </div>
                                </div>

                                <div class="min-w-0 flex-1 lg:max-w-md">
                                    <div class="mb-2 flex items-center justify-between text-sm font-semibold text-white">
                                        <span>Monitor level</span>
                                        <span>{{ monitorVolumePercent }}%</span>
                                    </div>
                                    <input
                                        v-model.number="monitorVolume"
                                        type="range"
                                        min="0"
                                        max="1"
                                        step="0.01"
                                        class="slider-track w-full"
                                    >
                                </div>
                            </div>
                        </div>

                    </div>
                </section>

                <section class="surface-card !p-6">
                    <p class="section-kicker">Session</p>
                    <h2 class="mt-3 text-2xl font-semibold tracking-tight text-slate-950">Preview source and file info</h2>
                    <div class="mt-5 grid gap-3">
                        <label class="block">
                            <span class="field-label">Preview waveform source</span>
                            <select
                                v-model="selectedSourceKey"
                                class="mt-2 w-full rounded-[1rem] border border-slate-200 bg-white px-4 py-3 text-sm text-slate-900 shadow-sm outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                            >
                                <option
                                    v-for="source in sources"
                                    :key="source.key"
                                    :value="source.key"
                                >
                                    {{ source.label }}
                                </option>
                            </select>
                        </label>

                        <div class="grid gap-3 sm:grid-cols-2">
                            <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/90 px-4 py-3">
                                <div class="field-label">Current time</div>
                                <div class="mt-2 text-lg font-semibold text-slate-950">{{ formatTime(currentTime) }}</div>
                            </div>
                            <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/90 px-4 py-3">
                                <div class="field-label">File size</div>
                                <div class="mt-2 text-lg font-semibold text-slate-950">
                                    {{ formatFileSize(selectedSource.file_size_bytes) }}
                                </div>
                            </div>
                        </div>

                        <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/90 px-4 py-4 text-sm leading-6 text-slate-600">
                            <div><span class="font-semibold text-slate-900">Family:</span> {{ selectedSource.family }}</div>
                            <div class="mt-1"><span class="font-semibold text-slate-900">Kind:</span> {{ selectedSource.kind }}</div>
                            <div class="mt-1"><span class="font-semibold text-slate-900">Recommended:</span> {{ selectedSource.recommended ? 'Yes' : 'No' }}</div>
                        </div>

                        <div
                            v-if="stemRewritePreviewNoOp"
                            class="rounded-[1.2rem] border border-amber-200 bg-amber-50 px-4 py-4 text-sm leading-6 text-amber-950"
                        >
                            Stem Rewrite does not rewrite {{ selectedSource.label }} itself in the current worker branch.
                            Its rewrite knobs move melody energy between other/guitar/piano/drums and the generated melody stem.
                            If you preview {{ selectedSource.label }}, you will only hear changes from the generic post-DSP layer controls,
                            not from the rewrite knobs alone.
                        </div>

                        <div
                            v-if="vocalsRewritePreviewNoOp"
                            class="rounded-[1.2rem] border border-amber-200 bg-amber-50 px-4 py-4 text-sm leading-6 text-amber-950"
                        >
                            Vocals Rewrite rebuilds only the <span class="font-semibold">vocals</span> stem from the raw track audio.
                            If you preview {{ selectedSource.label }}, this mode can sound unchanged because non-vocal stems are preserved from the current base set.
                        </div>

                        <div
                            v-if="vocalsRewriteMixPreviewTrap"
                            class="rounded-[1.2rem] border border-rose-200 bg-rose-50 px-4 py-4 text-sm leading-6 text-rose-900"
                        >
                            <div class="font-semibold text-rose-950">
                                Current Vocals Rewrite preset is still a full mix render.
                            </div>
                            <div class="mt-2">
                                Right now <span class="font-semibold">{{ vocalsRewriteEnabledLayerCount }}</span> stems are enabled, so
                                <span class="font-semibold">render_mix</span> will come back as a полноценный sum mix, not an isolated rewritten vocal stem.
                                If you want to judge the rewritten vocal itself, preview <span class="font-semibold">stem_vocals</span> or <span class="font-semibold">render_vocals</span>.
                                If you want the next render to be vocals-only, solo vocals before rendering.
                            </div>
                            <div class="mt-4 flex flex-wrap gap-3">
                                <button
                                    v-if="vocalsSourceKey"
                                    type="button"
                                    class="action-secondary !rounded-full !px-4 !py-2.5"
                                    @click="previewVocalsSource(false)"
                                >
                                    Preview stem_vocals
                                </button>
                                <button
                                    v-if="renderVocalsSourceKey"
                                    type="button"
                                    class="action-secondary !rounded-full !px-4 !py-2.5"
                                    @click="previewVocalsSource(true)"
                                >
                                    Preview render_vocals
                                </button>
                                <button
                                    v-if="vocalsSourceKey"
                                    type="button"
                                    class="action-secondary !rounded-full !px-4 !py-2.5"
                                    @click="soloVocalsForNextRender"
                                >
                                    Solo vocals for next render
                                </button>
                            </div>
                        </div>

                        <div
                            v-if="detectorSegments"
                            class="rounded-[1.2rem] border border-sky-200 bg-sky-50/80 px-4 py-4 text-sm text-slate-700"
                        >
                            <div class="flex items-start justify-between gap-3">
                                <div>
                                    <div class="field-label !text-sky-700">Detector segments</div>
                                    <div class="mt-2 text-sm leading-6 text-slate-700">
                                        Auto-detected drop+vocal regions from the stronger detector. Gray overlay = parent segment, blue/red overlay = review window.
                                        Click any card below to jump playback straight into that window and verify the placement by ear.
                                    </div>
                                </div>
                                <div class="rounded-[1rem] border border-sky-200 bg-white px-3 py-2 text-right">
                                    <div class="text-[11px] font-semibold uppercase tracking-[0.14em] text-sky-700">Generated</div>
                                    <div class="mt-1 text-sm font-semibold text-slate-950">
                                        {{ detectorSegments.generated_at ? detectorSegments.generated_at.slice(0, 19).replace('T', ' ') : 'unknown' }}
                                    </div>
                                </div>
                            </div>

                            <div class="mt-4 grid gap-3 sm:grid-cols-2">
                                <div class="rounded-[1rem] border border-sky-200 bg-white px-4 py-3">
                                    <div class="field-label !text-sky-700">Model</div>
                                    <div class="mt-2 text-sm font-semibold text-slate-950">
                                        {{ detectorSegments.source.model_name || 'unknown' }}
                                    </div>
                                </div>
                                <div class="rounded-[1rem] border border-sky-200 bg-white px-4 py-3">
                                    <div class="field-label !text-sky-700">Frame / window</div>
                                    <div class="mt-2 text-sm font-semibold text-slate-950">
                                        {{ detectorSegments.analysis.frame_sec.toFixed(2) }}s / {{ detectorSegments.analysis.window_sec.toFixed(1) }}s
                                    </div>
                                </div>
                            </div>

                            <div class="mt-4 space-y-3">
                                <button
                                    v-for="segment in detectorReviewCards"
                                    :key="segment.segmentId"
                                    type="button"
                                    class="w-full rounded-[1.1rem] border px-4 py-4 text-left transition"
                                    :class="activeDetectorSegmentId === segment.segmentId
                                        ? 'border-sky-300 bg-white shadow-[0_18px_50px_rgba(14,165,233,0.12)]'
                                        : 'border-sky-100 bg-white/90 hover:border-sky-200 hover:bg-white'"
                                    @click="jumpToDetectorSegment(segment.segmentId)"
                                >
                                    <div class="flex items-start justify-between gap-3">
                                        <div class="min-w-0">
                                            <div class="flex flex-wrap items-center gap-2">
                                                <div class="text-sm font-semibold text-slate-950">{{ segment.label }}</div>
                                                <span
                                                    v-if="segment.selected"
                                                    class="rounded-full border border-sky-200 bg-sky-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-sky-700"
                                                >
                                                    selected
                                                </span>
                                                <span class="rounded-full border border-slate-200 bg-slate-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                                                    {{ segment.structureLabel }}
                                                </span>
                                            </div>
                                            <div class="mt-2 text-sm text-slate-700">
                                                Window {{ formatTime(segment.windowStartSec) }} - {{ formatTime(segment.windowEndSec) }}
                                            </div>
                                            <div class="mt-1 text-xs leading-5 text-slate-500">
                                                Parent segment {{ formatTime(segment.startSec) }} - {{ formatTime(segment.endSec) }}
                                            </div>
                                        </div>
                                        <div class="text-right">
                                            <div class="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Priority</div>
                                            <div class="mt-1 text-sm font-semibold text-slate-950">{{ segment.segmentPriority.toFixed(3) }}</div>
                                        </div>
                                    </div>

                                    <div class="mt-3 flex flex-wrap gap-2">
                                        <span
                                            v-for="kind in segment.kinds"
                                            :key="`${segment.segmentId}-${kind}`"
                                            class="rounded-full border border-sky-100 bg-sky-50 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.12em] text-sky-700"
                                        >
                                            {{ formatDetectorKindLabel(kind) }}
                                        </span>
                                    </div>
                                </button>
                            </div>
                        </div>

                        <div
                            v-else
                            class="rounded-[1.2rem] border border-slate-200 bg-slate-50/90 px-4 py-4 text-sm leading-6 text-slate-600"
                        >
                            Detector segments are not cached for this track yet. The UI can only show the stronger drop+vocal windows after
                            <span class="font-semibold text-slate-900">define-track-segments.py</span> writes a shared cache entry for this track.
                        </div>
                    </div>
                </section>
            </div>

            <StemTuningLiveMonitor
                :waveform="liveMonitorWaveform"
                :spectrum="liveMonitorSpectrum"
                :source-label="selectedSource.label"
            />

            <section class="overflow-hidden rounded-[2.2rem] border border-slate-200 bg-white shadow-[0_30px_120px_rgba(15,23,42,0.08)]">
                <div class="border-b border-slate-200 px-6 py-6">
                    <div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <p class="section-kicker">Files / Stems</p>
                            <h2 class="mt-3 text-3xl font-semibold tracking-tight text-slate-950">Stem layer controls</h2>
                            <p class="mt-2 text-sm leading-6 text-slate-600">
                                This is the main authoring surface: enable stems, solo them, choose the main source, and set raw per-stem values that are sent into the container render loop.
                            </p>
                        </div>
                        <div class="rounded-[1.1rem] border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-600">
                            <div class="field-label">Preset state</div>
                            <div class="mt-2 text-lg font-semibold text-slate-950">{{ enabledLayerCount }} stems enabled</div>
                        </div>
                    </div>

                    <div class="mt-4 rounded-[1rem] border border-sky-200 bg-sky-50 px-4 py-3 text-sm text-sky-900">
                        Raw stem controls below are container-only values. They do not change the current audio before a new render comes back. The waveform above only previews the currently selected source file.
                    </div>
                </div>

                <div class="grid min-h-[72vh] gap-0 xl:grid-cols-[360px_minmax(0,1fr)]">
                    <aside class="border-b border-slate-200 bg-slate-50/80 px-4 py-4 xl:border-b-0 xl:border-r xl:px-5 xl:py-5">
                        <div class="mb-4 flex items-center justify-between gap-3">
                            <div>
                                <div class="field-label">Stem browser</div>
                                <div class="mt-1 text-sm text-slate-600">Pick one stem and edit it on the full canvas to the right.</div>
                            </div>
                            <div class="rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
                                {{ layerCards.length }} total
                            </div>
                        </div>

                        <div ref="stemBrowserRef" class="flex flex-col gap-3 xl:max-h-[calc(72vh-2rem)] xl:overflow-y-auto xl:pr-1">
                            <article
                                v-for="card in layerCards"
                                :key="card.key"
                                role="button"
                                tabindex="0"
                                class="rounded-[1.35rem] border px-4 py-4 text-left transition"
                                :class="editorLayerCard?.key === card.key
                                    ? 'border-sky-300 bg-sky-50/70 shadow-[0_20px_60px_rgba(14,165,233,0.12)]'
                                    : 'border-slate-200 bg-white/80 hover:border-sky-200 hover:bg-sky-50/40'"
                                @click="setEditorLayer(card.key)"
                                @keydown.enter.prevent="setEditorLayer(card.key)"
                                @keydown.space.prevent="setEditorLayer(card.key)"
                            >
                            <div class="flex items-start justify-between gap-4">
                                <div class="min-w-0">
                                    <div class="flex flex-wrap items-center gap-2">
                                        <h3 class="text-base font-semibold text-slate-950">{{ card.label }}</h3>
                                        <span
                                            v-if="card.primary"
                                            class="rounded-full border border-sky-200 bg-sky-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-sky-700"
                                        >
                                            Main
                                        </span>
                                        <span
                                            v-if="card.recommended"
                                            class="rounded-full border border-emerald-200 bg-emerald-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-700"
                                        >
                                            Recommended
                                        </span>
                                        <span
                                            v-if="card.key.startsWith('render_')"
                                            class="rounded-full border border-violet-200 bg-violet-100 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-violet-700"
                                        >
                                            Rendered
                                        </span>
                                    </div>
                                    <p class="mt-2 text-sm leading-6 text-slate-600">{{ card.description }}</p>
                                </div>

                                <button
                                    type="button"
                                    class="!rounded-full !px-4 !py-2.5 text-sm font-semibold transition"
                                    :class="card.enabled
                                        ? 'bg-slate-950 text-white hover:bg-slate-800'
                                        : 'border border-slate-200 bg-white text-slate-700 hover:border-slate-300'"
                                    @click.stop="toggleLayer(card.key)"
                                >
                                    {{ card.enabled ? 'On' : 'Off' }}
                                </button>
                            </div>
                            <div class="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                                <div class="rounded-[1rem] border border-slate-200 bg-white/80 px-3 py-3">
                                    <div class="field-label">State</div>
                                    <div class="mt-1 text-sm font-semibold text-slate-950">
                                        {{ card.enabled ? 'Audible' : 'Muted' }}
                                    </div>
                                </div>
                                <div class="rounded-[1rem] border border-slate-200 bg-white/80 px-3 py-3">
                                    <div class="field-label">Level</div>
                                    <div class="mt-1 text-sm font-semibold text-slate-950">{{ card.gainDb.toFixed(1) }} dB</div>
                                </div>
                                <div class="rounded-[1rem] border border-slate-200 bg-white/80 px-3 py-3">
                                    <div class="field-label">Tweaks</div>
                                    <div class="mt-1 text-sm font-semibold text-slate-950">
                                        {{ card.adjustmentCount ? `${card.adjustmentCount} changed` : 'Flat' }}
                                    </div>
                                </div>
                                <div class="rounded-[1rem] border border-slate-200 bg-white/80 px-3 py-3">
                                    <div class="field-label">Kind</div>
                                    <div class="mt-1 text-sm font-semibold text-slate-950">{{ card.kind }}</div>
                                </div>
                            </div>
                            </article>
                        </div>
                    </aside>

                    <section
                            v-if="editorLayerCard"
                            class="bg-white px-5 py-5 md:px-6 md:py-6 xl:px-8 xl:py-8"
                        >
                            <div class="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
                                <div class="min-w-0">
                                    <p class="section-kicker">Editing Stem</p>
                                    <h3 class="mt-3 text-2xl font-semibold tracking-tight text-slate-950">
                                        {{ editorLayerCard.label }}
                                    </h3>
                                    <p class="mt-2 text-sm leading-6 text-slate-600">
                                        Change raw per-stem values here. These controls are written into the next container render for this stem. They are not live EQ.
                                    </p>
                                </div>

                                <div class="flex flex-wrap gap-2">
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-full !px-4 !py-2.5"
                                        @click="focusLayer(editorLayerCard.key)"
                                    >
                                        Use as main
                                    </button>
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-full !px-4 !py-2.5"
                                        @click="soloLayer(editorLayerCard.key)"
                                    >
                                        Solo
                                    </button>
                                    <button
                                        type="button"
                                        class="action-secondary !rounded-full !px-4 !py-2.5"
                                        @click="resetLayerControls(editorLayerCard.key)"
                                    >
                                        Reset stem
                                    </button>
                                    <button
                                        type="button"
                                        class="!rounded-full !px-4 !py-2.5 text-sm font-semibold transition"
                                        :class="editorLayerCard.enabled
                                            ? 'bg-slate-950 text-white hover:bg-slate-800'
                                            : 'border border-slate-200 bg-white text-slate-700 hover:border-slate-300'"
                                        @click="toggleLayer(editorLayerCard.key)"
                                    >
                                        {{ editorLayerCard.enabled ? 'Disable stem' : 'Enable stem' }}
                                    </button>
                                </div>
                            </div>

                            <div class="mt-5 grid gap-3 md:grid-cols-3">
                                <div class="rounded-[1.05rem] border border-slate-200 bg-white px-4 py-3">
                                    <div class="field-label">Source file</div>
                                    <div class="mt-2 text-sm font-semibold text-slate-950">{{ editorLayerCard.file_name }}</div>
                                </div>
                                <div class="rounded-[1.05rem] border border-slate-200 bg-white px-4 py-3">
                                    <div class="field-label">Size</div>
                                    <div class="mt-2 text-sm font-semibold text-slate-950">{{ formatFileSize(editorLayerCard.file_size_bytes) }}</div>
                                </div>
                                <div class="rounded-[1.05rem] border border-slate-200 bg-white px-4 py-3">
                                    <div class="field-label">Changed params</div>
                                    <div class="mt-2 text-sm font-semibold text-slate-950">
                                        {{ editorChangedParamCount ? editorChangedParamCount : '0' }}
                                    </div>
                                </div>
                            </div>

                            <div class="mt-6 space-y-4">
                                <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                                    <div class="mb-4">
                                        <div class="field-label">Render pipeline</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            Pick which real container pipeline should run next. This is the honest switch between raw hybrid separation, stem rewrite, and post-DSP over existing stems.
                                        </p>
                                    </div>

                                    <div
                                        v-if="stemRewriteEditorNoOp"
                                        class="mb-4 rounded-[1rem] border border-amber-200 bg-amber-50 px-4 py-4 text-sm leading-6 text-amber-950"
                                    >
                                        You are editing {{ editorLayerCard.label }} while <span class="font-semibold">Stem Rewrite</span> is selected.
                                        In the current worker pipeline, rewrite knobs do not rewrite vocals or bass directly.
                                        Use <span class="font-semibold">Post-DSP</span> or <span class="font-semibold">Hybrid Build</span>
                                        if you want this stem itself to change, or preview a rendered melody/other/drums/guitar/piano source to hear the rewrite branch.
                                    </div>

                                    <div
                                        v-if="vocalsRewriteEditorNoOp"
                                        class="mb-4 rounded-[1rem] border border-amber-200 bg-amber-50 px-4 py-4 text-sm leading-6 text-amber-950"
                                    >
                                        You are editing {{ editorLayerCard.label }} while <span class="font-semibold">Vocals Rewrite</span> is selected.
                                        In the current worker pipeline, this mode rebuilds only the <span class="font-semibold">vocals</span> stem from source audio and preserves the rest.
                                        Use <span class="font-semibold">Post-DSP</span>, <span class="font-semibold">Hybrid Build</span>, or switch the preview/editor to vocals if you want to hear this mode directly.
                                    </div>

                                    <div class="grid gap-3 xl:grid-cols-4">
                                        <label class="rounded-[1rem] border border-slate-200 bg-white px-4 py-4">
                                            <div class="flex items-start gap-3">
                                                <input
                                                    :checked="currentRenderMode === 'hybrid_build'"
                                                    type="radio"
                                                    name="stem-tuning-render-mode"
                                                    class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200"
                                                    @change="setRenderMode('hybrid_build')"
                                                >
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Hybrid Build</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">Raw audio -> full hybrid separation</div>
                                                    <div class="mt-2 text-sm leading-6 text-slate-600">
                                                        Re-run the strong hybrid separator family from the original track and produce a brand new stem set.
                                                    </div>
                                                </div>
                                            </div>
                                        </label>

                                        <label class="rounded-[1rem] border border-slate-200 bg-white px-4 py-4">
                                            <div class="flex items-start gap-3">
                                                <input
                                                    :checked="currentRenderMode === 'stem_rewrite'"
                                                    type="radio"
                                                    name="stem-tuning-render-mode"
                                                    class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200"
                                                    @change="setRenderMode('stem_rewrite')"
                                                >
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Stem Rewrite</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">Existing stems -> melody/bleed rewrite</div>
                                                    <div class="mt-2 text-sm leading-6 text-slate-600">
                                                        Keep the current base stem set and run the stronger melody-first rewrite/repair pipeline over it.
                                                    </div>
                                                </div>
                                            </div>
                                        </label>

                                        <label class="rounded-[1rem] border border-slate-200 bg-white px-4 py-4">
                                            <div class="flex items-start gap-3">
                                                <input
                                                    :checked="currentRenderMode === 'vocals_rewrite'"
                                                    type="radio"
                                                    name="stem-tuning-render-mode"
                                                    class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200"
                                                    @change="setRenderMode('vocals_rewrite')"
                                                >
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Vocals Rewrite</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">Raw audio -> vocals only rebuild</div>
                                                    <div class="mt-2 text-sm leading-6 text-slate-600">
                                                        Rebuild only the vocals stem from the original track with the stronger hybrid vocal pipeline, then keep the other current stems as they are.
                                                    </div>
                                                </div>
                                            </div>
                                        </label>

                                        <label class="rounded-[1rem] border border-slate-200 bg-white px-4 py-4">
                                            <div class="flex items-start gap-3">
                                                <input
                                                    :checked="currentRenderMode === 'post_dsp'"
                                                    type="radio"
                                                    name="stem-tuning-render-mode"
                                                    class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200"
                                                    @change="setRenderMode('post_dsp')"
                                                >
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Post-DSP</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">Existing stems -> per-stem DSP render</div>
                                                    <div class="mt-2 text-sm leading-6 text-slate-600">
                                                        Use the current stem set and apply the fine-grained per-stem gain, HPSS, transient, body, and bleed controls.
                                                    </div>
                                                </div>
                                            </div>
                                        </label>
                                    </div>

                                    <div class="mt-5 rounded-[1rem] border border-sky-200 bg-sky-50 px-4 py-4">
                                        <div class="field-label !text-sky-700">Strong audible vocal presets</div>
                                        <p class="mt-2 text-sm leading-6 text-slate-700">
                                            This is the block you were asking for: one-click presets with deliberately strong values so the next render changes audibly, not by tiny hidden micro-steps.
                                            Clicking any preset below automatically switches the next render to <span class="font-semibold">Vocals Rewrite</span>.
                                        </p>

                                        <div class="mt-4 grid gap-3 xl:grid-cols-4">
                                            <button
                                                type="button"
                                                class="rounded-[1rem] border border-sky-200 bg-white px-4 py-4 text-left transition hover:border-sky-300 hover:bg-sky-50/40"
                                                @click="applyVocalsRewriteAudiblePreset('tight_clean')"
                                            >
                                                <div class="text-sm font-semibold text-slate-950">Tight Clean</div>
                                                <div class="mt-1 text-xs uppercase tracking-[0.14em] text-sky-700">audible cleanup</div>
                                                <p class="mt-3 text-sm leading-6 text-slate-600">
                                                    Stronger cleanup and bleed removal, but still aimed at sounding like a usable vocal instead of a shredded stem.
                                                </p>
                                            </button>

                                            <button
                                                type="button"
                                                class="rounded-[1rem] border border-sky-200 bg-white px-4 py-4 text-left transition hover:border-sky-300 hover:bg-sky-50/40"
                                                @click="applyVocalsRewriteAudiblePreset('hard_isolate')"
                                            >
                                                <div class="text-sm font-semibold text-slate-950">Hard Isolate</div>
                                                <div class="mt-1 text-xs uppercase tracking-[0.14em] text-sky-700">maximum separation</div>
                                                <p class="mt-3 text-sm leading-6 text-slate-600">
                                                    Deliberately aggressive isolation for “make it obviously different now” testing, even if the texture gets rougher.
                                                </p>
                                            </button>

                                            <button
                                                type="button"
                                                class="rounded-[1rem] border border-sky-200 bg-white px-4 py-4 text-left transition hover:border-sky-300 hover:bg-sky-50/40"
                                                @click="applyVocalsRewriteAudiblePreset('artifact_killer')"
                                            >
                                                <div class="text-sm font-semibold text-slate-950">Artifact Killer</div>
                                                <div class="mt-1 text-xs uppercase tracking-[0.14em] text-sky-700">repair-first</div>
                                                <p class="mt-3 text-sm leading-6 text-slate-600">
                                                    Pushes the worker hard toward cleanup and stability when the current vocal already sounds torn or noisy.
                                                </p>
                                            </button>

                                            <button
                                                type="button"
                                                class="rounded-[1rem] border border-sky-200 bg-white px-4 py-4 text-left transition hover:border-sky-300 hover:bg-sky-50/40"
                                                @click="applyVocalsRewriteAudiblePreset('airy_preserve')"
                                            >
                                                <div class="text-sm font-semibold text-slate-950">Airy Preserve</div>
                                                <div class="mt-1 text-xs uppercase tracking-[0.14em] text-sky-700">detail-first</div>
                                                <p class="mt-3 text-sm leading-6 text-slate-600">
                                                    Favors brightness and body retention so you can hear an obvious contrast against the harsher isolation presets.
                                                </p>
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                <div
                                    v-if="isHybridBuildMode"
                                    class="rounded-[1.2rem] border border-slate-200 bg-white px-4 py-4"
                                >
                                    <div class="mb-4">
                                        <div class="field-label">Hybrid build controls</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            These settings belong to the raw hybrid separator family itself. They affect the next full stem build from the original track audio.
                                        </p>
                                    </div>

                                    <div class="grid gap-5 xl:grid-cols-2">
                                        <label class="block">
                                            <div class="text-sm font-semibold text-slate-900">Hybrid model family</div>
                                            <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: hybridBuildConfig.modelName</div>
                                            <select
                                                :value="currentHybridBuildConfig.modelName"
                                                class="mt-3 w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                @change="setHybridBuildConfigValue('modelName', ($event.target as HTMLSelectElement).value)"
                                            >
                                                <option value="hybrid_htdemucs_ft_6s">hybrid_htdemucs_ft_6s</option>
                                                <option value="hybrid_htdemucs_ft_bs_roformer_6s">hybrid_htdemucs_ft_bs_roformer_6s</option>
                                            </select>
                                        </label>

                                        <div class="grid gap-3">
                                            <label class="flex items-start justify-between gap-4 rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Sharpen bleed</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: hybridBuildConfig.sharpenBleed</div>
                                                </div>
                                                <input
                                                    :checked="currentHybridBuildConfig.sharpenBleed"
                                                    type="checkbox"
                                                    class="mt-1 h-4 w-4 rounded border-slate-300 text-sky-500 focus:ring-sky-200"
                                                    @change="setHybridBuildConfigValue('sharpenBleed', ($event.target as HTMLInputElement).checked)"
                                                >
                                            </label>

                                            <label class="flex items-start justify-between gap-4 rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Include melody stem</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: hybridBuildConfig.includeMelodyStem</div>
                                                </div>
                                                <input
                                                    :checked="currentHybridBuildConfig.includeMelodyStem"
                                                    type="checkbox"
                                                    class="mt-1 h-4 w-4 rounded border-slate-300 text-sky-500 focus:ring-sky-200"
                                                    @change="setHybridBuildConfigValue('includeMelodyStem', ($event.target as HTMLInputElement).checked)"
                                                >
                                            </label>

                                            <label class="flex items-start justify-between gap-4 rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Aggressive melody-first mode</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: hybridBuildConfig.aggressiveMelodyFirst</div>
                                                </div>
                                                <input
                                                    :checked="currentHybridBuildConfig.aggressiveMelodyFirst"
                                                    type="checkbox"
                                                    class="mt-1 h-4 w-4 rounded border-slate-300 text-sky-500 focus:ring-sky-200"
                                                    @change="setHybridBuildConfigValue('aggressiveMelodyFirst', ($event.target as HTMLInputElement).checked)"
                                                >
                                            </label>
                                        </div>
                                    </div>
                                </div>

                                <div
                                    v-if="isVocalsRewriteMode"
                                    class="rounded-[1.2rem] border border-slate-200 bg-white px-4 py-4"
                                >
                                    <div class="mb-4">
                                        <div class="field-label">Vocals rewrite controls</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            These are the real hybrid vocal fusion and cleanup knobs used to rebuild only the vocals stem from the raw track audio.
                                        </p>
                                    </div>

                                    <div class="mb-5 rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-3 text-sm leading-6 text-slate-600">
                                        The worker has two mutually exclusive cleanup families here:
                                        <span class="font-semibold text-slate-900">subtractive cleanup</span> and
                                        <span class="font-semibold text-slate-900">hard mask cleanup</span>.
                                        Keep either <span class="font-semibold">Subtractive strength</span> or
                                        <span class="font-semibold">Hard mask threshold</span> at zero.
                                    </div>

                                    <div class="mb-5 rounded-[1rem] border border-sky-200 bg-sky-50 px-4 py-4">
                                        <div class="field-label">Task mode</div>
                                        <p class="mt-2 text-sm leading-6 text-slate-700">
                                            These are strong macro presets, not tiny knob bundles. Picking a task mode resets the vocal rewrite profile to a deliberate starting point so the next render changes audibly.
                                        </p>

                                        <div class="mt-4 grid gap-4 xl:grid-cols-3">
                                            <label :class="currentVocalsRewriteConfig.taskMode === 'clean_vocals' ? 'border-sky-300 bg-white ring-2 ring-sky-200' : 'border-slate-200 bg-white'" class="block cursor-pointer rounded-[1rem] border px-4 py-4 transition">
                                                <div class="flex items-start justify-between gap-4">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Clean vocals</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">balanced cleanup</div>
                                                        <p class="mt-3 text-sm leading-6 text-slate-600">General vocal cleanup with noticeable bleed reduction but without immediately collapsing detail.</p>
                                                    </div>
                                                    <input :checked="currentVocalsRewriteConfig.taskMode === 'clean_vocals'" type="radio" name="vocals-rewrite-task-mode" class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200" @change="applyVocalsRewriteTaskMode('clean_vocals')">
                                                </div>
                                            </label>

                                            <label :class="currentVocalsRewriteConfig.taskMode === 'minus_from_vocals' ? 'border-sky-300 bg-white ring-2 ring-sky-200' : 'border-slate-200 bg-white'" class="block cursor-pointer rounded-[1rem] border px-4 py-4 transition">
                                                <div class="flex items-start justify-between gap-4">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Minus from vocals</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">hard isolation</div>
                                                        <p class="mt-3 text-sm leading-6 text-slate-600">Aggressive vocal-only extraction for subtraction workflows where stronger separation matters more than softness.</p>
                                                    </div>
                                                    <input :checked="currentVocalsRewriteConfig.taskMode === 'minus_from_vocals'" type="radio" name="vocals-rewrite-task-mode" class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200" @change="applyVocalsRewriteTaskMode('minus_from_vocals')">
                                                </div>
                                            </label>

                                            <label :class="currentVocalsRewriteConfig.taskMode === 'artifact_repair' ? 'border-sky-300 bg-white ring-2 ring-sky-200' : 'border-slate-200 bg-white'" class="block cursor-pointer rounded-[1rem] border px-4 py-4 transition">
                                                <div class="flex items-start justify-between gap-4">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Artifact repair</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">stability first</div>
                                                        <p class="mt-3 text-sm leading-6 text-slate-600">Pulls the rewrite back toward safer, fuller vocals and spends more budget on suppressing grit and unstable edges.</p>
                                                    </div>
                                                    <input :checked="currentVocalsRewriteConfig.taskMode === 'artifact_repair'" type="radio" name="vocals-rewrite-task-mode" class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200" @change="applyVocalsRewriteTaskMode('artifact_repair')">
                                                </div>
                                            </label>

                                            <label :class="currentVocalsRewriteConfig.taskMode === 'aggressive_vocal_suppression' ? 'border-sky-300 bg-white ring-2 ring-sky-200' : 'border-slate-200 bg-white'" class="block cursor-pointer rounded-[1rem] border px-4 py-4 transition">
                                                <div class="flex items-start justify-between gap-4">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Aggressive vocal suppression</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">maximum separation</div>
                                                        <p class="mt-3 text-sm leading-6 text-slate-600">This is the blunt instrument. It pushes isolation and center-focused cleanup hard and accepts that texture can get rough.</p>
                                                    </div>
                                                    <input :checked="currentVocalsRewriteConfig.taskMode === 'aggressive_vocal_suppression'" type="radio" name="vocals-rewrite-task-mode" class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200" @change="applyVocalsRewriteTaskMode('aggressive_vocal_suppression')">
                                                </div>
                                            </label>

                                            <label :class="currentVocalsRewriteConfig.taskMode === 'detail_preserve' ? 'border-sky-300 bg-white ring-2 ring-sky-200' : 'border-slate-200 bg-white'" class="block cursor-pointer rounded-[1rem] border px-4 py-4 transition">
                                                <div class="flex items-start justify-between gap-4">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Detail preserve</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">open and natural</div>
                                                        <p class="mt-3 text-sm leading-6 text-slate-600">Biases the rewrite toward openness, body and air so the vocal keeps life even when cleanup stays active.</p>
                                                    </div>
                                                    <input :checked="currentVocalsRewriteConfig.taskMode === 'detail_preserve'" type="radio" name="vocals-rewrite-task-mode" class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200" @change="applyVocalsRewriteTaskMode('detail_preserve')">
                                                </div>
                                            </label>

                                            <label :class="currentVocalsRewriteConfig.taskMode === 'custom' ? 'border-sky-300 bg-white ring-2 ring-sky-200' : 'border-slate-200 bg-white'" class="block cursor-pointer rounded-[1rem] border px-4 py-4 transition">
                                                <div class="flex items-start justify-between gap-4">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Custom / expert</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">manual profile</div>
                                                        <p class="mt-3 text-sm leading-6 text-slate-600">Resets back to neutral defaults so you can build your own profile from the macro sliders and expert micro controls below.</p>
                                                    </div>
                                                    <input :checked="currentVocalsRewriteConfig.taskMode === 'custom'" type="radio" name="vocals-rewrite-task-mode" class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200" @change="applyVocalsRewriteTaskMode('custom')">
                                                </div>
                                            </label>
                                        </div>
                                    </div>

                                    <div class="mb-5 rounded-[1rem] border border-emerald-200 bg-emerald-50 px-4 py-4">
                                        <div class="field-label">Macro shaping</div>
                                        <p class="mt-2 text-sm leading-6 text-slate-700">
                                            These are the strong, audible task knobs. They drive the worker aggressively and also trigger extra vocal postprocess in the container, so this is where you should start before touching the expert micro layer.
                                        </p>

                                        <div class="mt-4 grid gap-5 xl:grid-cols-2">
                                            <label class="block">
                                                <div class="flex items-center justify-between gap-3">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Vocal isolation intensity</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.vocalIsolationIntensity</div>
                                                    </div>
                                                    <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.vocalIsolationIntensity.toFixed(2) }}</div>
                                                </div>
                                                <div class="mt-2 text-sm leading-6 text-slate-600">Pushes the rewrite toward a more isolated vocal candidate instead of a polite, barely-different cleanup.</div>
                                                <div class="mt-3 flex items-center gap-3">
                                                    <input :value="currentVocalsRewriteConfig.vocalIsolationIntensity" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('vocalIsolationIntensity', $event)">
                                                    <input :value="currentVocalsRewriteConfig.vocalIsolationIntensity" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('vocalIsolationIntensity', $event)">
                                                </div>
                                            </label>

                                            <label class="block">
                                                <div class="flex items-center justify-between gap-3">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Bleed removal intensity</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.bleedRemovalIntensity</div>
                                                    </div>
                                                    <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.bleedRemovalIntensity.toFixed(2) }}</div>
                                                </div>
                                                <div class="mt-2 text-sm leading-6 text-slate-600">Turns the cleanup family into a real separation lever instead of tiny subtractive micro-steps.</div>
                                                <div class="mt-3 flex items-center gap-3">
                                                    <input :value="currentVocalsRewriteConfig.bleedRemovalIntensity" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('bleedRemovalIntensity', $event)">
                                                    <input :value="currentVocalsRewriteConfig.bleedRemovalIntensity" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('bleedRemovalIntensity', $event)">
                                                </div>
                                            </label>

                                            <label class="block">
                                                <div class="flex items-center justify-between gap-3">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Air restore</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.airRestore</div>
                                                    </div>
                                                    <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.airRestore.toFixed(2) }}</div>
                                                </div>
                                                <div class="mt-2 text-sm leading-6 text-slate-600">Restores top-end openness after the cleanup pass so the vocal does not come back dull and shut-in.</div>
                                                <div class="mt-3 flex items-center gap-3">
                                                    <input :value="currentVocalsRewriteConfig.airRestore" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('airRestore', $event)">
                                                    <input :value="currentVocalsRewriteConfig.airRestore" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('airRestore', $event)">
                                                </div>
                                            </label>

                                            <label class="block">
                                                <div class="flex items-center justify-between gap-3">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Body preserve</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.bodyPreserve</div>
                                                    </div>
                                                    <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.bodyPreserve.toFixed(2) }}</div>
                                                </div>
                                                <div class="mt-2 text-sm leading-6 text-slate-600">Keeps low-mid weight and fullness instead of letting the cleanup leave only a thin bright shell of the vocal.</div>
                                                <div class="mt-3 flex items-center gap-3">
                                                    <input :value="currentVocalsRewriteConfig.bodyPreserve" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('bodyPreserve', $event)">
                                                    <input :value="currentVocalsRewriteConfig.bodyPreserve" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('bodyPreserve', $event)">
                                                </div>
                                            </label>

                                            <label class="block">
                                                <div class="flex items-center justify-between gap-3">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Artifact suppression</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.artifactSuppression</div>
                                                    </div>
                                                    <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.artifactSuppression.toFixed(2) }}</div>
                                                </div>
                                                <div class="mt-2 text-sm leading-6 text-slate-600">Spends more of the worker budget on controlling grit, harsh edges and unstable residue instead of chasing maximum isolation.</div>
                                                <div class="mt-3 flex items-center gap-3">
                                                    <input :value="currentVocalsRewriteConfig.artifactSuppression" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('artifactSuppression', $event)">
                                                    <input :value="currentVocalsRewriteConfig.artifactSuppression" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('artifactSuppression', $event)">
                                                </div>
                                            </label>

                                            <label class="block">
                                                <div class="flex items-center justify-between gap-3">
                                                    <div>
                                                        <div class="text-sm font-semibold text-slate-900">Center vocal suppression</div>
                                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.centerVocalSuppression</div>
                                                    </div>
                                                    <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.centerVocalSuppression.toFixed(2) }}</div>
                                                </div>
                                                <div class="mt-2 text-sm leading-6 text-slate-600">Applies a center-biased isolation proxy and side attenuation so off-center spill is pushed down harder inside the extracted vocal stem.</div>
                                                <div class="mt-3 flex items-center gap-3">
                                                    <input :value="currentVocalsRewriteConfig.centerVocalSuppression" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('centerVocalSuppression', $event)">
                                                    <input :value="currentVocalsRewriteConfig.centerVocalSuppression" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('centerVocalSuppression', $event)">
                                                </div>
                                            </label>
                                        </div>
                                    </div>

                                    <div class="mb-4">
                                        <div class="field-label">Expert micro overrides</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            These are the raw internal knobs. Use them only after the macro task mode and shaping layer already got you close, otherwise you are back to blind micro-patching.
                                        </p>
                                    </div>

                                    <div class="grid gap-5 xl:grid-cols-2">
                                        <label class="block">
                                            <div class="text-sm font-semibold text-slate-900">Vocal candidate family</div>
                                            <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.candidateMode</div>
                                            <select
                                                :value="currentVocalsRewriteConfig.candidateMode"
                                                class="mt-3 w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                @change="setVocalsRewriteConfigValue('candidateMode', ($event.target as HTMLSelectElement).value)"
                                            >
                                                <option value="hybrid_vocals">hybrid_vocals</option>
                                                <option value="ft_vocals">ft_vocals</option>
                                                <option value="bs_vocals">bs_vocals</option>
                                            </select>
                                        </label>

                                        <div class="rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-4 text-sm leading-6 text-slate-600">
                                            <div class="font-semibold text-slate-900">Mode logic</div>
                                            <p class="mt-2">
                                                <span class="font-semibold">hybrid_vocals</span> blends FT vocals with BS-RoFormer vocals.
                                                <span class="font-semibold">ft_vocals</span> keeps only the FT candidate.
                                                <span class="font-semibold">bs_vocals</span> keeps only the BS candidate.
                                            </p>
                                        </div>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Low BS weight</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.lowBsWeight</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.lowBsWeight.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.lowBsWeight" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('lowBsWeight', $event)">
                                                <input :value="currentVocalsRewriteConfig.lowBsWeight" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('lowBsWeight', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">High BS weight</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.highBsWeight</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.highBsWeight.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.highBsWeight" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('highBsWeight', $event)">
                                                <input :value="currentVocalsRewriteConfig.highBsWeight" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('highBsWeight', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Inactive low BS weight</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.inactiveLowBsWeight</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.inactiveLowBsWeight.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.inactiveLowBsWeight" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('inactiveLowBsWeight', $event)">
                                                <input :value="currentVocalsRewriteConfig.inactiveLowBsWeight" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('inactiveLowBsWeight', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Inactive high BS weight</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.inactiveHighBsWeight</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.inactiveHighBsWeight.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.inactiveHighBsWeight" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('inactiveHighBsWeight', $event)">
                                                <input :value="currentVocalsRewriteConfig.inactiveHighBsWeight" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('inactiveHighBsWeight', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Crossover (Hz)</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.crossoverHz</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ Math.round(currentVocalsRewriteConfig.crossoverHz) }} Hz</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.crossoverHz" type="range" min="200" max="12000" step="10" class="slider-track w-full" @input="onVocalsRewriteConfigInput('crossoverHz', $event)">
                                                <input :value="currentVocalsRewriteConfig.crossoverHz" type="number" min="200" max="12000" step="10" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('crossoverHz', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Activity absolute floor ratio</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.activityAbsFloorRatio</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.activityAbsFloorRatio.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.activityAbsFloorRatio" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('activityAbsFloorRatio', $event)">
                                                <input :value="currentVocalsRewriteConfig.activityAbsFloorRatio" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('activityAbsFloorRatio', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Activity relative floor</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.activityRelFloor</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.activityRelFloor.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.activityRelFloor" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('activityRelFloor', $event)">
                                                <input :value="currentVocalsRewriteConfig.activityRelFloor" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('activityRelFloor', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Activity smoothing frames</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.activitySmoothingFrames</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.activitySmoothingFrames }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.activitySmoothingFrames" type="range" min="1" max="101" step="1" class="slider-track w-full" @input="onVocalsRewriteConfigInput('activitySmoothingFrames', $event)">
                                                <input :value="currentVocalsRewriteConfig.activitySmoothingFrames" type="number" min="1" max="101" step="1" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('activitySmoothingFrames', $event)">
                                            </div>
                                        </label>

                                        <div class="rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-4">
                                            <div class="text-sm font-semibold text-slate-900">Cleanup family</div>
                                            <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">Explicit worker branch inside vocalsRewriteConfig</div>
                                            <div class="mt-3 grid gap-3 sm:grid-cols-3">
                                                <label class="flex items-start gap-3 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700">
                                                    <input
                                                        :checked="currentVocalsRewriteCleanupMode === 'none'"
                                                        type="radio"
                                                        name="vocals-rewrite-cleanup-mode"
                                                        class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200"
                                                        @change="setVocalsRewriteCleanupMode('none')"
                                                    >
                                                    <span>
                                                        <span class="block font-semibold text-slate-900">None</span>
                                                        <span class="mt-1 block text-xs leading-5 text-slate-500">Blend only, with optional sharpen/tail gate.</span>
                                                    </span>
                                                </label>
                                                <label class="flex items-start gap-3 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700">
                                                    <input
                                                        :checked="currentVocalsRewriteCleanupMode === 'subtractive'"
                                                        type="radio"
                                                        name="vocals-rewrite-cleanup-mode"
                                                        class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200"
                                                        @change="setVocalsRewriteCleanupMode('subtractive')"
                                                    >
                                                    <span>
                                                        <span class="block font-semibold text-slate-900">Subtractive</span>
                                                        <span class="mt-1 block text-xs leading-5 text-slate-500">Subtract accompaniment energy from the vocal candidate.</span>
                                                    </span>
                                                </label>
                                                <label class="flex items-start gap-3 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-3 text-sm text-slate-700">
                                                    <input
                                                        :checked="currentVocalsRewriteCleanupMode === 'hard_mask'"
                                                        type="radio"
                                                        name="vocals-rewrite-cleanup-mode"
                                                        class="mt-1 h-4 w-4 border-slate-300 text-sky-500 focus:ring-sky-200"
                                                        @change="setVocalsRewriteCleanupMode('hard_mask')"
                                                    >
                                                    <span>
                                                        <span class="block font-semibold text-slate-900">Hard mask</span>
                                                        <span class="mt-1 block text-xs leading-5 text-slate-500">Use dominance masking instead of subtractive cleanup.</span>
                                                    </span>
                                                </label>
                                            </div>
                                        </div>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Vocal sharpen alpha</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.vocalSharpenAlpha</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.vocalSharpenAlpha.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.vocalSharpenAlpha" type="range" min="0" max="4" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('vocalSharpenAlpha', $event)">
                                                <input :value="currentVocalsRewriteConfig.vocalSharpenAlpha" type="number" min="0" max="4" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('vocalSharpenAlpha', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Tail gate strength</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.tailGateStrength</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.tailGateStrength.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.tailGateStrength" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('tailGateStrength', $event)">
                                                <input :value="currentVocalsRewriteConfig.tailGateStrength" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('tailGateStrength', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Tail gate floor ratio</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.tailGateFloorRatio</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.tailGateFloorRatio.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.tailGateFloorRatio" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('tailGateFloorRatio', $event)">
                                                <input :value="currentVocalsRewriteConfig.tailGateFloorRatio" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('tailGateFloorRatio', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Tail gate smoothing frames</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.tailGateSmoothingFrames</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.tailGateSmoothingFrames }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.tailGateSmoothingFrames" type="range" min="1" max="101" step="1" class="slider-track w-full" @input="onVocalsRewriteConfigInput('tailGateSmoothingFrames', $event)">
                                                <input :value="currentVocalsRewriteConfig.tailGateSmoothingFrames" type="number" min="1" max="101" step="1" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('tailGateSmoothingFrames', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'subtractive'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Subtractive strength</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.subtractiveStrength</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.subtractiveStrength.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.subtractiveStrength" type="range" min="0" max="4" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('subtractiveStrength', $event)">
                                                <input :value="currentVocalsRewriteConfig.subtractiveStrength" type="number" min="0" max="4" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('subtractiveStrength', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'subtractive'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Subtractive power</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.subtractivePower</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.subtractivePower.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.subtractivePower" type="range" min="0.1" max="8" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('subtractivePower', $event)">
                                                <input :value="currentVocalsRewriteConfig.subtractivePower" type="number" min="0.1" max="8" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('subtractivePower', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'subtractive'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Subtractive residual floor</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.subtractiveResidualFloor</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.subtractiveResidualFloor.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.subtractiveResidualFloor" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('subtractiveResidualFloor', $event)">
                                                <input :value="currentVocalsRewriteConfig.subtractiveResidualFloor" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('subtractiveResidualFloor', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'subtractive'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Subtractive active restore</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.subtractiveActiveRestore</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.subtractiveActiveRestore.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.subtractiveActiveRestore" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('subtractiveActiveRestore', $event)">
                                                <input :value="currentVocalsRewriteConfig.subtractiveActiveRestore" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('subtractiveActiveRestore', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'subtractive'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Subtractive target RMS ratio</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.subtractiveTargetRmsRatio</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.subtractiveTargetRmsRatio.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.subtractiveTargetRmsRatio" type="range" min="0" max="4" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('subtractiveTargetRmsRatio', $event)">
                                                <input :value="currentVocalsRewriteConfig.subtractiveTargetRmsRatio" type="number" min="0" max="4" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('subtractiveTargetRmsRatio', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'subtractive'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Subtractive max gain</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.subtractiveMaxGain</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.subtractiveMaxGain.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.subtractiveMaxGain" type="range" min="1" max="4" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('subtractiveMaxGain', $event)">
                                                <input :value="currentVocalsRewriteConfig.subtractiveMaxGain" type="number" min="1" max="4" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('subtractiveMaxGain', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'hard_mask'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Hard mask threshold</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.hardMaskThreshold</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.hardMaskThreshold.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.hardMaskThreshold" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('hardMaskThreshold', $event)">
                                                <input :value="currentVocalsRewriteConfig.hardMaskThreshold" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('hardMaskThreshold', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'hard_mask'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Hard mask power</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.hardMaskPower</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.hardMaskPower.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.hardMaskPower" type="range" min="1" max="12" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('hardMaskPower', $event)">
                                                <input :value="currentVocalsRewriteConfig.hardMaskPower" type="number" min="1" max="12" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('hardMaskPower', $event)">
                                            </div>
                                        </label>

                                        <label v-if="currentVocalsRewriteCleanupMode === 'hard_mask'" class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Hard mask residual floor</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: vocalsRewriteConfig.hardMaskResidualFloor</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentVocalsRewriteConfig.hardMaskResidualFloor.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentVocalsRewriteConfig.hardMaskResidualFloor" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onVocalsRewriteConfigInput('hardMaskResidualFloor', $event)">
                                                <input :value="currentVocalsRewriteConfig.hardMaskResidualFloor" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onVocalsRewriteConfigInput('hardMaskResidualFloor', $event)">
                                            </div>
                                        </label>
                                    </div>
                                </div>

                                <div
                                    v-if="isStemRewriteMode"
                                    class="rounded-[1.2rem] border border-slate-200 bg-white px-4 py-4"
                                >
                                    <div class="mb-4">
                                        <div class="field-label">Stem rewrite controls</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            These are the real selective melody-first rewrite knobs from the hybrid stem repair path. They affect the next rewrite of the current base stem set.
                                        </p>
                                    </div>

                                    <div class="mb-5 rounded-[1rem] border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900">
                                        In the current worker, stem rewrite preserves <span class="font-semibold">vocals</span> and <span class="font-semibold">bass</span> exactly for full-track renders.
                                        These rewrite knobs currently target
                                        <span class="font-semibold">{{ stemRewriteAffectedSources.map((source) => source.label).join(', ') || 'drums, other, guitar, piano' }}</span>.
                                        <template v-if="stemRewriteSelectedSourcePreserved">
                                            Your current preview source is preserved in this mode, so it can sound unchanged even after a successful render.
                                        </template>
                                        <template v-if="stemRewriteMixWillSoundUnchanged">
                                            Right now only preserved stems are enabled in the mix, so <span class="font-semibold">render_mix</span> can also come back sounding unchanged.
                                        </template>
                                    </div>

                                    <div class="grid gap-5 xl:grid-cols-2">
                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Other move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.otherMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.otherMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.otherMove" type="range" min="0" max="1.5" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('otherMove', $event)">
                                                <input :value="currentStemRewriteConfig.otherMove" type="number" min="0" max="1.5" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('otherMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Drums masked move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.drumsMaskedMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.drumsMaskedMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.drumsMaskedMove" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('drumsMaskedMove', $event)">
                                                <input :value="currentStemRewriteConfig.drumsMaskedMove" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('drumsMaskedMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Target melody RMS</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.targetMelodyRms</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.targetMelodyRms.toFixed(3) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.targetMelodyRms" type="range" min="0.001" max="0.1" step="0.001" class="slider-track w-full" @input="onStemRewriteConfigInput('targetMelodyRms', $event)">
                                                <input :value="currentStemRewriteConfig.targetMelodyRms" type="number" min="0.001" max="0.1" step="0.001" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('targetMelodyRms', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Confidence floor</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.confidenceFloor</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.confidenceFloor.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.confidenceFloor" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('confidenceFloor', $event)">
                                                <input :value="currentStemRewriteConfig.confidenceFloor" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('confidenceFloor', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Guitar move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.guitarMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.guitarMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.guitarMove" type="range" min="0" max="1.5" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('guitarMove', $event)">
                                                <input :value="currentStemRewriteConfig.guitarMove" type="number" min="0" max="1.5" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('guitarMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Piano move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.pianoMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.pianoMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.pianoMove" type="range" min="0" max="1.5" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('pianoMove', $event)">
                                                <input :value="currentStemRewriteConfig.pianoMove" type="number" min="0" max="1.5" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('pianoMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Max melody gain</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.maxMelodyGain</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.maxMelodyGain.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.maxMelodyGain" type="range" min="1" max="3" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('maxMelodyGain', $event)">
                                                <input :value="currentStemRewriteConfig.maxMelodyGain" type="number" min="1" max="3" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('maxMelodyGain', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Stage drum harmonic move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.stageDrumHarmonicMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.stageDrumHarmonicMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.stageDrumHarmonicMove" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('stageDrumHarmonicMove', $event)">
                                                <input :value="currentStemRewriteConfig.stageDrumHarmonicMove" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('stageDrumHarmonicMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Stage harmonic stem percussive move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.stageHarmonicStemPercussiveMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.stageHarmonicStemPercussiveMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.stageHarmonicStemPercussiveMove" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('stageHarmonicStemPercussiveMove', $event)">
                                                <input :value="currentStemRewriteConfig.stageHarmonicStemPercussiveMove" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('stageHarmonicStemPercussiveMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Stage cleanup drum harmonic move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.stageCleanupDrumHarmonicMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.stageCleanupDrumHarmonicMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.stageCleanupDrumHarmonicMove" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('stageCleanupDrumHarmonicMove', $event)">
                                                <input :value="currentStemRewriteConfig.stageCleanupDrumHarmonicMove" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('stageCleanupDrumHarmonicMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Stage cleanup harmonic stem percussive move</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.stageCleanupHarmonicStemPercussiveMove</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.stageCleanupHarmonicStemPercussiveMove.toFixed(2) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.stageCleanupHarmonicStemPercussiveMove" type="range" min="0" max="1" step="0.01" class="slider-track w-full" @input="onStemRewriteConfigInput('stageCleanupHarmonicStemPercussiveMove', $event)">
                                                <input :value="currentStemRewriteConfig.stageCleanupHarmonicStemPercussiveMove" type="number" min="0" max="1" step="0.01" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('stageCleanupHarmonicStemPercussiveMove', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Harmonic count</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: stemRewriteConfig.harmonicCount</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentStemRewriteConfig.harmonicCount }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentStemRewriteConfig.harmonicCount" type="range" min="1" max="8" step="1" class="slider-track w-full" @input="onStemRewriteConfigInput('harmonicCount', $event)">
                                                <input :value="currentStemRewriteConfig.harmonicCount" type="number" min="1" max="8" step="1" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onStemRewriteConfigInput('harmonicCount', $event)">
                                            </div>
                                        </label>

                                        <div class="rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-4 xl:col-span-2">
                                            <div class="text-sm font-semibold text-slate-900">No hidden rewrite knobs in this mode</div>
                                            <div class="mt-2 text-sm leading-6 text-slate-600">
                                                Every preset parameter that the current stem rewrite container mode actually uses is now surfaced here. Nothing important is supposed to live only in JSON anymore.
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div class="rounded-[1.2rem] border border-slate-200 bg-white px-4 py-4 xl:col-span-2">
                                    <div class="mb-4 flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                                        <div>
                                            <div class="field-label">Expert mode config</div>
                                            <p class="mt-1 text-sm leading-6 text-slate-600">
                                                This is the full raw JSON object for the current container mode:
                                                <span class="font-semibold text-slate-900">{{ activeModeConfigField }}</span>.
                                                If a key is part of the current mode contract, you can edit it here directly instead of hunting across sliders.
                                            </p>
                                        </div>
                                        <div class="rounded-[1rem] border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-600">
                                            {{ activeModeConfigLabel }} expert surface. No hidden mode keys inside the current container contract.
                                        </div>
                                    </div>

                                    <div class="grid gap-4 xl:grid-cols-[minmax(0,1fr),minmax(0,1fr)]">
                                        <div class="rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-4">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Current live {{ activeModeConfigLabel }} object</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">Read-only mirror of the active container contract</div>
                                                </div>
                                                <button type="button" class="action-secondary !rounded-[0.95rem] !px-3 !py-2" @click="copyActiveModeConfigJson">
                                                    Copy current
                                                </button>
                                            </div>
                                            <pre class="mt-3 overflow-x-auto rounded-[0.9rem] border border-slate-200 bg-white px-4 py-4 font-mono text-xs leading-6 text-slate-700">{{ activeModeConfigJson }}</pre>
                                        </div>

                                        <div class="rounded-[1rem] border border-slate-200 bg-slate-50/70 px-4 py-4">
                                            <div class="flex items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Edit {{ activeModeConfigLabel }} JSON</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">Auto-applies only to the active mode object, not the whole preset</div>
                                                </div>
                                                <button type="button" class="action-secondary !rounded-[0.95rem] !px-3 !py-2" @click="loadActiveModeConfigDraft">
                                                    Reload current
                                                </button>
                                            </div>
                                            <div class="mt-3 text-xs uppercase tracking-[0.14em] text-slate-500">
                                                Paste valid mode JSON here. It auto-applies after a short pause.
                                            </div>
                                            <textarea
                                                v-model="activeModeConfigDraft"
                                                rows="18"
                                                class="mt-3 min-h-[20rem] w-full rounded-[0.9rem] border border-slate-200 bg-white px-4 py-4 font-mono text-xs leading-6 text-slate-700 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                spellcheck="false"
                                            ></textarea>
                                            <div class="mt-3 text-xs font-semibold uppercase tracking-[0.14em]" :class="jsonDraftStatusClasses(activeModeConfigDraftStatusTone)">
                                                {{ activeModeConfigDraftStatusMessage }}
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <template v-if="isPostDspMode">
                                <div class="rounded-[1.2rem] border border-slate-200 bg-white px-4 py-4">
                                    <div class="mb-4">
                                        <div class="field-label">Layer mix / routing</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            `gainDb` is saved into the next rendered stem. It does not change the current playback until the container finishes a new render.
                                        </p>
                                    </div>

                                    <label class="block">
                                        <div class="flex flex-wrap items-center justify-between gap-3">
                                            <div>
                                                <div class="text-sm font-semibold text-slate-900">Stem level</div>
                                                <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: gainDb</div>
                                            </div>
                                            <div class="text-sm font-semibold text-slate-900">{{ editorLayerCard.gainDb.toFixed(1) }} dB</div>
                                        </div>
                                        <div class="mt-3 flex items-center gap-3">
                                            <input
                                                :value="editorLayerCard.gainDb"
                                                type="range"
                                                min="-48"
                                                max="12"
                                                step="0.1"
                                                class="slider-track w-full"
                                                @input="onLayerGainInput(editorLayerCard.key, $event)"
                                            >
                                            <input
                                                :value="editorLayerCard.gainDb"
                                                type="number"
                                                min="-48"
                                                max="12"
                                                step="0.1"
                                                class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                @input="onLayerGainInput(editorLayerCard.key, $event)"
                                            >
                                        </div>
                                    </label>
                                </div>

                                <div class="rounded-[1.2rem] border border-slate-200 bg-white px-4 py-4">
                                    <div class="mb-4">
                                        <div class="field-label">Raw stem processors</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            These four sliders are only the per-stem layer values. They are not the whole render engine.
                                        </p>
                                    </div>

                                    <div class="space-y-5">
                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Transient / attack</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: transientDb</div>
                                                    <div class="mt-1 text-sm text-slate-600">Pushes snap and attack forward. Positive = harder hits, negative = softer attacks.</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ editorLayerCard.transientDb.toFixed(1) }} dB</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input
                                                    :value="editorLayerCard.transientDb"
                                                    type="range"
                                                    min="-18"
                                                    max="18"
                                                    step="0.1"
                                                    class="slider-track w-full"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'transientDb', $event)"
                                                >
                                                <input
                                                    :value="editorLayerCard.transientDb"
                                                    type="number"
                                                    min="-18"
                                                    max="18"
                                                    step="0.1"
                                                    class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'transientDb', $event)"
                                                >
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Body</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: bodyDb</div>
                                                    <div class="mt-1 text-sm text-slate-600">Controls low-mid weight and thickness. Positive = fuller, negative = leaner.</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ editorLayerCard.bodyDb.toFixed(1) }} dB</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input
                                                    :value="editorLayerCard.bodyDb"
                                                    type="range"
                                                    min="-18"
                                                    max="18"
                                                    step="0.1"
                                                    class="slider-track w-full"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'bodyDb', $event)"
                                                >
                                                <input
                                                    :value="editorLayerCard.bodyDb"
                                                    type="number"
                                                    min="-18"
                                                    max="18"
                                                    step="0.1"
                                                    class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'bodyDb', $event)"
                                                >
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Harmonic duck</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: harmonicDuckDb</div>
                                                    <div class="mt-1 text-sm text-slate-600">Pushes pitched melodic residue down inside this stem. More negative = less melody bleed.</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ editorLayerCard.harmonicDuckDb.toFixed(1) }} dB</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input
                                                    :value="editorLayerCard.harmonicDuckDb"
                                                    type="range"
                                                    min="-18"
                                                    max="0"
                                                    step="0.1"
                                                    class="slider-track w-full"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'harmonicDuckDb', $event)"
                                                >
                                                <input
                                                    :value="editorLayerCard.harmonicDuckDb"
                                                    type="number"
                                                    min="-18"
                                                    max="0"
                                                    step="0.1"
                                                    class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'harmonicDuckDb', $event)"
                                                >
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Bleed duck</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: bleedDuckDb</div>
                                                    <div class="mt-1 text-sm text-slate-600">Pushes upper-mid spill down. More negative = less contamination from neighboring stems.</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ editorLayerCard.bleedDuckDb.toFixed(1) }} dB</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input
                                                    :value="editorLayerCard.bleedDuckDb"
                                                    type="range"
                                                    min="-18"
                                                    max="0"
                                                    step="0.1"
                                                    class="slider-track w-full"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'bleedDuckDb', $event)"
                                                >
                                                <input
                                                    :value="editorLayerCard.bleedDuckDb"
                                                    type="number"
                                                    min="-18"
                                                    max="0"
                                                    step="0.1"
                                                    class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                                                    @input="onLayerProcessorInput(editorLayerCard.key, 'bleedDuckDb', $event)"
                                                >
                                            </div>
                                        </label>
                                    </div>
                                </div>

                                <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-4">
                                    <div class="mb-4">
                                        <div class="field-label">Container DSP config</div>
                                        <p class="mt-1 text-sm leading-6 text-slate-600">
                                            These are the real global DSP constants the worker uses during stem render. They affect the next container render for all stems, not just this one editor card.
                                        </p>
                                        <p class="mt-2 text-sm leading-6 text-slate-600">
                                            <span class="font-semibold text-slate-900">Important:</span> harmonic and percussive margins affect the HPSS split directly.
                                            The band controls below are companion knobs:
                                            <span class="font-semibold">transient cutoff</span> matters only when some stem has <span class="font-semibold">transientDb != 0</span>,
                                            <span class="font-semibold">body low/high</span> matters only when some stem has <span class="font-semibold">bodyDb != 0</span>,
                                            and <span class="font-semibold">bleed band</span> matters only when some stem has <span class="font-semibold">bleedDuckDb != 0</span>.
                                        </p>
                                    </div>

                                    <div
                                        v-if="postDspBypassed"
                                        class="mb-4 rounded-[1rem] border border-amber-200 bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-900"
                                    >
                                        DSP bypass is ON. Post-DSP processor sliders are ignored by the container until bypass is turned OFF.
                                        If you change any DSP slider below, bypass is now turned OFF automatically.
                                    </div>

                                    <label class="flex items-start justify-between gap-4 rounded-[1rem] border border-slate-200 bg-white px-4 py-3">
                                        <div>
                                            <div class="text-sm font-semibold text-slate-900">Bypass DSP processors</div>
                                            <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: processorConfig.bypass</div>
                                            <div class="mt-1 text-sm text-slate-600">Leaves only stem gain/routing and skips HPSS plus transient/body/bleed processing.</div>
                                        </div>
                                        <input
                                            :checked="currentProcessorConfig.bypass"
                                            type="checkbox"
                                            class="mt-1 h-4 w-4 rounded border-slate-300 text-sky-500 focus:ring-sky-200"
                                            @change="setProcessorConfigValue('bypass', ($event.target as HTMLInputElement).checked)"
                                        >
                                    </label>

                                    <div class="mt-5 grid gap-5 xl:grid-cols-2">
                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Harmonic margin</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: processorConfig.harmonicMargin</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentProcessorConfig.harmonicMargin.toFixed(1) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentProcessorConfig.harmonicMargin" type="range" min="0.5" max="4" step="0.1" class="slider-track w-full" @input="onProcessorConfigInput('harmonicMargin', $event)">
                                                <input :value="currentProcessorConfig.harmonicMargin" type="number" min="0.5" max="4" step="0.1" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onProcessorConfigInput('harmonicMargin', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Percussive margin</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: processorConfig.percussiveMargin</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ currentProcessorConfig.percussiveMargin.toFixed(1) }}</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentProcessorConfig.percussiveMargin" type="range" min="0.5" max="6" step="0.1" class="slider-track w-full" @input="onProcessorConfigInput('percussiveMargin', $event)">
                                                <input :value="currentProcessorConfig.percussiveMargin" type="number" min="0.5" max="6" step="0.1" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onProcessorConfigInput('percussiveMargin', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Transient cutoff (Hz)</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: processorConfig.transientCutoffHz</div>
                                                    <div class="mt-1 text-sm text-slate-600">Companion knob for per-layer <span class="font-semibold">transientDb</span>.</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ Math.round(currentProcessorConfig.transientCutoffHz) }} Hz</div>
                                            </div>
                                            <div class="mt-3 flex items-center gap-3">
                                                <input :value="currentProcessorConfig.transientCutoffHz" type="range" min="800" max="8000" step="10" class="slider-track w-full" @input="onProcessorConfigInput('transientCutoffHz', $event)">
                                                <input :value="currentProcessorConfig.transientCutoffHz" type="number" min="800" max="8000" step="10" class="w-24 rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onProcessorConfigInput('transientCutoffHz', $event)">
                                            </div>
                                        </label>

                                        <label class="block">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Body low / high (Hz)</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON keys: processorConfig.bodyLowHz / bodyHighHz</div>
                                                    <div class="mt-1 text-sm text-slate-600">Companion band for per-layer <span class="font-semibold">bodyDb</span>.</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ Math.round(currentProcessorConfig.bodyLowHz) }} - {{ Math.round(currentProcessorConfig.bodyHighHz) }} Hz</div>
                                            </div>
                                            <div class="mt-3 grid gap-3 md:grid-cols-2">
                                                <input :value="currentProcessorConfig.bodyLowHz" type="number" min="40" max="400" step="5" class="w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onProcessorConfigInput('bodyLowHz', $event)">
                                                <input :value="currentProcessorConfig.bodyHighHz" type="number" min="120" max="1200" step="5" class="w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onProcessorConfigInput('bodyHighHz', $event)">
                                            </div>
                                        </label>

                                        <label class="block xl:col-span-2">
                                            <div class="flex flex-wrap items-center justify-between gap-3">
                                                <div>
                                                    <div class="text-sm font-semibold text-slate-900">Bleed band low / high (Hz)</div>
                                                    <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON keys: processorConfig.bleedLowHz / bleedHighHz</div>
                                                    <div class="mt-1 text-sm text-slate-600">Companion band for per-layer <span class="font-semibold">bleedDuckDb</span>.</div>
                                                </div>
                                                <div class="text-sm font-semibold text-slate-900">{{ Math.round(currentProcessorConfig.bleedLowHz) }} - {{ Math.round(currentProcessorConfig.bleedHighHz) }} Hz</div>
                                            </div>
                                            <div class="mt-3 grid gap-3 md:grid-cols-2">
                                                <input :value="currentProcessorConfig.bleedLowHz" type="number" min="120" max="2500" step="10" class="w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onProcessorConfigInput('bleedLowHz', $event)">
                                                <input :value="currentProcessorConfig.bleedHighHz" type="number" min="500" max="9000" step="10" class="w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200" @input="onProcessorConfigInput('bleedHighHz', $event)">
                                            </div>
                                        </label>
                                    </div>
                                </div>
                                </template>
                            </div>
                        </section>
                </div>
            </section>

            <details class="surface-card !p-6">
                <summary class="cursor-pointer list-none">
                    <div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <p class="section-kicker !text-amber-700">Unsafe Internals</p>
                            <h2 class="mt-3 text-2xl font-semibold tracking-tight text-slate-950">Vendor-level model overrides</h2>
                            <p class="mt-2 text-sm leading-6 text-slate-600">
                                This is deliberately separate from the normal tuning UI. It overrides repo paths, checkpoint paths, BS-RoFormer YAML/runtime flags, hybrid stage models, and stage-specific devices inside the container.
                            </p>
                        </div>
                        <div class="rounded-[1.1rem] border border-amber-200 bg-amber-50/80 px-4 py-3 text-sm text-amber-800">
                            Hidden by default on purpose. This is unsafe model internals, not normal lab tuning.
                        </div>
                    </div>
                </summary>

                <div class="mt-5 rounded-[1rem] border border-amber-200 bg-amber-50/60 px-4 py-4 text-sm leading-6 text-amber-900">
                    These keys can break the render if you point them at missing repos, mismatched configs, incompatible checkpoints, invalid CLI args, or a broken YAML override. Use this only when you intentionally want to override vendor/runtime internals.
                </div>

                <div class="mt-5 space-y-6">
                    <section class="rounded-[1.6rem] border border-amber-200 bg-white px-5 py-5">
                        <div class="mb-4">
                            <p class="section-kicker !text-amber-700">Hybrid Runtime</p>
                            <h3 class="mt-2 text-xl font-semibold tracking-tight text-slate-950">Top-level hybrid routing</h3>
                            <p class="mt-2 text-sm leading-6 text-slate-600">
                                These are the first-class unsafe controls for the hybrid container path. They are no longer JSON-only.
                            </p>
                        </div>
                        <div class="grid gap-5 lg:grid-cols-2">
                            <label
                                v-for="field in unsafeHybridRootFields"
                                :key="field.key"
                                class="block rounded-[1.2rem] border border-amber-100 bg-amber-50/30 px-4 py-4"
                            >
                                <div class="flex items-start justify-between gap-4">
                                    <div>
                                        <div class="text-sm font-semibold text-slate-900">{{ field.label }}</div>
                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: unsafeModelInternals.hybrid.{{ field.key }}</div>
                                    </div>
                                </div>
                                <div class="mt-2 text-sm leading-6 text-slate-600">{{ field.description }}</div>
                                <input
                                    :value="unsafeHybridFieldValue(field.key)"
                                    :placeholder="field.placeholder"
                                    type="text"
                                    class="mt-3 w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-200"
                                    @input="onUnsafeHybridInput(field.key, $event, 'text')"
                                >
                            </label>
                        </div>
                    </section>

                    <div class="grid gap-6 xl:grid-cols-2">
                        <section
                            v-for="stageNumber in [1, 2]"
                            :key="stageNumber"
                            class="rounded-[1.6rem] border border-amber-200 bg-white px-5 py-5"
                        >
                            <div class="mb-4">
                                <p class="section-kicker !text-amber-700">Hybrid Stage {{ stageNumber }}</p>
                                <h3 class="mt-2 text-xl font-semibold tracking-tight text-slate-950">Stage {{ stageNumber }} runtime contract</h3>
                                <p class="mt-2 text-sm leading-6 text-slate-600">
                                    This maps to the per-stage Demucs runtime: model choice, requested device, and direct `apply_model(...)` overrides.
                                </p>
                            </div>
                            <div class="grid gap-5">
                                <template v-for="field in unsafeHybridStageFields" :key="`${stageNumber}-${field.key}`">
                                    <label
                                        v-if="field.kind !== 'checkbox'"
                                        class="block rounded-[1.2rem] border border-amber-100 bg-amber-50/30 px-4 py-4"
                                    >
                                        <div class="flex items-start justify-between gap-4">
                                            <div>
                                                <div class="text-sm font-semibold text-slate-900">{{ field.label }}</div>
                                                <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: unsafeModelInternals.hybrid.{{ hybridStageFieldKey(stageNumber, field.key) }}</div>
                                            </div>
                                        </div>
                                        <div class="mt-2 text-sm leading-6 text-slate-600">{{ field.description }}</div>
                                        <input
                                            :value="unsafeHybridStageValue(stageNumber, field.key)"
                                            :placeholder="field.placeholder"
                                            :type="field.kind === 'number' ? 'number' : 'text'"
                                            :min="field.min"
                                            :max="field.max"
                                            :step="field.step"
                                            class="mt-3 w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-200"
                                            @input="onUnsafeHybridStageInput(stageNumber, field.key, $event, field.kind === 'number' ? 'number' : 'text')"
                                        >
                                    </label>

                                    <label
                                        v-else
                                        class="flex items-start justify-between gap-4 rounded-[1.2rem] border border-amber-100 bg-amber-50/30 px-4 py-4"
                                    >
                                        <div>
                                            <div class="text-sm font-semibold text-slate-900">{{ field.label }}</div>
                                            <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: unsafeModelInternals.hybrid.{{ hybridStageFieldKey(stageNumber, field.key) }}</div>
                                            <div class="mt-2 text-sm leading-6 text-slate-600">{{ field.description }}</div>
                                        </div>
                                        <input
                                            :checked="Boolean(unsafeHybridStageValue(stageNumber, field.key))"
                                            type="checkbox"
                                            class="mt-1 h-4 w-4 rounded border-slate-300 text-amber-500 focus:ring-amber-200"
                                            @change="onUnsafeHybridStageInput(stageNumber, field.key, $event, 'checkbox')"
                                        >
                                    </label>
                                </template>
                            </div>
                        </section>
                    </div>

                    <section class="rounded-[1.6rem] border border-amber-200 bg-white px-5 py-5">
                        <div class="mb-4">
                            <p class="section-kicker !text-amber-700">BS-RoFormer Runtime</p>
                            <h3 class="mt-2 text-xl font-semibold tracking-tight text-slate-950">Wrapper and vendor config</h3>
                            <p class="mt-2 text-sm leading-6 text-slate-600">
                                These are the first-class unsafe wrapper controls for the BS-RoFormer vendor path, including device ids and progress-bar control.
                            </p>
                        </div>
                        <div class="grid gap-5 xl:grid-cols-2">
                            <template v-for="field in unsafeBsRoformerFields" :key="field.key">
                                <label
                                    v-if="field.kind !== 'checkbox' && field.kind !== 'textarea'"
                                    class="block rounded-[1.2rem] border border-amber-100 bg-amber-50/30 px-4 py-4"
                                >
                                    <div class="flex items-start justify-between gap-4">
                                        <div>
                                            <div class="text-sm font-semibold text-slate-900">{{ field.label }}</div>
                                            <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: unsafeModelInternals.bsRoformer.{{ field.key }}</div>
                                        </div>
                                    </div>
                                    <div class="mt-2 text-sm leading-6 text-slate-600">{{ field.description }}</div>
                                    <textarea
                                        v-if="field.kind === 'string_array'"
                                        :value="unsafeBsTextFieldValue(field.key)"
                                        :placeholder="field.placeholder"
                                        rows="4"
                                        class="mt-3 w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 font-mono text-sm text-slate-900 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-200"
                                        @input="onUnsafeBsRoformerInput(field.key, $event, field.kind)"
                                    ></textarea>
                                    <input
                                        v-else
                                        :value="unsafeBsTextFieldValue(field.key)"
                                        :placeholder="field.placeholder"
                                        type="text"
                                        class="mt-3 w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-200"
                                        @input="onUnsafeBsRoformerInput(field.key, $event, field.kind)"
                                    >
                                </label>

                                <label
                                    v-else-if="field.kind === 'textarea'"
                                    class="block rounded-[1.2rem] border border-amber-100 bg-amber-50/30 px-4 py-4 xl:col-span-2"
                                >
                                    <div class="flex items-start justify-between gap-4">
                                        <div>
                                            <div class="text-sm font-semibold text-slate-900">{{ field.label }}</div>
                                            <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: unsafeModelInternals.bsRoformer.{{ field.key }}</div>
                                        </div>
                                    </div>
                                    <div class="mt-2 text-sm leading-6 text-slate-600">{{ field.description }}</div>
                                    <textarea
                                        :value="unsafeBsTextFieldValue(field.key)"
                                        :placeholder="field.placeholder"
                                        rows="12"
                                        class="mt-3 w-full rounded-[0.9rem] border border-slate-200 bg-white px-3 py-2 font-mono text-xs leading-6 text-slate-900 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-200"
                                        spellcheck="false"
                                        @input="onUnsafeBsRoformerInput(field.key, $event, field.kind)"
                                    ></textarea>
                                </label>

                                <label
                                    v-else
                                    class="flex items-start justify-between gap-4 rounded-[1.2rem] border border-amber-100 bg-amber-50/30 px-4 py-4"
                                >
                                    <div>
                                        <div class="text-sm font-semibold text-slate-900">{{ field.label }}</div>
                                        <div class="mt-1 text-xs uppercase tracking-[0.14em] text-slate-500">JSON key: unsafeModelInternals.bsRoformer.{{ field.key }}</div>
                                        <div class="mt-2 text-sm leading-6 text-slate-600">{{ field.description }}</div>
                                    </div>
                                    <input
                                        :checked="Boolean(unsafeBsFieldValue(field.key))"
                                        type="checkbox"
                                        class="mt-1 h-4 w-4 rounded border-slate-300 text-amber-500 focus:ring-amber-200"
                                        @change="onUnsafeBsRoformerInput(field.key, $event, field.kind)"
                                    >
                                </label>
                            </template>
                        </div>
                    </section>

                    <div class="grid gap-6 xl:grid-cols-2">
                        <div>
                            <div class="mb-2 flex items-center justify-between">
                                <span class="field-label !text-amber-700">Current unsafe internals</span>
                                <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="copyUnsafeModelInternalsJson">
                                    Copy current
                                </button>
                            </div>
                            <pre class="min-h-[360px] overflow-x-auto rounded-[1.2rem] border border-amber-200 bg-white px-4 py-4 font-mono text-xs leading-6 text-slate-900">{{ unsafeModelInternalsJson }}</pre>
                        </div>

                        <div>
                            <div class="mb-2 flex items-center justify-between">
                                <span class="field-label !text-amber-700">Edit unsafe internals JSON</span>
                                <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="loadUnsafeModelInternalsDraft">
                                    Reload current
                                </button>
                            </div>
                            <div class="mb-2 text-xs uppercase tracking-[0.14em] text-slate-500">
                                Paste valid unsafe JSON here. It auto-applies after a short pause.
                            </div>
                            <textarea
                                v-model="unsafeModelInternalsDraft"
                                rows="18"
                                class="min-h-[360px] w-full rounded-[1.2rem] border border-amber-200 bg-white px-4 py-4 font-mono text-xs leading-6 text-slate-900 outline-none transition focus:border-amber-400 focus:ring-2 focus:ring-amber-200"
                                spellcheck="false"
                            ></textarea>
                            <div class="mt-3 text-xs font-semibold uppercase tracking-[0.14em]" :class="jsonDraftStatusClasses(unsafeModelInternalsDraftStatusTone)">
                                {{ unsafeModelInternalsDraftStatusMessage }}
                            </div>
                        </div>
                    </div>
                </div>
            </details>

            <details class="surface-card !p-6">
                <summary class="cursor-pointer list-none">
                    <div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                        <div>
                            <p class="section-kicker">Advanced JSON</p>
                            <h2 class="mt-3 text-2xl font-semibold tracking-tight text-slate-950">Export, import, and debugging</h2>
                            <p class="mt-2 text-sm leading-6 text-slate-600">
                                This is the raw preset only. Use it when you want to copy a preset, debug exact values, or paste a JSON manually.
                            </p>
                        </div>
                        <div class="rounded-[1.1rem] border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-600">
                            Hidden by default on purpose. The main workflow is the stem editor above.
                        </div>
                    </div>
                </summary>

                <div class="mt-5 flex flex-wrap gap-3">
                    <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="copyPresetJson">
                        Copy current JSON
                    </button>
                    <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="downloadPresetJson">
                        Download JSON
                    </button>
                    <button type="button" class="action-secondary !rounded-[1.1rem] !px-4 !py-3" @click="resetPreset">
                        Reset preset
                    </button>
                </div>

                <div class="mt-5 grid gap-6 xl:grid-cols-2">
                    <div>
                        <div class="mb-2 flex items-center justify-between">
                            <span class="field-label">Current preset output</span>
                        </div>
                        <textarea
                            :value="displayPresetJson"
                            readonly
                            class="min-h-[360px] w-full rounded-[1.2rem] border border-slate-200 bg-slate-950 px-4 py-4 font-mono text-xs leading-6 text-slate-100 outline-none"
                        ></textarea>
                    </div>

                    <div>
                        <div class="mb-2 flex items-center justify-between">
                            <span class="field-label">Editor / import</span>
                            <button type="button" class="text-sm font-semibold text-sky-600 transition hover:text-sky-700" @click="loadCurrentJsonIntoEditor">
                                Load current JSON
                            </button>
                        </div>
                        <div class="mb-2 text-xs uppercase tracking-[0.14em] text-slate-500">
                            Paste any valid preset JSON here. It auto-applies after a short pause.
                        </div>
                        <textarea
                            v-model="importBuffer"
                            class="min-h-[360px] w-full rounded-[1.2rem] border border-slate-200 bg-white px-4 py-4 font-mono text-xs leading-6 text-slate-900 outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-200"
                        ></textarea>
                        <div class="mt-3 text-xs font-semibold uppercase tracking-[0.14em]" :class="jsonDraftStatusClasses(importBufferStatusTone)">
                            {{ importBufferStatusMessage }}
                        </div>
                    </div>
                </div>
            </details>
        </template>
    </div>
</template>

<style scoped>
.slider-track {
    appearance: none;
    height: 10px;
    border-radius: 999px;
    background:
        linear-gradient(90deg, rgba(56, 189, 248, 0.18), rgba(14, 165, 233, 0.52)),
        #e2e8f0;
}

.slider-track::-webkit-slider-thumb {
    appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 999px;
    background: #0f172a;
    border: 2px solid #f8fafc;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.22);
    cursor: pointer;
}

.slider-track::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 999px;
    background: #0f172a;
    border: 2px solid #f8fafc;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.22);
    cursor: pointer;
}
</style>

