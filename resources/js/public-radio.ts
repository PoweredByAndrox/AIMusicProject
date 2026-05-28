import '../css/app.css'
import Hls from 'hls.js'
import WaveSurfer from 'wavesurfer.js'
import RegionsPlugin from 'wavesurfer.js/dist/plugins/regions.js'

const AUTH_TOKEN_STORAGE_KEY = 'music-ai-auth-token'
const AUTH_USER_STORAGE_KEY = 'music-ai-auth-user'
const RADIO_LISTENER_CLIENT_STORAGE_KEY = 'public-radio-listener-client-id'
const RADIO_VOLUME_COOKIE_NAME = 'public-radio-volume'
const RADIO_BITRATE_COOKIE_NAME = 'public-radio-bitrate'
const RADIO_TRANSPORT_COOKIE_NAME = 'public-radio-transport'
const RADIO_ACCESS_TOKEN_QUERY_KEY = 'access_token'
const RADIO_BUILD_STAMP = 'public-radio-2026-03-28-radio-fix-2'
const FRIENDS_SHARE_DRAFT_STORAGE_KEY = 'music-ai-friends-share-draft'
const DEFAULT_RADIO_VOLUME = 0.15
const DEFAULT_RADIO_BITRATE = 'auto'
const DEFAULT_RADIO_TRANSPORT_MODE = 'direct'
const RADIO_DIAGNOSTICS_HISTORY_LIMIT = 80
const RADIO_DIAGNOSTICS_BATCH_LIMIT = 20
const RADIO_DIAGNOSTICS_FLUSH_DELAY_MS = 2000
const RADIO_DIAGNOSTICS_SAMPLING_WINDOWS_MS: Partial<Record<string, number>> = {
    'loadedmetadata': 2500,
    'fine-sync-rate': 10000,
    'soft-sync-rate': 6000,
    'hard-sync-seek': 4000,
}
const LISTENER_PRESENCE_INTERVAL_MS = 10000
const SOFT_SYNC_DRIFT_SECONDS = 1.15
const HARD_SYNC_DRIFT_SECONDS = 6
const LIVE_SYNC_MIN_INTERVAL_MS = 2500
const FINE_SYNC_DRIFT_SECONDS = 0.35
const VISIBLE_STATION_COUNT = 7
const VISIBLE_WAVE_GENRE_COUNT = 8
const PERSONAL_WAVE_GENRE_FLUSH_DELAY_MS = 280
const SEGMENT_TYPE_LABELS: Record<string, string> = {
    intro: 'INTRO',
    melodic: 'MELODIC',
    outro: 'OUTRO',
    breakdown: 'BREAKDOWN',
    break: 'BREAK',
    buildup: 'BUILD-UP',
    build_up: 'BUILD-UP',
    drop: 'DROP',
    drop_bass: 'DROP BASS',
    verse: 'VERSE',
    chorus: 'CHORUS',
    bridge: 'BRIDGE',
    vocal: 'VOCAL',
    instrumental: 'INSTRUMENTAL',
    hook: 'HOOK',
    climax: 'CLIMAX',
    groove: 'GROOVE',
    ambient: 'AMBIENT',
    bass: 'BASS',
    lead: 'LEAD',
    theme: 'THEME',
    segment: 'SEGMENT',
    unknown: 'SEGMENT',
}

interface StationDescriptor {
    slug: string
    name: string
    description?: string
    public_url?: string
    embed_url?: string
    genres?: string[]
    available_genres?: string[]
    can_manage?: boolean
    is_personal?: boolean
    owner?: {
        id: number
        name: string
        email: string
    } | null
    selected?: boolean
}

interface StationTrack {
    track_id: number
    title: string
    artist: string | null
    duration: number
    offset_seconds: number
    remaining_seconds: number
    stream_url: string
    waveform_url?: string
    hls_master_url?: string
    started_at: string
}

interface StationSnapshot {
    station?: StationDescriptor
    stations?: StationDescriptor[]
    listener_count?: number
    viewer?: { id?: number; name?: string; email?: string } | null
    generated_at?: string
    now_playing?: StationTrack | null
    now_playing_waveform?: TrackWaveformPayload | null
    up_next?: StationTrack[]
    history?: StationTrack[]
}

function hasPlayableQueue(snapshot: StationSnapshot | null | undefined): boolean {
    return Boolean(
        snapshot?.now_playing
        || (Array.isArray(snapshot?.up_next) && snapshot.up_next.length > 0),
    )
}

interface WaveStylePreset {
    slug: string
    label: string
    description: string
    genres: string[]
}

interface RadioSegment {
    id: number
    start: number
    end: number
    type?: string | null
    display_label?: string | null
    importance?: number | null
    rating?: number | null
    is_liked?: boolean | null
    pattern_name?: string | null
}

interface PrivateOverlayPayload {
    track_id: number
    track_rating?: number | null
    explicit_track_rating?: number | null
    computed_track_rating?: number | null
    preference_score?: number | null
    sentiment?: string | null
    segments?: RadioSegment[]
}

interface TrackWaveformPayload {
    duration?: number
    peaks?: number[]
}

interface RadioChatUser {
    id: number
    name: string
}

interface RadioChatSegmentShare {
    trackId: number
    trackTitle: string
    artist?: string | null
    segmentId: number
    label: string
    start: number
    end: number
}

interface RadioChatEntry {
    id: string
    type: 'message' | 'segment_share'
    stationSlug: string
    user: RadioChatUser
    text?: string
    segment?: RadioChatSegmentShare
    createdAt: string
}

const body = document.body
body.dataset.radioBuildStamp = RADIO_BUILD_STAMP
console.info('[public-radio]', RADIO_BUILD_STAMP, 'module-loaded')
const slug = body.dataset.radioSlug || 'test'
const searchParams = new URLSearchParams(window.location.search)
const isPersonalStation = /^personal-\d+$/i.test(slug)

const audio = document.getElementById('radio-audio') as HTMLAudioElement | null
const loading = document.getElementById('radio-loading') as HTMLDivElement | null
const error = document.getElementById('radio-error') as HTMLDivElement | null
const playToggleButton = document.getElementById('radio-play-toggle') as HTMLButtonElement | null
const muteToggleButton = document.getElementById('radio-mute-toggle') as HTMLButtonElement | null
const volumeInput = document.getElementById('radio-volume') as HTMLInputElement | null
const bitrateSelect = document.getElementById('radio-bitrate') as HTMLSelectElement | null
const transportSelect = document.getElementById('radio-transport') as HTMLSelectElement | null
const timeLabel = document.getElementById('radio-time') as HTMLSpanElement | null
const nowTitle = document.getElementById('radio-now-title') as HTMLDivElement | null
const nowArtist = document.getElementById('radio-now-artist') as HTMLDivElement | null
const stationName = document.getElementById('radio-station-name') as HTMLHeadingElement | null
const stationDescription = document.getElementById('radio-station-description') as HTMLParagraphElement | null
const publicUrl = document.getElementById('radio-public-url') as HTMLDivElement | null
const embedUrl = document.getElementById('radio-embed-url') as HTMLDivElement | null
const stationsContainer = document.getElementById('radio-stations') as HTMLDivElement | null
const listenerCount = document.getElementById('radio-listener-count') as HTMLDivElement | null
const remaining = document.getElementById('radio-remaining') as HTMLSpanElement | null
const shareTrackButton = document.getElementById('radio-direct-share-track') as HTMLButtonElement | null
const upNext = document.getElementById('radio-up-next') as HTMLDivElement | null
const history = document.getElementById('radio-history') as HTMLDivElement | null
const feedbackAuth = document.getElementById('radio-feedback-auth') as HTMLDivElement | null
const feedbackPanel = document.getElementById('radio-feedback-panel') as HTMLDivElement | null
const trackFitValue = document.getElementById('radio-track-fit-value') as HTMLDivElement | null
const trackRatingValue = document.getElementById('radio-track-rating-value') as HTMLDivElement | null
const trackStars = document.getElementById('radio-track-stars') as HTMLDivElement | null
const segmentsStatus = document.getElementById('radio-segments-status') as HTMLDivElement | null
const segmentsContainer = document.getElementById('radio-segments') as HTMLDivElement | null
const waveGenresShell = document.getElementById('radio-wave-genres-shell') as HTMLDivElement | null
const waveGenresStatus = document.getElementById('radio-wave-genres-status') as HTMLDivElement | null
const waveGenresPanel = document.getElementById('radio-wave-genres-panel') as HTMLDivElement | null
const chatStatus = document.getElementById('radio-chat-status') as HTMLDivElement | null
const chatAuth = document.getElementById('radio-chat-auth') as HTMLDivElement | null
const chatPanel = document.getElementById('radio-chat-panel') as HTMLDivElement | null
const chatList = document.getElementById('radio-chat-list') as HTMLDivElement | null
const chatForm = document.getElementById('radio-chat-form') as HTMLFormElement | null
const chatInput = document.getElementById('radio-chat-input') as HTMLInputElement | null
const chatSubmit = document.getElementById('radio-chat-submit') as HTMLButtonElement | null
const waveformStatus = document.getElementById('radio-waveform-status') as HTMLDivElement | null
const waveformRoot = document.getElementById('radio-waveform') as HTMLDivElement | null

let snapshot: StationSnapshot | null = null
let pendingSeek: number | null = null
let authToken: string | null = null
let radioAccessToken: string | null = searchParams.get(RADIO_ACCESS_TOKEN_QUERY_KEY)
let radioAccessTokenExpiresAt = radioAccessToken ? Number.POSITIVE_INFINITY : 0
let radioAccessTokenPromise: Promise<string | null> | null = null
let authUser: { id?: number; name?: string; email?: string } | null = null
let feedbackTrackId: number | null = null
let activeSegments: RadioSegment[] = []
let wavesurfer: WaveSurfer | null = null
let regions: ReturnType<typeof RegionsPlugin.create> | null = null
let hls: Hls | null = null
let currentPlaybackMode: 'direct' | 'hls-js' | 'hls-native' | null = null
let waveformTrackId: number | null = null
let waveformPayloadTrackId: number | null = null
let currentWaveformPayload: TrackWaveformPayload | null = null
let waveformLoadRequestId = 0
let activeWaveformSegmentId: number | null = null
const prewarmedUpcomingTrackIds = new Set<number>()
const prewarmingUpcomingTrackIds = new Map<number, Promise<void>>()
let listenerSocket: WebSocket | null = null
let listenerReconnectTimer: number | null = null
let listenerPresenceTimer: number | null = null
let listenerSocketOpen = false
let listenerSocketConnectPromise: Promise<void> | null = null
let privateSocket: WebSocket | null = null
let privateSocketOpen = false
let privateReconnectTimer: number | null = null
let previousVolume = DEFAULT_RADIO_VOLUME
let selectedBitrate = DEFAULT_RADIO_BITRATE
let selectedTransportMode: RadioTransportMode = DEFAULT_RADIO_TRANSPORT_MODE
let shuttingDown = false
let shouldResumeAfterSourceLoad = false
let playbackIntent = false
let autoplayBootstrapAttempted = false
let suppressPausePresence = false
let lastLiveSyncAt = 0
const listenerClientId = readOrCreateListenerClientId()

const waveformRegions = new Map<number, any>()
let chatEntries: RadioChatEntry[] = []
let stationsExpanded = false
let waveGenresExpanded = false
let pendingPersonalWaveGenres: string[] | null = null
let personalWaveGenresFlushTimer: number | null = null
let personalWaveGenresFlushScheduled = false
let personalWaveGenresRequestInFlight = false
let currentListeningHistoryId: number | null = null
let currentListeningHistoryTrackId: number | null = null
let currentListeningStartedAtMs = 0
let playbackRecoveryTimer: number | null = null
let lastRadioDiagnosticsEvent = 'init'
let nativeHlsFallbackTrackId: number | null = null
let diagnosticsFlushTimer: number | null = null
let diagnosticsFlushInFlight = false
let pendingRadioDiagnostics: RadioDiagnosticsSnapshot[] = []
const sampledDiagnosticsEventsAt = new Map<string, number>()

type RadioDiagnosticsSnapshot = {
    event: string
    timestamp: string
    playbackMode: string
    transportPreference: RadioTransportMode
    selectedBitrate: string
    trackId: number | null
    currentSrc: string
    streamUrl: string
    hlsSourceUrl: string
    currentTime: number
    liveOffset: number | null
    drift: number | null
    playbackRate: number
    mediaErrorCode: number | null
    waiting: boolean
    paused: boolean
    readyState: number | null
}

type RadioTransportMode = 'direct' | 'hls'

type RadioDiagnosticsBuffer = {
    current: RadioDiagnosticsSnapshot | null
    history: RadioDiagnosticsSnapshot[]
}

function canUseNativeHlsPlayback() {
    return audio?.canPlayType('application/vnd.apple.mpegurl') !== ''
}

function shouldPreferDirectTransportOverNativeHls() {
    return !Hls.isSupported() && canUseNativeHlsPlayback()
}

function detectPlaybackCapability() {
    if (Hls.isSupported()) {
        return 'hls-js'
    }

    if (shouldPreferDirectTransportOverNativeHls()) {
        return 'direct-preferred-over-native-hls'
    }

    if (canUseNativeHlsPlayback()) {
        return 'hls-native'
    }

    return 'direct'
}

function updateRuntimeDiagnostics() {
    body.dataset.radioPlaybackCapability = detectPlaybackCapability()
    body.dataset.radioPlaybackMode = currentPlaybackMode || 'idle'
    body.dataset.radioTransportPreference = selectedTransportMode
    body.dataset.radioWaveformTransport = 'precomputed-peaks'
    body.dataset.radioBitrate = selectedBitrate
    body.dataset.radioWaveformTrackId = waveformTrackId === null ? '' : String(waveformTrackId)
    body.dataset.radioLastEvent = lastRadioDiagnosticsEvent
}

function publishRadioDiagnostics(event: string, current: StationTrack | null = snapshot?.now_playing || null) {
    lastRadioDiagnosticsEvent = event
    const liveOffset = current ? deriveLiveOffsetSeconds(current) : null
    const currentTime = audio && Number.isFinite(audio.currentTime) ? audio.currentTime : 0
    const drift = liveOffset === null ? null : Math.abs(currentTime - liveOffset)
    const payload: RadioDiagnosticsSnapshot = {
        event,
        timestamp: new Date().toISOString(),
        playbackMode: currentPlaybackMode || 'idle',
        transportPreference: selectedTransportMode,
        selectedBitrate,
        trackId: current ? Number(current.track_id) : null,
        currentSrc: audio?.currentSrc || '',
        streamUrl: audio?.dataset.streamUrl || '',
        hlsSourceUrl: audio?.dataset.hlsSourceUrl || '',
        currentTime: Number(currentTime.toFixed(3)),
        liveOffset: liveOffset === null ? null : Number(liveOffset.toFixed(3)),
        drift: drift === null ? null : Number(drift.toFixed(3)),
        playbackRate: audio ? Number(audio.playbackRate.toFixed(3)) : 1,
        mediaErrorCode: audio?.error?.code ?? null,
        waiting: audio ? audio.readyState < HTMLMediaElement.HAVE_FUTURE_DATA : false,
        paused: audio ? audio.paused : true,
        readyState: audio ? audio.readyState : null,
    }

    body.dataset.radioDrift = payload.drift === null ? '' : String(payload.drift)
    body.dataset.radioCurrentSrc = payload.currentSrc
    updateRuntimeDiagnostics()
    const diagnosticsWindow = window as Window & { __PUBLIC_RADIO_DIAGNOSTICS__?: RadioDiagnosticsBuffer }
    const diagnostics = diagnosticsWindow.__PUBLIC_RADIO_DIAGNOSTICS__ && Array.isArray(diagnosticsWindow.__PUBLIC_RADIO_DIAGNOSTICS__.history)
        ? diagnosticsWindow.__PUBLIC_RADIO_DIAGNOSTICS__
        : { current: null, history: [] }

    diagnostics.current = payload
    diagnostics.history.push(payload)
    if (diagnostics.history.length > RADIO_DIAGNOSTICS_HISTORY_LIMIT) {
        diagnostics.history.splice(0, diagnostics.history.length - RADIO_DIAGNOSTICS_HISTORY_LIMIT)
    }

    diagnosticsWindow.__PUBLIC_RADIO_DIAGNOSTICS__ = diagnostics

    if (!shouldEnqueueRadioDiagnostics(payload)) {
        return
    }

    pendingRadioDiagnostics.push(payload)
    scheduleRadioDiagnosticsFlush(shouldFlushDiagnosticsImmediately(event))
}

function shouldEnqueueRadioDiagnostics(payload: RadioDiagnosticsSnapshot) {
    const samplingWindowMs = RADIO_DIAGNOSTICS_SAMPLING_WINDOWS_MS[payload.event]
    if (!samplingWindowMs) {
        return true
    }

    const samplingKey = `${payload.event}:${payload.trackId ?? 0}:${payload.playbackMode}:${payload.transportPreference}`
    const now = Date.now()
    const lastLoggedAt = sampledDiagnosticsEventsAt.get(samplingKey) ?? 0

    if (now - lastLoggedAt < samplingWindowMs) {
        return false
    }

    sampledDiagnosticsEventsAt.set(samplingKey, now)
    return true
}

function clearRadioDiagnosticsFlushTimer() {
    if (diagnosticsFlushTimer !== null) {
        window.clearTimeout(diagnosticsFlushTimer)
        diagnosticsFlushTimer = null
    }
}

function shouldFlushDiagnosticsImmediately(event: string) {
    return event === 'transport-change'
        || event === 'bitrate-change'
        || event === 'waiting'
        || event === 'stalled'
        || event === 'audio-error'
        || event === 'native-hls-fallback-armed'
        || event.startsWith('hls-error:')
        || event.startsWith('recovery:')
}

function radioDiagnosticsHeaders(): HeadersInit {
    const headers = apiHeaders(true) as Record<string, string>

    if (radioAccessToken) {
        headers['X-Radio-Access-Token'] = radioAccessToken
    }

    return headers
}

function buildRadioDiagnosticsUrl() {
    const url = new URL(`/api/public/radio/${encodeURIComponent(slug)}/diagnostics`, window.location.origin)

    if (radioAccessToken) {
        url.searchParams.set(RADIO_ACCESS_TOKEN_QUERY_KEY, radioAccessToken)
    }

    return url.toString()
}

function scheduleRadioDiagnosticsFlush(immediate = false) {
    if (pendingRadioDiagnostics.length === 0) {
        return
    }

    if (immediate) {
        clearRadioDiagnosticsFlushTimer()
        void flushRadioDiagnostics(true)
        return
    }

    if (pendingRadioDiagnostics.length >= RADIO_DIAGNOSTICS_BATCH_LIMIT) {
        clearRadioDiagnosticsFlushTimer()
        void flushRadioDiagnostics(true)
        return
    }

    if (diagnosticsFlushTimer !== null) {
        return
    }

    diagnosticsFlushTimer = window.setTimeout(() => {
        diagnosticsFlushTimer = null
        void flushRadioDiagnostics()
    }, RADIO_DIAGNOSTICS_FLUSH_DELAY_MS)
}

async function flushRadioDiagnostics(force = false) {
    if (diagnosticsFlushInFlight || pendingRadioDiagnostics.length === 0) {
        return
    }

    if (!force && document.hidden && pendingRadioDiagnostics.length < RADIO_DIAGNOSTICS_BATCH_LIMIT) {
        return
    }

    diagnosticsFlushInFlight = true
    clearRadioDiagnosticsFlushTimer()
    const batch = pendingRadioDiagnostics.splice(0, RADIO_DIAGNOSTICS_BATCH_LIMIT)
    let shouldRetry = false
    let delivered = false

    try {
        const response = await fetch(buildRadioDiagnosticsUrl(), {
            method: 'POST',
            headers: radioDiagnosticsHeaders(),
            body: JSON.stringify({
                client_id: listenerClientId,
                events: batch,
            }),
            keepalive: true,
        })

        if (!response.ok) {
            pendingRadioDiagnostics = [...batch, ...pendingRadioDiagnostics]
            shouldRetry = true
        } else {
            delivered = true
        }
    } catch {
        pendingRadioDiagnostics = [...batch, ...pendingRadioDiagnostics]
        shouldRetry = true
    } finally {
        diagnosticsFlushInFlight = false
    }

    if (shouldRetry) {
        scheduleRadioDiagnosticsFlush(true)
        return
    }

    if (delivered && pendingRadioDiagnostics.length > 0) {
        scheduleRadioDiagnosticsFlush(force)
    }
}

function clearPlaybackRecoveryTimer() {
    if (playbackRecoveryTimer !== null) {
        window.clearTimeout(playbackRecoveryTimer)
        playbackRecoveryTimer = null
    }
}

function schedulePlaybackRecovery(reason: string) {
    if (!audio || !snapshot?.now_playing || playbackRecoveryTimer !== null || !shouldAutoResumePlayback()) {
        return
    }

    playbackRecoveryTimer = window.setTimeout(() => {
        playbackRecoveryTimer = null

        if (!audio || !snapshot?.now_playing || !shouldAutoResumePlayback()) {
            return
        }

        const current = snapshot.now_playing
        const liveOffsetSeconds = deriveLiveOffsetSeconds(current)
        const preservedSeekTarget = Number.isFinite(audio.currentTime) && audio.currentTime > 0
            ? Math.min(current.duration, Math.max(audio.currentTime, liveOffsetSeconds))
            : liveOffsetSeconds

        publishRadioDiagnostics(`recovery:${reason}`, current)
        loadTrackSource(current, true, preservedSeekTarget)
    }, 900)
}

function applyPitchPreservation() {
    if (!audio) {
        return
    }

    const pitchAwareAudio = audio as HTMLAudioElement & {
        preservesPitch?: boolean
        mozPreservesPitch?: boolean
        webkitPreservesPitch?: boolean
    }

    if (typeof pitchAwareAudio.preservesPitch === 'boolean') {
        pitchAwareAudio.preservesPitch = true
    }

    if (typeof pitchAwareAudio.mozPreservesPitch === 'boolean') {
        pitchAwareAudio.mozPreservesPitch = true
    }

    if (typeof pitchAwareAudio.webkitPreservesPitch === 'boolean') {
        pitchAwareAudio.webkitPreservesPitch = true
    }
}

function readStoredUser(): { id?: number; name?: string; email?: string } | null {
    const raw = window.localStorage.getItem(AUTH_USER_STORAGE_KEY)
    if (!raw) return null

    try {
        return JSON.parse(raw) as { id?: number; name?: string; email?: string }
    } catch {
        return null
    }
}

function applyViewerState(viewer: { id?: number; name?: string; email?: string } | null | undefined) {
    authUser = viewer && typeof viewer === 'object' ? viewer : null

    if (authUser) {
        window.localStorage.setItem(AUTH_USER_STORAGE_KEY, JSON.stringify(authUser))
        setFeedbackVisible(true)
        setChatVisible(true)
        renderWaveGenres(snapshot?.station)
        if (feedbackAuth) {
            feedbackAuth.textContent = authUser.name
                ? `Signed in as ${authUser.name}`
                : 'Signed in'
        }
        if (chatAuth) {
            chatAuth.textContent = authUser.name
                ? `Signed in as ${authUser.name}`
                : 'Signed in'
        }
        updateChatStatus('Connected')
        requestChatHistory()
    } else {
        window.localStorage.removeItem(AUTH_USER_STORAGE_KEY)
        setFeedbackVisible(false)
        setChatVisible(false)
        renderWaveGenres(undefined)
        renderTrackFit(null)
        if (feedbackAuth) {
            feedbackAuth.textContent = authToken
                ? 'Sign in to rate the current track and its segments.'
                : 'Sign in to rate the current track and its segments.'
        }
        if (chatAuth) {
            chatAuth.textContent = 'Sign in to chat and share segments with other listeners.'
        }
        updateChatStatus('Sign in to join the chat.')
        setChatEntries([])
    }

    renderStations(snapshot?.stations)
}

function readOrCreateListenerClientId() {
    const existing = window.sessionStorage.getItem(RADIO_LISTENER_CLIENT_STORAGE_KEY)
    if (existing) {
        return existing
    }

    const nextId = window.crypto?.randomUUID?.() || `radio-${Date.now()}-${Math.random().toString(16).slice(2)}`
    window.sessionStorage.setItem(RADIO_LISTENER_CLIENT_STORAGE_KEY, nextId)
    return nextId
}

function readCookie(name: string) {
    const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const match = document.cookie.match(new RegExp(`(?:^|; )${escapedName}=([^;]*)`))
    return match ? decodeURIComponent(match[1]) : null
}

function persistVolumePreference(volume: number) {
    const normalized = Math.max(0, Math.min(1, volume))
    const secure = window.location.protocol === 'https:' ? '; Secure' : ''
    document.cookie = `${RADIO_VOLUME_COOKIE_NAME}=${encodeURIComponent(normalized.toFixed(2))}; Max-Age=31536000; Path=/; SameSite=Lax${secure}`
}

function readVolumePreference() {
    const raw = readCookie(RADIO_VOLUME_COOKIE_NAME)
    if (raw === null) {
        return DEFAULT_RADIO_VOLUME
    }

    const parsed = Number(raw)
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
        return DEFAULT_RADIO_VOLUME
    }

    return parsed
}

function normalizeBitratePreference(value: string | null | undefined) {
    const normalized = String(value || '').trim().toLowerCase()
    const aliases: Record<string, string> = {
        source: 'auto',
        original: 'auto',
    }

    if (aliases[normalized]) {
        return aliases[normalized]
    }

    if (['auto', '32', '48', '64', '96', '128', '160', '192', '256', '320'].includes(normalized)) {
        return normalized
    }

    return DEFAULT_RADIO_BITRATE
}

function defaultTransportPreference(): RadioTransportMode {
    if (Hls.isSupported()) {
        return 'hls'
    }

    return DEFAULT_RADIO_TRANSPORT_MODE
}

function normalizeTransportPreference(value: string | null | undefined): RadioTransportMode {
    return String(value || '').trim().toLowerCase() === 'hls' ? 'hls' : 'direct'
}

function persistTransportPreference(value: RadioTransportMode) {
    const secure = window.location.protocol === 'https:' ? '; Secure' : ''
    const normalized = normalizeTransportPreference(value)
    document.cookie = `${RADIO_TRANSPORT_COOKIE_NAME}=${encodeURIComponent(normalized)}; Max-Age=31536000; Path=/; SameSite=Lax${secure}`
}

function readTransportPreference(): RadioTransportMode {
    const raw = readCookie(RADIO_TRANSPORT_COOKIE_NAME)
    if (raw === null) {
        return defaultTransportPreference()
    }

    return normalizeTransportPreference(raw)
}

function persistBitratePreference(value: string) {
    const secure = window.location.protocol === 'https:' ? '; Secure' : ''
    const normalized = normalizeBitratePreference(value)
    document.cookie = `${RADIO_BITRATE_COOKIE_NAME}=${encodeURIComponent(normalized)}; Max-Age=31536000; Path=/; SameSite=Lax${secure}`
}

function readBitratePreference() {
    return normalizeBitratePreference(readCookie(RADIO_BITRATE_COOKIE_NAME))
}

async function ensureRadioAccessToken(forceRefresh = false): Promise<string | null> {
    if (!forceRefresh && radioAccessToken) {
        return radioAccessToken
    }

    radioAccessToken = authToken || searchParams.get(RADIO_ACCESS_TOKEN_QUERY_KEY) || null
    radioAccessTokenExpiresAt = radioAccessToken ? Number.POSITIVE_INFINITY : 0
    radioAccessTokenPromise = Promise.resolve(radioAccessToken)
    const nextToken = await radioAccessTokenPromise
    radioAccessTokenPromise = null
    return nextToken
}

function apiHeaders(includeJson: boolean): HeadersInit {
    const headers: Record<string, string> = {
        Accept: 'application/json',
    }

    if (authToken) {
        headers.Authorization = `Bearer ${authToken}`
    }

    if (includeJson) {
        headers['Content-Type'] = 'application/json'
    }

    if (radioAccessToken) {
        headers['X-Radio-Access-Token'] = radioAccessToken
    }

    return headers
}

function resetListeningHistoryState() {
    currentListeningHistoryId = null
    currentListeningHistoryTrackId = null
    currentListeningStartedAtMs = 0
}

async function completeListeningHistory(completed = false) {
    if (currentListeningHistoryId === null) {
        resetListeningHistoryState()
        return
    }

    if (!authToken) {
        resetListeningHistoryState()
        return
    }

    const historyId = currentListeningHistoryId
    const startedAtMs = currentListeningStartedAtMs
    resetListeningHistoryState()

    const durationPlayed = startedAtMs > 0
        ? Math.max(0, (Date.now() - startedAtMs) / 1000)
        : 0

    try {
        await fetch(`/api/listening-history/${historyId}/complete`, {
            method: 'PUT',
            headers: apiHeaders(true),
            body: JSON.stringify({
                duration_played: Number(durationPlayed.toFixed(3)),
                completed,
            }),
            keepalive: true,
        })
    } catch {
        // Ignore transient network failures for public radio listen tracking.
    }
}

async function startListeningHistory(trackId: number) {
    if (!authToken || trackId <= 0) {
        resetListeningHistoryState()
        return
    }

    if (currentListeningHistoryId !== null && currentListeningHistoryTrackId === trackId) {
        return
    }

    if (currentListeningHistoryId !== null) {
        await completeListeningHistory(false)
    }

    try {
        const response = await fetch(`/api/tracks/${trackId}/play`, {
            method: 'POST',
            headers: apiHeaders(true),
            body: JSON.stringify({ source: 'radio' }),
        })

        if (!response.ok) {
            return
        }

        const payload = await response.json() as {
            history?: {
                id?: number
            }
        }
        const historyId = Number(payload.history?.id || 0)

        if (!historyId) {
            return
        }

        currentListeningHistoryId = historyId
        currentListeningHistoryTrackId = trackId
        currentListeningStartedAtMs = Date.now()
    } catch {
        resetListeningHistoryState()
    }
}

function formatTime(totalSeconds: number): string {
    if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) return '0:00'
    const minutes = Math.floor(totalSeconds / 60)
    const seconds = Math.floor(totalSeconds % 60)
    return `${minutes}:${String(seconds).padStart(2, '0')}`
}

function toDisplayUrl(pathOrUrl: string | undefined) {
    if (!pathOrUrl) return ''

    try {
        return new URL(pathOrUrl, window.location.origin).toString()
    } catch {
        return pathOrUrl
    }
}

function setError(message: string) {
    if (!error) return
    error.textContent = message
    error.classList.remove('hidden')
}

function clearError() {
    if (!error) return
    error.textContent = ''
    error.classList.add('hidden')
}

function setWaveformStatus(message: string, hidden = false) {
    if (!waveformStatus) return
    waveformStatus.textContent = message
    waveformStatus.classList.toggle('hidden', hidden)
}

function setFeedbackVisible(visible: boolean) {
    if (!feedbackPanel || !feedbackAuth) return
    feedbackPanel.classList.toggle('hidden', !visible)
    feedbackAuth.classList.toggle('hidden', visible)
}

function setChatVisible(visible: boolean) {
    if (!chatPanel || !chatAuth) return
    chatPanel.classList.toggle('hidden', !visible)
    chatAuth.classList.toggle('hidden', visible)
}

function setWaveGenresVisible(visible: boolean) {
    if (!waveGenresShell) return
    waveGenresShell.classList.toggle('hidden', !visible)
}

function updateChatStatus(message: string) {
    if (!chatStatus) return
    chatStatus.textContent = message
}

function updateWaveGenresStatus(message: string) {
    if (!waveGenresStatus) return
    waveGenresStatus.textContent = message
}

function escapeHtml(value: string) {
    return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;')
}

function formatChatTimestamp(isoDate: string) {
    const timestamp = new Date(isoDate)
    if (Number.isNaN(timestamp.getTime())) {
        return ''
    }

    return timestamp.toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit',
    })
}

function normalizeChatEntries(entries: unknown) {
    if (!Array.isArray(entries)) {
        return []
    }

    return entries.filter((entry): entry is RadioChatEntry => {
        if (!entry || typeof entry !== 'object') return false
        const candidate = entry as Partial<RadioChatEntry>
        return typeof candidate.id === 'string'
            && typeof candidate.stationSlug === 'string'
            && typeof candidate.createdAt === 'string'
            && candidate.user !== undefined
    })
}

function normalizeGenreName(value: string | null | undefined) {
    return String(value || '').trim().toLowerCase()
}

function normalizeGenreSelection(preferredGenres: string[], availableGenres: string[]) {
    const requestedGenres = new Set(
        preferredGenres
            .map((genre) => normalizeGenreName(genre))
            .filter(Boolean),
    )
    const normalizedSelection: string[] = []

    availableGenres.forEach((genre) => {
        const normalizedGenre = normalizeGenreName(genre)
        if (!normalizedGenre || !requestedGenres.has(normalizedGenre) || normalizedSelection.includes(normalizedGenre)) {
            return
        }

        normalizedSelection.push(normalizedGenre)
    })

    return normalizedSelection
}

function sameGenreSelection(left: string[] | null | undefined, right: string[] | null | undefined) {
    const leftSelection = Array.isArray(left) ? [...left].sort() : []
    const rightSelection = Array.isArray(right) ? [...right].sort() : []

    if (leftSelection.length !== rightSelection.length) {
        return false
    }

    return leftSelection.every((genre, index) => genre === rightSelection[index])
}

function formatWaveStylePresetLabel(name: string) {
    return String(name || '')
        .replace(/\s+radio$/i, '')
        .trim()
}

function getWaveStylePresets(station: StationDescriptor | undefined) {
    const availableGenres = Array.isArray(station?.available_genres) ? station.available_genres : []
    if (availableGenres.length === 0 || !Array.isArray(snapshot?.stations)) {
        return []
    }

    const availableSelection = normalizeGenreSelection(availableGenres, availableGenres)
    const seenKeys = new Set<string>()
    const presets: WaveStylePreset[] = []

    snapshot.stations.forEach((candidate) => {
        if (candidate.is_personal || candidate.slug === 'test' || !Array.isArray(candidate.genres) || candidate.genres.length <= 1) {
            return
        }

        const genres = normalizeGenreSelection(candidate.genres, availableSelection)
        if (genres.length <= 1) {
            return
        }

        const dedupeKey = genres.slice().sort().join('|')
        if (!dedupeKey || seenKeys.has(dedupeKey)) {
            return
        }

        seenKeys.add(dedupeKey)
        presets.push({
            slug: candidate.slug,
            label: formatWaveStylePresetLabel(candidate.name),
            description: String(candidate.description || '').trim(),
            genres,
        })
    })

    return presets
}

function applyLocalPersonalStationGenres(preferredGenres: string[]) {
    if (!snapshot?.station?.is_personal) {
        return
    }

    snapshot.station.genres = [...preferredGenres]

    if (!Array.isArray(snapshot.stations)) {
        renderWaveGenres(snapshot.station)
        return
    }

    const currentStation = snapshot.stations.find((candidate) => candidate.slug === snapshot?.station?.slug)
    if (currentStation) {
        currentStation.genres = [...preferredGenres]
    }

    renderWaveGenres(snapshot.station)
}

function clearPersonalWaveGenresFlushTimer() {
    if (personalWaveGenresFlushTimer !== null) {
        window.clearTimeout(personalWaveGenresFlushTimer)
        personalWaveGenresFlushTimer = null
    }
}

function flushPersonalStationGenres() {
    clearPersonalWaveGenresFlushTimer()
    personalWaveGenresFlushScheduled = false

    if (!pendingPersonalWaveGenres || !snapshot?.station?.is_personal || !snapshot.station.can_manage) {
        return
    }

    const availableGenres = Array.isArray(snapshot.station.available_genres) ? snapshot.station.available_genres : []
    const normalizedSelection = normalizeGenreSelection(pendingPersonalWaveGenres, availableGenres)
    pendingPersonalWaveGenres = [...normalizedSelection]

    if (!listenerSocket || !listenerSocketOpen) {
        updateWaveGenresStatus('Saving your wave genres when the connection is ready...')
        return
    }

    if (personalWaveGenresRequestInFlight) {
        updateWaveGenresStatus('Saving your wave genres...')
        return
    }

    personalWaveGenresRequestInFlight = true
    listenerSocket.send(JSON.stringify({
        type: 'update_personal_station_genres',
        stationSlug: slug,
        clientId: listenerClientId,
        accessToken: radioAccessToken,
        preferredGenres: normalizedSelection,
    }))
    updateWaveGenresStatus('Updating your wave genres...')
}

function schedulePersonalStationGenresFlush() {
    clearPersonalWaveGenresFlushTimer()
    personalWaveGenresFlushScheduled = true
    personalWaveGenresFlushTimer = window.setTimeout(() => {
        flushPersonalStationGenres()
    }, PERSONAL_WAVE_GENRE_FLUSH_DELAY_MS)
}

function sendPersonalStationGenres(preferredGenres: string[]) {
    if (!snapshot?.station?.is_personal || !snapshot.station.can_manage) {
        return
    }

    const availableGenres = Array.isArray(snapshot.station.available_genres) ? snapshot.station.available_genres : []
    const normalizedSelection = normalizeGenreSelection(preferredGenres, availableGenres)

    if (sameGenreSelection(pendingPersonalWaveGenres, normalizedSelection) && personalWaveGenresFlushScheduled) {
        applyLocalPersonalStationGenres(normalizedSelection)
        return
    }

    pendingPersonalWaveGenres = [...normalizedSelection]
    applyLocalPersonalStationGenres(normalizedSelection)
    updateWaveGenresStatus('Saving your wave genres...')
    schedulePersonalStationGenresFlush()
}

function togglePersonalWaveStyle(styleSlug: string) {
    if (!snapshot?.station?.is_personal || !snapshot.station.can_manage) {
        return
    }

    const preset = getWaveStylePresets(snapshot.station).find((candidate) => candidate.slug === styleSlug)
    if (!preset) {
        return
    }

    const availableGenres = Array.isArray(snapshot.station.available_genres) ? snapshot.station.available_genres : []
    const nextSelectedGenres = new Set(normalizeGenreSelection(snapshot.station.genres || [], availableGenres))
    const presetFullyActive = preset.genres.every((genre) => nextSelectedGenres.has(genre))

    if (presetFullyActive) {
        preset.genres.forEach((genre) => {
            nextSelectedGenres.delete(genre)
        })
    } else {
        preset.genres.forEach((genre) => {
            nextSelectedGenres.add(genre)
        })
    }

    sendPersonalStationGenres(Array.from(nextSelectedGenres))
}

function focusSharedSegment(segmentId: number) {
    const target = segmentsContainer?.querySelector<HTMLElement>(`[data-segment-id="${segmentId}"]`)
    if (!target) {
        return
    }

    target.scrollIntoView({ behavior: 'smooth', block: 'center' })
    target.classList.add('ring-2', 'ring-sky-300')
    window.setTimeout(() => {
        target.classList.remove('ring-2', 'ring-sky-300')
    }, 1400)
}

function renderChatEntries() {
    if (!chatList) return

    if (chatEntries.length === 0) {
        chatList.innerHTML = '<p class="text-sm text-slate-500">No chat messages yet.</p>'
        return
    }

    chatList.innerHTML = chatEntries.map((entry) => {
        const userName = escapeHtml(entry.user?.name || `User ${entry.user?.id || ''}`)
        const createdAt = escapeHtml(formatChatTimestamp(entry.createdAt))

        if (entry.type === 'segment_share' && entry.segment) {
            const segment = entry.segment
            const trackTitle = escapeHtml(segment.trackTitle)
            const artist = escapeHtml(segment.artist || 'Unknown artist')
            const label = escapeHtml(segment.label)
            const timeRange = `${formatTime(segment.start)} - ${formatTime(segment.end)}`

            return `
                <article class="rounded-[1rem] border border-slate-200 bg-white px-3 py-3 shadow-sm">
                    <div class="flex items-start justify-between gap-3">
                        <div>
                            <div class="text-sm font-semibold text-slate-950">${userName}</div>
                            <div class="mt-1 text-xs uppercase tracking-[0.16em] text-slate-400">Shared segment</div>
                        </div>
                        <div class="text-xs text-slate-400">${createdAt}</div>
                    </div>
                    <div class="mt-3 rounded-[0.95rem] border border-sky-200 bg-sky-50/80 px-3 py-3">
                        <div class="text-sm font-semibold text-slate-950">${trackTitle}</div>
                        <div class="mt-1 text-xs text-slate-500">${artist}</div>
                        <div class="mt-3 text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">${label}</div>
                        <div class="mt-1 text-sm text-slate-700">${escapeHtml(timeRange)}</div>
                        <button
                            type="button"
                            class="mt-3 rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-600 transition hover:border-sky-200 hover:text-sky-700"
                            data-radio-chat-focus-segment="${segment.segmentId}"
                        >
                            Focus segment
                        </button>
                    </div>
                </article>
            `
        }

        const text = escapeHtml(String(entry.text || ''))
        return `
            <article class="rounded-[1rem] border border-slate-200 bg-white px-3 py-3 shadow-sm">
                <div class="flex items-start justify-between gap-3">
                    <div class="text-sm font-semibold text-slate-950">${userName}</div>
                    <div class="text-xs text-slate-400">${createdAt}</div>
                </div>
                <p class="mt-2 whitespace-pre-wrap break-words text-sm leading-6 text-slate-700">${text}</p>
            </article>
        `
    }).join('')

    chatList.scrollTop = chatList.scrollHeight
}

function setChatEntries(entries: RadioChatEntry[]) {
    chatEntries = entries
    renderChatEntries()
}

function appendChatEntry(entry: RadioChatEntry) {
    if (chatEntries.some((existing) => existing.id === entry.id)) {
        return
    }

    chatEntries = [...chatEntries.slice(-49), entry]
    renderChatEntries()
}

function updateListenerCount(count: number) {
    if (!listenerCount) return
    listenerCount.textContent = String(Math.max(0, count))
}

function renderTrackFit(score: number | null) {
    if (!trackFitValue) return

    if (score === null || !Number.isFinite(score)) {
        trackFitValue.textContent = 'No personal fit score yet'
        return
    }

    const percent = Math.max(0, Math.min(99, Math.round(score * 100)))
    trackFitValue.textContent = `${percent}% fit to your taste`
}

function closePrivateSocket() {
    if (privateReconnectTimer !== null) {
        window.clearTimeout(privateReconnectTimer)
        privateReconnectTimer = null
    }

    privateSocketOpen = false

    if (privateSocket) {
        privateSocket.close()
        privateSocket = null
    }
}

async function connectPrivateSocket() {
    return
}

async function applyWaveformPayload(trackId: number, waveform: TrackWaveformPayload | null | undefined, fallbackDuration: number) {
    if (!wavesurfer || waveformTrackId !== trackId) {
        return
    }

    const duration = Number(waveform?.duration || fallbackDuration || 0)
    const peaks = Array.isArray(waveform?.peaks)
        ? waveform.peaks.map((value) => Number(value)).filter((value) => Number.isFinite(value))
        : []

    if (duration <= 0 || peaks.length === 0) {
        if (waveformPayloadTrackId === trackId) {
            waveformPayloadTrackId = null
        }
        currentWaveformPayload = null
        setWaveformStatus('Failed to load waveform.')
        return
    }

    waveformPayloadTrackId = trackId
    currentWaveformPayload = {
        duration,
        peaks,
    }
    await wavesurfer.load('', [peaks], duration)
}

function closeListenerSocket() {
    if (listenerReconnectTimer !== null) {
        window.clearTimeout(listenerReconnectTimer)
        listenerReconnectTimer = null
    }

    listenerSocketOpen = false

    if (listenerPresenceTimer !== null) {
        window.clearInterval(listenerPresenceTimer)
        listenerPresenceTimer = null
    }

    if (listenerSocket) {
        listenerSocket.close()
        listenerSocket = null
    }
}

function destroyHls() {
    if (!hls) {
        return
    }

    hls.destroy()
    hls = null
}

function unloadTrackSource(resetTrackState = true) {
    if (!audio) {
        return
    }

    const hadTransport = Boolean(hls || audio.currentSrc || audio.dataset.streamUrl || audio.dataset.hlsSourceUrl)
    if (!hadTransport && !resetTrackState) {
        return
    }

    suppressPausePresence = suppressPausePresence || hadTransport
    destroyHls()
    audio.pause()
    audio.removeAttribute('src')
    audio.load()
    audio.playbackRate = 1
    lastLiveSyncAt = 0

    if (resetTrackState) {
        delete audio.dataset.trackId
        delete audio.dataset.hlsMasterUrl
        delete audio.dataset.hlsSourceUrl
        delete audio.dataset.streamUrl
        nativeHlsFallbackTrackId = null
        waveformPayloadTrackId = null
        currentWaveformPayload = null
    }

    pendingSeek = null
    shouldResumeAfterSourceLoad = false
    currentPlaybackMode = null
    clearPlaybackRecoveryTimer()
    updateRuntimeDiagnostics()
}

function isActivelyListening() {
    return Boolean(audio && !audio.paused && !audio.ended && snapshot?.now_playing)
}

function shouldAutoResumePlayback() {
    return playbackIntent && !shuttingDown
}

function shouldMaintainPrivateOverlaySession() {
    return Boolean(authToken && snapshot?.now_playing && shouldAutoResumePlayback())
}

function updatePlayToggleLabel() {
    if (!playToggleButton) return

    if (!snapshot?.now_playing) {
        playToggleButton.textContent = hasPlayableQueue(snapshot) ? 'Waiting for track' : 'No tracks available'
        playToggleButton.disabled = true
        return
    }

    playToggleButton.disabled = false
    playToggleButton.textContent = audio && !audio.paused ? 'Pause' : 'Start radio'
}

function updateVolumeUi() {
    if (!audio) return

    if (volumeInput) {
        volumeInput.value = String(audio.volume)
    }

    if (muteToggleButton) {
        muteToggleButton.textContent = audio.muted || audio.volume === 0 ? 'Unmute' : 'Mute'
    }
}

function updateBitrateUi() {
    if (!bitrateSelect) return
    bitrateSelect.value = selectedBitrate
}

function updateTransportUi() {
    if (!transportSelect) return

    const hlsSupported = Hls.isSupported() || canUseNativeHlsPlayback()
    const hlsOption = transportSelect.querySelector('option[value="hls"]')

    if (hlsOption instanceof HTMLOptionElement) {
        hlsOption.disabled = !hlsSupported
    }

    if (!hlsSupported && selectedTransportMode !== 'direct') {
        selectedTransportMode = 'direct'
        persistTransportPreference(selectedTransportMode)
    }

    transportSelect.value = selectedTransportMode
}

function initializeVolumePreference() {
    if (!audio) return

    const preferredVolume = readVolumePreference()
    audio.volume = preferredVolume
    audio.muted = preferredVolume === 0
    previousVolume = preferredVolume > 0 ? preferredVolume : DEFAULT_RADIO_VOLUME
    persistVolumePreference(preferredVolume)
    updateVolumeUi()
}

function initializeBitratePreference() {
    selectedBitrate = readBitratePreference()
    persistBitratePreference(selectedBitrate)
    updateBitrateUi()
    updateRuntimeDiagnostics()
}

function initializeTransportPreference() {
    selectedTransportMode = readTransportPreference()
    persistTransportPreference(selectedTransportMode)
    updateTransportUi()
    updateRuntimeDiagnostics()
}

function buildStarButtons(kind: 'track' | 'segment', currentValue: number | null, trackId: number, segmentId: number | null) {
    return [1, 2, 3, 4, 5].map((star) => {
        const active = currentValue !== null && currentValue >= star
        const activeClass = active ? 'text-amber-500' : 'text-slate-300 hover:text-amber-400'
        const attrs = [
            `data-radio-rate-kind="${kind}"`,
            `data-track-id="${trackId}"`,
            `data-rating="${star}"`,
        ]

        if (segmentId !== null) {
            attrs.push(`data-segment-id="${segmentId}"`)
        }

        return `<button type="button" class="text-lg font-semibold transition-transform hover:scale-110 ${activeClass}" ${attrs.join(' ')} aria-label="Rate ${star} out of 5">&#9733;</button>`
    }).join('')
}

function buildTrackSentimentButtons(trackId: number, rating: number | null) {
    const buttons: Array<{ label: string; rating: number; activeClass: string }> = [
        { label: 'Dislike', rating: 1, activeClass: 'border-rose-300 bg-rose-50 text-rose-700' },
        { label: 'Like', rating: 5, activeClass: 'border-emerald-300 bg-emerald-50 text-emerald-700' },
    ]

    return buttons.map((button) => {
        const active = rating === button.rating
        const classes = active
            ? button.activeClass
            : 'border-slate-200 bg-white text-slate-500 hover:border-sky-200 hover:text-sky-700'

        return `<button type="button" class="rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] transition ${classes}" data-radio-rate-kind="track" data-track-id="${trackId}" data-rating="${button.rating}">${button.label}</button>`
    }).join('')
}

function formatTrackRatingSummary(rating: number | null, sentiment?: string | null, computedTrackRating?: number | null) {
    if (rating === null) {
        return 'Not rated yet'
    }

    if (sentiment === 'dislike') {
        return 'Dislike'
    }

    if (sentiment === 'love' || sentiment === 'like') {
        return 'Like'
    }

    if (computedTrackRating !== null && Number.isFinite(computedTrackRating)) {
        return `${rating}/5 overall`
    }

    return `${rating}/5`
}

function renderTrackRating(trackId: number, rating: number | null, sentiment?: string | null, computedTrackRating?: number | null) {
    if (!trackRatingValue || !trackStars) return

    trackRatingValue.textContent = formatTrackRatingSummary(rating, sentiment, computedTrackRating)
    trackStars.innerHTML = `
        <div class="flex flex-wrap items-center justify-end gap-2">
            ${buildTrackSentimentButtons(trackId, rating)}
            <div class="flex items-center gap-1">
                ${buildStarButtons('track', rating, trackId, null)}
            </div>
        </div>
    `
}

function getSegmentRegionColor(segment: RadioSegment, active: boolean) {
    if (segment.rating !== null && Number(segment.rating) >= 4) {
        return active ? 'rgba(16, 185, 129, 0.48)' : 'rgba(16, 185, 129, 0.26)'
    }

    if (segment.rating !== null && Number(segment.rating) <= 2) {
        return active ? 'rgba(244, 63, 94, 0.5)' : 'rgba(244, 63, 94, 0.26)'
    }

    if (segment.rating !== null && Number(segment.rating) === 3) {
        return active ? 'rgba(245, 158, 11, 0.48)' : 'rgba(245, 158, 11, 0.24)'
    }

    if (segment.is_liked === true) {
        return active ? 'rgba(16, 185, 129, 0.48)' : 'rgba(16, 185, 129, 0.26)'
    }

    if (segment.is_liked === false) {
        return active ? 'rgba(244, 63, 94, 0.5)' : 'rgba(244, 63, 94, 0.26)'
    }

    return active ? 'rgba(56, 189, 248, 0.36)' : 'rgba(148, 163, 184, 0.2)'
}

function formatSegmentTypeLabel(type: string | null | undefined) {
    const normalized = String(type || 'segment').trim().toLowerCase()
    if (!normalized) {
        return 'SEGMENT'
    }

    if (SEGMENT_TYPE_LABELS[normalized]) {
        return SEGMENT_TYPE_LABELS[normalized]
    }

    return normalized
        .split('_')
        .filter(Boolean)
        .map((part) => part.toUpperCase())
        .join(' ') || 'SEGMENT'
}

function clearWaveformSegments() {
    waveformRegions.forEach((region) => {
        try {
            region.remove()
        } catch {
            // Ignore region cleanup failures during track switches.
        }
    })
    waveformRegions.clear()
    activeWaveformSegmentId = null
}

function syncWaveformPosition() {
    if (!wavesurfer || !audio) return
    const nowPlaying = snapshot?.now_playing
    if (!nowPlaying?.duration) return

    const currentTime = audio.currentTime || 0
    const waveformTime = wavesurfer.getCurrentTime()
    if (Math.abs(waveformTime - currentTime) > 0.6) {
        wavesurfer.seekTo(Math.max(0, Math.min(1, currentTime / nowPlaying.duration)))
    }
}

function setRegionActive(segmentId: number | null) {
    if (segmentId === activeWaveformSegmentId) return

    if (activeWaveformSegmentId !== null) {
        const previousSegment = activeSegments.find((segment) => Number(segment.id) === activeWaveformSegmentId)
        const previousRegion = waveformRegions.get(activeWaveformSegmentId)
        if (previousSegment && previousRegion) {
            previousRegion.setOptions({ color: getSegmentRegionColor(previousSegment, false) })
            previousRegion.element?.classList.remove('ring-2', 'ring-sky-300')
        }
    }

    if (segmentId !== null) {
        const nextSegment = activeSegments.find((segment) => Number(segment.id) === segmentId)
        const nextRegion = waveformRegions.get(segmentId)
        if (nextSegment && nextRegion) {
            nextRegion.setOptions({ color: getSegmentRegionColor(nextSegment, true) })
            nextRegion.element?.classList.add('ring-2', 'ring-sky-300')
        }
    }

    activeWaveformSegmentId = segmentId
}

function updateActiveSegmentHighlight() {
    if (!segmentsContainer || !audio) return
    const currentTime = audio.currentTime || 0
    let nextActiveSegmentId: number | null = null

    for (const node of segmentsContainer.querySelectorAll<HTMLElement>('[data-radio-segment-row]')) {
        const start = Number(node.getAttribute('data-start') || '0')
        const end = Number(node.getAttribute('data-end') || '0')
        const segmentId = Number(node.getAttribute('data-segment-id') || '0')
        const active = currentTime >= start && currentTime < end

        if (active) {
            nextActiveSegmentId = segmentId
        }

        node.classList.toggle('bg-blue-50', active)
        node.classList.toggle('border-blue-200', active)
        node.classList.toggle('bg-slate-50/85', !active)
        node.classList.toggle('border-slate-200', !active)
    }

    setRegionActive(nextActiveSegmentId)
}

function renderWaveformSegments() {
    if (!regions) return

    clearWaveformSegments()

    for (const segment of activeSegments) {
        const region = regions.addRegion({
            id: `radio-segment-${segment.id}`,
            start: Number(segment.start),
            end: Number(segment.end),
            color: getSegmentRegionColor(segment, false),
            drag: false,
            resize: false,
        })

        if (region.element) {
            region.element.style.pointerEvents = 'none'
        }

        waveformRegions.set(Number(segment.id), region)
    }

    updateActiveSegmentHighlight()
}

function renderSegments(trackId: number, segments: RadioSegment[]) {
    activeSegments = Array.isArray(segments) ? segments : []

    if (!segmentsStatus || !segmentsContainer) return

    if (activeSegments.length === 0) {
        segmentsStatus.textContent = 'No analyzed segments for this track yet.'
        segmentsContainer.innerHTML = ''
        renderWaveformSegments()
        return
    }

    segmentsStatus.textContent = `${activeSegments.length} segment(s) available`
    segmentsContainer.innerHTML = activeSegments.map((segment) => {
        const typeLabel = segment.display_label || formatSegmentTypeLabel(segment.type)
        const currentRating = segment.rating === null ? null : Number(segment.rating)

        return `
            <article
                class="rounded-[1.15rem] border border-slate-200 bg-slate-50/85 px-4 py-3 transition"
                data-radio-segment-row="1"
                data-segment-id="${segment.id}"
                data-start="${segment.start}"
                data-end="${segment.end}"
            >
                <div class="flex flex-col gap-3">
                    <div class="min-w-0">
                        <div class="text-sm font-semibold text-slate-950">${formatTime(Number(segment.start))} - ${formatTime(Number(segment.end))}</div>
                        <div class="mt-1 text-xs uppercase tracking-[0.16em] text-slate-500">${typeLabel}</div>
                    </div>
                    <div class="flex items-center justify-between gap-3">
                        <span class="badge-soft !normal-case !tracking-normal">Importance ${segment.importance ?? 1}</span>
                        <div class="flex items-center gap-2">
                            ${authUser ? `
                                <button
                                    type="button"
                                    class="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500 transition hover:border-sky-200 hover:text-sky-700"
                                    data-radio-share-segment="${segment.id}"
                                    data-track-id="${trackId}"
                                >
                                    Share
                                </button>
                                <button
                                    type="button"
                                    class="rounded-full border border-slate-200 bg-white px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500 transition hover:border-sky-200 hover:text-sky-700"
                                    data-radio-direct-share-segment="${segment.id}"
                                    data-track-id="${trackId}"
                                >
                                    Friend
                                </button>
                            ` : ''}
                            <div class="flex items-center gap-1">
                                ${buildStarButtons('segment', currentRating, trackId, segment.id)}
                            </div>
                        </div>
                    </div>
                </div>
            </article>
        `
    }).join('')

    renderWaveformSegments()
}

function setTrackShareVisible(visible: boolean) {
    if (!shareTrackButton) {
        return
    }

    shareTrackButton.classList.toggle('hidden', !visible)
}

function persistFriendsShareDraft(payload: Record<string, unknown>) {
    window.sessionStorage.setItem(FRIENDS_SHARE_DRAFT_STORAGE_KEY, JSON.stringify(payload))
}

function openFriendsWithDraft(payload: Record<string, unknown>) {
    if (!authToken) {
        setError('Sign in to send tracks or segments to friends.')
        return
    }

    persistFriendsShareDraft(payload)
    window.location.assign('/friends')
}

function sliceWaveformPeaksForSegment(peaks: number[], trackDuration: number, start: number, end: number) {
    if (!Array.isArray(peaks) || peaks.length === 0 || trackDuration <= 0) {
        return []
    }

    const startRatio = Math.max(0, Math.min(1, start / trackDuration))
    const endRatio = Math.max(startRatio, Math.min(1, end / trackDuration))
    const startIndex = Math.max(0, Math.min(peaks.length - 1, Math.floor(startRatio * peaks.length)))
    const endIndex = Math.max(startIndex + 1, Math.min(peaks.length, Math.ceil(endRatio * peaks.length)))
    const slice = peaks.slice(startIndex, endIndex)

    if (slice.length <= 96) {
        return slice.map((peak) => Number(peak || 0))
    }

    const bucketSize = slice.length / 96
    const downsampled = []
    for (let bucket = 0; bucket < 96; bucket += 1) {
        const bucketStart = Math.floor(bucket * bucketSize)
        const bucketEnd = Math.floor((bucket + 1) * bucketSize)
        const chunk = slice.slice(bucketStart, Math.max(bucketStart + 1, bucketEnd))
        downsampled.push(Math.max(...chunk.map((peak) => Number(peak || 0))))
    }

    return downsampled
}

function shareTrackWithFriends() {
    const track = snapshot?.now_playing
    if (!track) {
        setError('Current track is not available to share.')
        return
    }

    openFriendsWithDraft({
        type: 'track',
        share: {
            type: 'track',
            track_id: track.track_id,
        },
        preview: {
            title: track.title,
            subtitle: `${track.artist || 'Unknown artist'} · ${formatTime(track.duration)}`,
        },
    })
}

function shareSegmentWithFriends(segmentId: number) {
    const segment = activeSegments.find((candidate) => Number(candidate.id) === segmentId)
    const track = snapshot?.now_playing
    if (!segment || !track) {
        setError('Current segment is not available to share.')
        return
    }

    const waveformPeaks = currentWaveformPayload?.peaks || snapshot?.now_playing_waveform?.peaks || []
    const waveformDuration = Number(currentWaveformPayload?.duration || snapshot?.now_playing_waveform?.duration || track.duration || 0)

    openFriendsWithDraft({
        type: 'segment',
        share: {
            type: 'segment',
            segment_id: Number(segment.id),
        },
        preview: {
            title: track.title,
            subtitle: track.artist || 'Unknown artist',
            label: segment.display_label || formatSegmentTypeLabel(segment.type),
            start: Number(segment.start),
            end: Number(segment.end),
            waveform: waveformPeaks.length > 0 ? {
                duration: Math.max(0.1, Number(segment.end) - Number(segment.start)),
                peaks: sliceWaveformPeaksForSegment(
                    waveformPeaks,
                    waveformDuration > 0 ? waveformDuration : Number(track.duration || 0),
                    Number(segment.start),
                    Number(segment.end),
                ),
            } : undefined,
        },
    })
}

function requestChatHistory() {
    if (!listenerSocket || !listenerSocketOpen || !authUser) {
        return
    }

    listenerSocket.send(JSON.stringify({
        type: 'request_chat_history',
        stationSlug: slug,
        clientId: listenerClientId,
        accessToken: radioAccessToken,
    }))
}

function sendChatMessage(text: string) {
    if (!listenerSocket || !listenerSocketOpen || !authUser) {
        setError('Sign in to send chat messages.')
        return
    }

    const normalized = text.trim()
    if (!normalized) {
        return
    }

    listenerSocket.send(JSON.stringify({
        type: 'chat_message',
        stationSlug: slug,
        clientId: listenerClientId,
        accessToken: radioAccessToken,
        text: normalized,
    }))
    updateChatStatus('Sending...')
}

function shareSegmentInChat(segmentId: number) {
    if (!listenerSocket || !listenerSocketOpen || !authUser) {
        setError('Sign in to share segments in chat.')
        return
    }

    const segment = activeSegments.find((candidate) => Number(candidate.id) === segmentId)
    const track = snapshot?.now_playing
    if (!segment || !track) {
        setError('Current segment is not available to share.')
        return
    }

    listenerSocket.send(JSON.stringify({
        type: 'chat_share_segment',
        stationSlug: slug,
        clientId: listenerClientId,
        accessToken: radioAccessToken,
        segment: {
            trackId: track.track_id,
            trackTitle: track.title,
            artist: track.artist,
            segmentId: Number(segment.id),
            label: segment.display_label || formatSegmentTypeLabel(segment.type),
            start: Number(segment.start),
            end: Number(segment.end),
        },
    }))
    updateChatStatus('Sharing segment...')
}

function renderWaveGenres(station: StationDescriptor | undefined) {
    if (!waveGenresPanel) {
        return
    }

    const canManage = Boolean(station?.is_personal && station?.can_manage)
    if (!canManage) {
        setWaveGenresVisible(false)
        waveGenresPanel.innerHTML = ''
        updateWaveGenresStatus('Choose genres for your recommendation-driven wave.')
        return
    }

    const availableGenres = Array.isArray(station?.available_genres) ? station.available_genres : []
    const selectedGenres = new Set(normalizeGenreSelection(station?.genres || [], availableGenres))
    const stylePresets = getWaveStylePresets(station)

    setWaveGenresVisible(true)

    if (availableGenres.length === 0) {
        waveGenresPanel.innerHTML = '<p class="text-sm text-slate-500">No genre options are available for your wave yet.</p>'
        updateWaveGenresStatus('No genre options available.')
        return
    }

    updateWaveGenresStatus(
        selectedGenres.size > 0
            ? `${selectedGenres.size} genre(s) active for your wave.`
            : 'No genre filter. Your wave uses all recommendation genres.',
    )

    const hasHiddenGenres = availableGenres.length > VISIBLE_WAVE_GENRE_COUNT
    waveGenresPanel.innerHTML = `
        ${stylePresets.length > 0 ? `
            <section class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4 shadow-sm shadow-slate-200/40">
                <div class="flex flex-col gap-2 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                        <div class="text-[11px] font-semibold uppercase tracking-[0.16em] text-sky-600">Quick styles</div>
                        <p class="mt-1 text-sm text-slate-500">Use mixed presets first. The detailed genres below will mark themselves automatically.</p>
                    </div>
                    <div class="text-xs text-slate-400">Built from your radio style mixes.</div>
                </div>
                <div class="mt-4 flex flex-wrap gap-2">
                    ${stylePresets.map((preset) => {
        const matchedGenres = preset.genres.filter((genre) => selectedGenres.has(genre)).length
        const isFullyActive = matchedGenres === preset.genres.length
        const isPartiallyActive = matchedGenres > 0 && matchedGenres < preset.genres.length
        const buttonClasses = isFullyActive
            ? 'border-sky-500 bg-sky-500 text-white shadow-sm shadow-sky-200/70'
            : isPartiallyActive
                ? 'border-sky-300 bg-sky-50 text-sky-700'
                : 'border-slate-200 bg-white text-slate-600 hover:border-sky-200 hover:text-sky-700'
        const badgeClasses = isFullyActive
            ? 'border border-white/35 bg-white/10 text-white'
            : isPartiallyActive
                ? 'border border-sky-200 bg-white text-sky-700'
                : 'border border-slate-200 bg-slate-50 text-slate-500'
        const badgeLabel = isFullyActive
            ? 'Active'
            : isPartiallyActive
                ? `${matchedGenres}/${preset.genres.length}`
                : 'Mix'

        return `
                        <button
                            type="button"
                            class="inline-flex items-center gap-2 rounded-xl border px-4 py-2 text-sm font-semibold transition ${buttonClasses}"
                            data-radio-wave-style="${escapeHtml(preset.slug)}"
                            title="${escapeHtml(preset.description)}"
                        >
                            <span>${escapeHtml(preset.label)}</span>
                            <span class="rounded-full px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] ${badgeClasses}">${escapeHtml(badgeLabel)}</span>
                        </button>
                    `
    }).join('')}
                </div>
            </section>
        ` : ''}
        <section class="${stylePresets.length > 0 ? 'mt-4' : ''}">
            <div class="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                <div class="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">Detailed genres</div>
                <div class="text-xs text-slate-400">${selectedGenres.size} detailed genre(s) selected</div>
            </div>
            <div class="mt-3 flex flex-wrap gap-2">
                ${availableGenres.map((genre, index) => {
        const normalizedGenre = String(genre).trim().toLowerCase()
        const checked = selectedGenres.has(normalizedGenre) ? 'checked' : ''
        const hidden = !waveGenresExpanded && index >= VISIBLE_WAVE_GENRE_COUNT && !selectedGenres.has(normalizedGenre)
            ? 'hidden'
            : ''

        return `
            <label class="${hidden} inline-flex cursor-pointer items-center gap-2 rounded-full border border-slate-200 bg-slate-50/85 px-3 py-2 text-xs font-semibold uppercase tracking-[0.14em] text-slate-700 transition hover:border-sky-200 hover:text-sky-700">
                <input
                    type="checkbox"
                    value="${escapeHtml(normalizedGenre)}"
                    data-radio-wave-genre="${escapeHtml(normalizedGenre)}"
                    class="sr-only peer"
                    ${checked}
                >
                <span class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 text-[10px] text-transparent peer-checked:border-sky-500 peer-checked:bg-sky-500 peer-checked:text-white">✓</span>
                <span>${escapeHtml(genre)}</span>
            </label>
        `
            }).join('')}
            </div>
        </section>
        ${hasHiddenGenres ? `
            <button
                type="button"
                class="mt-3 rounded-full border border-slate-200 bg-white px-3 py-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500 transition hover:border-sky-200 hover:text-sky-700"
                data-radio-wave-genres-toggle="${waveGenresExpanded ? 'collapse' : 'expand'}"
            >
                ${waveGenresExpanded ? 'Show fewer genres' : `Show ${availableGenres.length - VISIBLE_WAVE_GENRE_COUNT} more genres`}
            </button>
        ` : ''}
    `
}

function updatePersonalStationGenres() {
    if (!snapshot?.station?.is_personal || !snapshot.station.can_manage || !waveGenresPanel) {
        return
    }

    const preferredGenres = Array.from(
        waveGenresPanel.querySelectorAll<HTMLInputElement>('input[data-radio-wave-genre]:checked'),
    ).map((input) => input.value)

    sendPersonalStationGenres(preferredGenres)
}

function renderStations(stations: StationDescriptor[] | undefined) {
    if (!stationsContainer) return

    const renderedStations = mergeViewerStations(stations)
    if (renderedStations.length === 0) {
        stationsContainer.innerHTML = ''
        return
    }

    const hasHiddenStations = renderedStations.length > VISIBLE_STATION_COUNT
    stationsContainer.innerHTML = renderedStations.map((station, index) => {
        const href = toDisplayUrl(station.public_url)
        const classes = station.selected
            ? 'action-primary !rounded-xl !px-4 !py-2 !text-sm'
            : 'action-secondary !rounded-xl !px-4 !py-2 !text-sm'
        const hidden = !stationsExpanded && index >= VISIBLE_STATION_COUNT && !station.selected ? ' hidden' : ''
        const badge = station.is_personal
            ? '<span class="ml-2 rounded-full border border-white/30 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em]">My wave</span>'
            : Array.isArray(station.genres) && station.genres.length > 1
                ? '<span class="ml-2 rounded-full border border-slate-200 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">Mix</span>'
                : ''

        return `<a href="${href}" class="${classes}${hidden}" data-radio-station-link="${station.slug}">${station.name}${badge}</a>`
    }).join('') + (hasHiddenStations
        ? `<button
                type="button"
                class="rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-500 transition hover:border-sky-200 hover:text-sky-700"
                data-radio-stations-toggle="${stationsExpanded ? 'collapse' : 'expand'}"
            >${stationsExpanded ? 'Show fewer stations' : `Show ${renderedStations.length - VISIBLE_STATION_COUNT} more stations`}</button>`
        : '')
}

function mergeViewerStations(stations: StationDescriptor[] | undefined) {
    const nextStations = Array.isArray(stations) ? [...stations] : []
    const viewerId = Number(authUser?.id || 0)
    if (viewerId <= 0) {
        return nextStations
    }

    const personalSlug = `personal-${viewerId}`
    if (nextStations.some((station) => station.slug === personalSlug)) {
        return nextStations
    }

    nextStations.push({
        slug: personalSlug,
        name: 'Your wave',
        description: 'Your private recommendation-driven station.',
        public_url: `/radio/${personalSlug}`,
        selected: slug === personalSlug,
    })

    return nextStations
}

async function refreshRadioAccessTokenAndReconnect() {
    const nextToken = await ensureRadioAccessToken(true)
    if (!nextToken) {
        return
    }

    closeListenerSocket()
    void connectListenerSocket()
}

async function initializeAuth() {
    const queryToken = searchParams.get('token')
    if (queryToken) {
        window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, queryToken)
    }

    authToken = queryToken || window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
    authUser = readStoredUser()

    if (authUser) {
        renderStations(snapshot?.stations)
    }

    if (!authToken) {
        resetListeningHistoryState()
        setFeedbackVisible(false)
        setChatVisible(false)
        setTrackShareVisible(false)
        renderWaveGenres(undefined)
        setChatEntries([])
        updateChatStatus('Sign in to join the chat.')
        renderTrackFit(null)
        if (feedbackAuth) {
            feedbackAuth.textContent = 'Sign in to rate the current track and its segments.'
        }
        if (chatAuth) {
            chatAuth.textContent = 'Sign in to chat and share segments with other listeners.'
        }
        renderStations(snapshot?.stations)
        return
    }

    setFeedbackVisible(true)
    setChatVisible(Boolean(authUser))
    setTrackShareVisible(Boolean(authUser && snapshot?.now_playing))
    renderWaveGenres(snapshot?.station)
    if (feedbackAuth) {
        feedbackAuth.textContent = authUser?.name
            ? `Signed in as ${authUser.name}`
            : 'Connecting your account...'
    }
    if (chatAuth) {
        chatAuth.textContent = authUser?.name
            ? `Signed in as ${authUser.name}`
            : 'Connecting your account...'
    }
    updateChatStatus(authUser ? 'Connected' : 'Connecting...')

    if (listenerSocket) {
        closeListenerSocket()
    }

    void connectListenerSocket()
    renderStations(snapshot?.stations)
}

async function loadFeedbackForTrack(trackId: number) {
    feedbackTrackId = trackId

    if (!authToken) {
        setFeedbackVisible(false)
        renderSegments(trackId, [])
        renderTrackRating(trackId, null)
        renderTrackFit(null)
        return
    }

    if (!shouldMaintainPrivateOverlaySession()) {
        setFeedbackVisible(true)
        renderTrackFit(null)
        renderTrackRating(trackId, null)
        renderSegments(trackId, [])
        if (segmentsStatus) {
            segmentsStatus.textContent = 'Start radio to load live private feedback.'
        }
        return
    }

    const accessToken = await ensureRadioAccessToken()
    if (!accessToken) {
        if (feedbackAuth) {
            feedbackAuth.textContent = 'Sign in to load private radio feedback.'
        }
        setFeedbackVisible(false)
        return
    }
    setFeedbackVisible(true)
    if (segmentsStatus) {
        segmentsStatus.textContent = 'Loading segments...'
    }

    if (!listenerSocketOpen || !listenerSocket) {
        return
    }

    requestPrivateOverlay(trackId)
}

async function submitTrackRating(trackId: number, rating: number) {
    const response = await fetch(`/api/tracks/${trackId}/rating`, {
        method: 'POST',
        headers: apiHeaders(true),
        body: JSON.stringify({ rating }),
    })

    if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.message || 'Failed to rate track')
    }

    const payload = await response.json().catch(() => null) as {
        effective_rating?: number | null
        computed_track_rating?: number | null
        sentiment?: string | null
        feedback?: PrivateOverlayPayload | null
    } | null

    if (payload?.feedback) {
        renderPrivateOverlay(payload.feedback)
    } else {
        renderTrackRating(
            trackId,
            typeof payload?.effective_rating === 'number' ? payload.effective_rating : rating,
            payload?.sentiment ?? null,
            typeof payload?.computed_track_rating === 'number' ? payload.computed_track_rating : null,
        )
    }
    requestPrivateOverlay(trackId)
}

async function submitSegmentRating(trackId: number, segmentId: number, rating: number) {
    const response = await fetch(`/api/segments/${segmentId}/rate`, {
        method: 'POST',
        headers: apiHeaders(true),
        body: JSON.stringify({ rating }),
    })

    if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.message || 'Failed to rate segment')
    }

    const payload = await response.json().catch(() => null) as {
        track_rating?: number | null
        computed_track_rating?: number | null
        sentiment?: string | null
        feedback?: PrivateOverlayPayload | null
    } | null

    if (payload?.feedback) {
        renderPrivateOverlay(payload.feedback)
    } else {
        activeSegments = activeSegments.map((segment) => (
            Number(segment.id) === Number(segmentId)
                ? { ...segment, rating }
                : segment
        ))

        renderSegments(trackId, activeSegments)
        renderTrackRating(
            trackId,
            typeof payload?.track_rating === 'number'
                ? payload.track_rating
                : (typeof payload?.computed_track_rating === 'number'
                    ? Math.max(1, Math.min(5, Math.round(payload.computed_track_rating)))
                    : null),
            payload?.sentiment ?? null,
            typeof payload?.computed_track_rating === 'number' ? payload.computed_track_rating : null,
        )
    }
    requestPrivateOverlay(trackId)
}

function renderList(container: HTMLDivElement | null, items: StationTrack[] | undefined, emptyText: string, prefix: string) {
    if (!container) return
    if (!items || items.length === 0) {
        container.innerHTML = `<p class="text-sm text-slate-500">${emptyText}</p>`
        return
    }

    container.innerHTML = items.map((track) => {
        const time = new Date(track.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        return `
            <article class="rounded-[1.15rem] border border-slate-200 bg-slate-50/85 px-4 py-3">
                <div class="font-semibold text-slate-950">${track.title}</div>
                <div class="mt-1 text-sm text-slate-500">${track.artist || 'Unknown artist'}</div>
                <div class="mt-2 text-xs font-medium uppercase tracking-[0.14em] text-slate-400">${prefix} ${time}</div>
            </article>
        `
    }).join('')
}

function syncPlaybackChromeFromAudio() {
    if (!audio || !snapshot?.now_playing) {
        if (timeLabel) timeLabel.textContent = '0:00 / 0:00'
        if (remaining) remaining.textContent = 'Awaiting next track'
        updatePlayToggleLabel()
        return
    }

    const currentTime = audio.currentTime || deriveLiveOffsetSeconds(snapshot.now_playing)
    const totalDuration = audio.duration || snapshot.now_playing.duration || 0
    const remainingSeconds = Math.max(0, totalDuration - currentTime)

    if (timeLabel) timeLabel.textContent = `${formatTime(currentTime)} / ${formatTime(totalDuration)}`
    if (remaining) remaining.textContent = `${formatTime(remainingSeconds)} remaining`

    syncWaveformPosition()
    updateActiveSegmentHighlight()
    updatePlayToggleLabel()
}

function buildDirectStreamUrl(baseUrl: string) {
    const url = new URL(baseUrl, window.location.origin)
    if (selectedBitrate !== 'auto') {
        url.searchParams.set('bitrate', selectedBitrate)
    } else {
        url.searchParams.delete('bitrate')
    }

    return url.toString()
}

function buildHlsMasterUrl(current: StationTrack) {
    if (!current.track_id) {
        return null
    }

    return new URL(`/api/tracks/${current.track_id}/hls/master.m3u8`, window.location.origin).toString()
}

function buildHlsSourceUrl(current: StationTrack) {
    const masterUrl = buildHlsMasterUrl(current)
    const baseUrl = masterUrl ? new URL(masterUrl) : null

    if (!baseUrl) {
        return null
    }

    if (selectedBitrate === 'auto') {
        return baseUrl.toString()
    }

    const variantUrl = new URL(baseUrl.toString())
    variantUrl.pathname = variantUrl.pathname.replace(/\/master\.m3u8$/i, `/${selectedBitrate}k/index.m3u8`)
    return variantUrl.toString()
}

function buildWaveformUrl(current: StationTrack) {
    const path = current.waveform_url || `/api/tracks/${current.track_id}/waveform`
    return new URL(path, window.location.origin).toString()
}

function deriveLiveOffsetSeconds(current: StationTrack) {
    const startedAt = Date.parse(current.started_at || '')
    if (Number.isFinite(startedAt)) {
        return Math.max(0, Math.min(current.duration, (Date.now() - startedAt) / 1000))
    }

    return Math.max(0, Math.min(current.duration, Number(current.offset_seconds || 0)))
}

function applySelectedHlsLevel() {
    if (!hls) {
        return
    }

    if (selectedBitrate === 'auto') {
        hls.currentLevel = -1
        hls.loadLevel = -1
        hls.nextLevel = -1
        return
    }

    const selectedLabel = `${selectedBitrate}k`
    const levelIndex = hls.levels.findIndex((level) => (
        Array.isArray(level.url)
            ? level.url.some((item) => item.includes(`/${selectedLabel}/index.m3u8`))
            : String(level.url || '').includes(`/${selectedLabel}/index.m3u8`)
    ))

    if (levelIndex >= 0) {
        hls.loadLevel = levelIndex
        hls.nextLevel = levelIndex
    }
}

async function loadWaveformTrack(current: StationTrack, force = false) {
    if (!wavesurfer || !listenerSocketOpen || !listenerSocket) {
        return
    }

    if (waveformPayloadTrackId === current.track_id) {
        return
    }

    if (!force && waveformTrackId === current.track_id) {
        return
    }

    const requestId = ++waveformLoadRequestId
    waveformTrackId = current.track_id
    updateRuntimeDiagnostics()
    setWaveformStatus('Loading waveform...')
    clearWaveformSegments()
    listenerSocket.send(JSON.stringify({
        type: 'request_waveform',
        stationSlug: slug,
        clientId: listenerClientId,
        trackId: current.track_id,
        requestId,
    }))
}

async function prewarmUpcomingTrack(track: StationTrack | null | undefined) {
    if (!track || !listenerSocketOpen || !listenerSocket) {
        return
    }

    if (prewarmedUpcomingTrackIds.has(track.track_id)) {
        return
    }

    const pending = prewarmingUpcomingTrackIds.get(track.track_id)
    if (pending) {
        await pending
        return
    }

    const request = (async () => {
        listenerSocket.send(JSON.stringify({
            type: 'prewarm_track',
            stationSlug: slug,
            clientId: listenerClientId,
            trackId: track.track_id,
        }))
        prewarmedUpcomingTrackIds.add(track.track_id)
    })()

    prewarmingUpcomingTrackIds.set(track.track_id, request)

    try {
        await request
    } finally {
        prewarmingUpcomingTrackIds.delete(track.track_id)
    }
}

function loadTrackSource(current: StationTrack, shouldResume: boolean, seekTargetSeconds = current.offset_seconds) {
    if (!audio) {
        return
    }

    if (currentListeningHistoryId !== null && currentListeningHistoryTrackId !== current.track_id) {
        void completeListeningHistory(true)
    }

    const hlsMasterUrl = buildHlsMasterUrl(current)
    const hlsSourceUrl = buildHlsSourceUrl(current)
    const directStreamUrl = buildDirectStreamUrl(current.stream_url)
    const canUseNativeHls = canUseNativeHlsPlayback()
    const shouldUseHlsJs = selectedTransportMode === 'hls'
        && Boolean(hlsMasterUrl)
        && Hls.isSupported()
    const shouldUseNativeHls = Boolean(hlsSourceUrl)
        && canUseNativeHls
        && !Hls.isSupported()
        && (selectedTransportMode === 'hls' || nativeHlsFallbackTrackId === current.track_id)
    const willTriggerInternalPause = !audio.paused || Boolean(audio.currentSrc)

    pendingSeek = Math.max(0, Math.min(current.duration, seekTargetSeconds))
    shouldResumeAfterSourceLoad = shouldResume
    suppressPausePresence = willTriggerInternalPause
    audio.dataset.trackId = String(current.track_id)
    audio.dataset.hlsMasterUrl = hlsMasterUrl || ''
    audio.dataset.hlsSourceUrl = hlsSourceUrl || ''
    audio.dataset.streamUrl = directStreamUrl
    audio.playbackRate = 1
    applyPitchPreservation()

    if (shouldUseHlsJs) {
        currentPlaybackMode = 'hls-js'
        updateRuntimeDiagnostics()
        destroyHls()
        const hlsEntryUrl = hlsSourceUrl || hlsMasterUrl
        hls = new Hls({
            enableWorker: true,
            lowLatencyMode: false,
            backBufferLength: 18,
            maxBufferLength: 12,
            maxMaxBufferLength: 18,
            startFragPrefetch: false,
        })
        hls.attachMedia(audio)
        hls.on(Hls.Events.MEDIA_ATTACHED, () => {
            hls?.loadSource(hlsEntryUrl)
        })
        hls.on(Hls.Events.MANIFEST_PARSED, () => {
            if (hlsEntryUrl === hlsMasterUrl) {
                applySelectedHlsLevel()
            }
        })
        hls.on(Hls.Events.ERROR, (_event, data) => {
            if (data.fatal) {
                publishRadioDiagnostics(`hls-error:${data.type}`, snapshot?.now_playing || null)
                if (data.type === Hls.ErrorTypes.NETWORK_ERROR && hls) {
                    hls.startLoad(Number.isFinite(audio.currentTime) ? audio.currentTime : -1)
                    return
                }

                if (data.type === Hls.ErrorTypes.MEDIA_ERROR && hls) {
                    hls.recoverMediaError()
                    return
                }

                destroyHls()
                currentPlaybackMode = 'direct'
                updateRuntimeDiagnostics()
                audio.src = directStreamUrl
                audio.load()
            }
        })
        return
    }

    destroyHls()

    if (shouldUseNativeHls) {
        currentPlaybackMode = 'hls-native'
        updateRuntimeDiagnostics()
        audio.src = hlsSourceUrl
        audio.load()
        return
    }

    currentPlaybackMode = 'direct'
    updateRuntimeDiagnostics()
    audio.src = directStreamUrl
    audio.load()
}

function syncToLiveOffset(current: StationTrack) {
    if (!audio) {
        return
    }

    const liveOffsetSeconds = deriveLiveOffsetSeconds(current)
    const currentTime = Number.isFinite(audio.currentTime) ? audio.currentTime : 0
    const delta = liveOffsetSeconds - currentTime
    const drift = Math.abs(delta)
    const now = Date.now()

    if (drift >= HARD_SYNC_DRIFT_SECONDS) {
        if (now - lastLiveSyncAt >= LIVE_SYNC_MIN_INTERVAL_MS) {
            audio.currentTime = liveOffsetSeconds
            lastLiveSyncAt = now
            publishRadioDiagnostics('hard-sync-seek', current)
        }
        audio.playbackRate = 1
        return
    }

    if (drift >= SOFT_SYNC_DRIFT_SECONDS) {
        if (delta > 0) {
            audio.playbackRate = drift >= SOFT_SYNC_DRIFT_SECONDS * 2 ? 1.08 : 1.04
        } else {
            audio.playbackRate = 0.96
        }
        lastLiveSyncAt = now
        publishRadioDiagnostics('soft-sync-rate', current)
        return
    }

    if (drift >= FINE_SYNC_DRIFT_SECONDS) {
        audio.playbackRate = delta > 0 ? 1.02 : 0.98
        publishRadioDiagnostics('fine-sync-rate', current)
        return
    }

    if (audio.playbackRate !== 1) {
        audio.playbackRate = 1
    }
}

function syncAudio(current: StationTrack) {
    if (!audio) return

    const currentTrackId = Number(audio.dataset.trackId || '0')
    const trackChanged = currentTrackId !== current.track_id

    if (trackChanged && nativeHlsFallbackTrackId !== current.track_id) {
        nativeHlsFallbackTrackId = null
    }

    if (!shouldAutoResumePlayback()) {
        if (trackChanged || waveformTrackId !== current.track_id) {
            void loadWaveformTrack(current, true)
        }
        unloadTrackSource()
        return
    }

    const hasInlineWaveformPayload = waveformPayloadTrackId === current.track_id
    void loadWaveformTrack(current, !hasInlineWaveformPayload && (trackChanged || waveformTrackId !== current.track_id))
    const nextShouldResume = shouldAutoResumePlayback()
    const hlsMasterUrl = buildHlsMasterUrl(current) || ''
    const hlsSourceUrl = buildHlsSourceUrl(current) || ''
    const directStreamUrl = buildDirectStreamUrl(current.stream_url)
    const currentStreamUrl = audio.dataset.streamUrl || ''
    const currentHlsMasterUrl = audio.dataset.hlsMasterUrl || ''
    const currentHlsSourceUrl = audio.dataset.hlsSourceUrl || ''
    const hlsMasterChanged = currentHlsMasterUrl !== hlsMasterUrl
    const hlsSourceChanged = currentHlsSourceUrl !== hlsSourceUrl
    const directSourceChanged = currentStreamUrl !== directStreamUrl
    const liveOffsetSeconds = deriveLiveOffsetSeconds(current)
    const preservedSeekTarget = Number.isFinite(audio.currentTime) && audio.currentTime > 0
        ? Math.min(current.duration, Math.max(audio.currentTime, liveOffsetSeconds))
        : liveOffsetSeconds

    if (!trackChanged && currentPlaybackMode === 'hls-js' && hls && hlsSourceChanged) {
        loadTrackSource(current, nextShouldResume, preservedSeekTarget)
        return
    }

    if (!trackChanged && currentPlaybackMode === 'hls-js' && hls) {
        audio.dataset.hlsMasterUrl = hlsMasterUrl
        audio.dataset.hlsSourceUrl = hlsSourceUrl
        audio.dataset.streamUrl = directStreamUrl
        applySelectedHlsLevel()
        syncToLiveOffset(current)
        return
    }

    if (!trackChanged && currentPlaybackMode === 'hls-native' && hlsSourceChanged) {
        loadTrackSource(current, nextShouldResume, preservedSeekTarget)
        return
    }

    if (!trackChanged && currentPlaybackMode === 'direct' && directSourceChanged) {
        loadTrackSource(current, nextShouldResume, preservedSeekTarget)
        return
    }

    if (trackChanged || hlsMasterChanged || hlsSourceChanged || directSourceChanged) {
        loadTrackSource(current, nextShouldResume, trackChanged ? liveOffsetSeconds : preservedSeekTarget)
        return
    }

    syncToLiveOffset(current)
}

function renderSnapshot(data: StationSnapshot) {
    snapshot = data
    const current = data.now_playing || null
    const hasQueue = hasPlayableQueue(data)
    body.dataset.radioSnapshotState = current ? 'now-playing' : (hasQueue ? 'queued' : 'empty')
    body.dataset.radioSnapshotGeneratedAt = data.generated_at || ''
    console.info('[public-radio]', RADIO_BUILD_STAMP, 'renderSnapshot ENTER', {
        station: data.station?.slug || slug,
        hasQueue,
        hasNowPlaying: Boolean(current),
        generatedAt: data.generated_at || null,
    })

    if (data.station?.is_personal) {
        const confirmedGenres = normalizeGenreSelection(
            Array.isArray(data.station.genres) ? data.station.genres : [],
            Array.isArray(data.station.available_genres) ? data.station.available_genres : [],
        )

        if (sameGenreSelection(pendingPersonalWaveGenres, confirmedGenres)) {
            pendingPersonalWaveGenres = null
            personalWaveGenresRequestInFlight = false
        } else if (personalWaveGenresRequestInFlight) {
            personalWaveGenresRequestInFlight = false
            if (pendingPersonalWaveGenres && !personalWaveGenresFlushScheduled) {
                schedulePersonalStationGenresFlush()
            }
        }
    } else {
        pendingPersonalWaveGenres = null
        personalWaveGenresRequestInFlight = false
    }

    clearError()
    loading?.classList.add('hidden')

    if (stationName) stationName.textContent = data.station?.name || 'Radio station'
    if (stationDescription) stationDescription.textContent = data.station?.description || ''
    if (publicUrl) publicUrl.textContent = toDisplayUrl(data.station?.public_url)
    if (embedUrl) embedUrl.textContent = toDisplayUrl(data.station?.embed_url)
    updateListenerCount(data.listener_count ?? 0)
    renderStations(data.stations)
    renderWaveGenres(data.station)

    renderList(upNext, data.up_next || [], 'No tracks queued yet.', 'starts')
    renderList(history, data.history || [], 'History will appear after the station advances.', 'played at')

    if (!current) {
        void completeListeningHistory(false)
        waveformTrackId = null
        updateRuntimeDiagnostics()
        unloadTrackSource()
        setTrackShareVisible(false)
        if (nowTitle) nowTitle.textContent = hasQueue ? 'Waiting for station queue' : 'No playable tracks available'
        if (nowArtist) nowArtist.textContent = hasQueue ? 'Unknown artist' : 'Radio library is empty or tracks are not ready'
        if (remaining) remaining.textContent = hasQueue ? 'Awaiting next track' : 'Add completed tracks with audio files'
        if (timeLabel) timeLabel.textContent = '0:00 / 0:00'
        setWaveformStatus(
            hasQueue
                ? 'Waveform will appear when the station has an active track.'
                : 'Waveform is unavailable until the radio has playable tracks.',
        )
        clearWaveformSegments()
        void wavesurfer?.empty()
        updatePlayToggleLabel()
        console.info('[public-radio]', RADIO_BUILD_STAMP, 'renderSnapshot EXIT empty', {
            station: data.station?.slug || slug,
            hasQueue,
        })
        return
    }

    if (nowTitle) nowTitle.textContent = current.title
    if (nowArtist) nowArtist.textContent = current.artist || 'Unknown artist'
    setTrackShareVisible(Boolean(authUser))
    const liveOffsetSeconds = deriveLiveOffsetSeconds(current)
    if (remaining) remaining.textContent = `${formatTime(Math.max(0, current.duration - liveOffsetSeconds))} remaining`
    if (timeLabel) timeLabel.textContent = `${formatTime(liveOffsetSeconds)} / ${formatTime(current.duration)}`

    if (data.now_playing_waveform && waveformTrackId !== current.track_id) {
        waveformTrackId = current.track_id
        waveformPayloadTrackId = current.track_id
        updateRuntimeDiagnostics()
        void applyWaveformPayload(current.track_id, data.now_playing_waveform, current.duration)
    }

    syncAudio(current)
    updatePlayToggleLabel()

    if (!shouldMaintainPrivateOverlaySession() && authToken && feedbackTrackId !== current.track_id) {
        feedbackTrackId = current.track_id
        renderTrackFit(null)
        renderTrackRating(current.track_id, null)
        renderSegments(current.track_id, [])
        if (segmentsStatus) {
            segmentsStatus.textContent = 'Start radio to load live private feedback.'
        }
    }

    if (shouldMaintainPrivateOverlaySession() && feedbackTrackId !== current.track_id) {
        void loadFeedbackForTrack(current.track_id)
    }

    if (shouldAutoResumePlayback()) {
        void prewarmUpcomingTrack(data.up_next?.[0])
    }

    console.info('[public-radio]', RADIO_BUILD_STAMP, 'renderSnapshot EXIT now-playing', {
        station: data.station?.slug || slug,
        trackId: current.track_id,
    })
}

function requestStationSnapshot() {
    if (!listenerSocket || !listenerSocketOpen) {
        return
    }

    listenerSocket.send(JSON.stringify({
        type: 'request_snapshot',
        stationSlug: slug,
        clientId: listenerClientId,
        listening: isActivelyListening(),
        accessToken: radioAccessToken,
    }))
}

function scheduleListenerReconnect() {
    if (listenerReconnectTimer !== null) return

    listenerReconnectTimer = window.setTimeout(() => {
        listenerReconnectTimer = null
        connectListenerSocket()
    }, 2000)
}

function getListenerSocketUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = new URL(`${protocol}//${window.location.host}/ws/radio-listeners`)
    url.searchParams.set('station', slug)
    url.searchParams.set('clientId', listenerClientId)
    if (radioAccessToken) {
        url.searchParams.set('accessToken', radioAccessToken)
    }
    return url.toString()
}

function getPrivateSocketUrl() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = new URL(`${protocol}//${window.location.host}/ws/radio-private`)
    url.searchParams.set('station', slug)
    url.searchParams.set('clientId', listenerClientId)
    if (radioAccessToken) {
        url.searchParams.set('accessToken', radioAccessToken)
    }
    return url.toString()
}

function renderPrivateOverlay(overlay: PrivateOverlayPayload) {
    const trackId = Number(overlay.track_id || 0)
    if (!trackId || feedbackTrackId !== trackId) {
        return
    }

    renderTrackRating(
        trackId,
        overlay.track_rating ?? null,
        overlay.sentiment ?? null,
        typeof overlay.computed_track_rating === 'number' ? overlay.computed_track_rating : null,
    )
    renderTrackFit(typeof overlay.preference_score === 'number' ? Number(overlay.preference_score) : null)
    renderSegments(trackId, overlay.segments || [])
}

function schedulePrivateReconnect() {
    if (!shouldMaintainPrivateOverlaySession()) return
    if (privateReconnectTimer !== null) return

    privateReconnectTimer = window.setTimeout(() => {
        privateReconnectTimer = null
        connectPrivateSocket()
    }, 2000)
}

function requestPrivateOverlay(trackId: number) {
    if (!shouldMaintainPrivateOverlaySession() || !listenerSocket || !listenerSocketOpen || !radioAccessToken) {
        return
    }

    listenerSocket.send(JSON.stringify({
        type: 'request_overlay',
        stationSlug: slug,
        clientId: listenerClientId,
        accessToken: radioAccessToken,
        trackId,
    }))
}

function sendListenerPresence() {
    if (!listenerSocket || !listenerSocketOpen) return

    listenerSocket.send(JSON.stringify({
        type: 'presence',
        stationSlug: slug,
        clientId: listenerClientId,
        listening: isActivelyListening(),
    }))
}

function startListenerPresenceHeartbeat() {
    if (listenerPresenceTimer !== null) {
        window.clearInterval(listenerPresenceTimer)
    }

    listenerPresenceTimer = window.setInterval(() => {
        sendListenerPresence()
    }, LISTENER_PRESENCE_INTERVAL_MS)
}

async function connectListenerSocket() {
    if (listenerSocket || listenerSocketConnectPromise || typeof window.WebSocket === 'undefined') return

    listenerSocketConnectPromise = (async () => {
        if (authToken || isPersonalStation) {
            await ensureRadioAccessToken()
            if (isPersonalStation && !radioAccessToken) {
                return
            }
        }

        const socket = new window.WebSocket(getListenerSocketUrl())
        listenerSocket = socket

        socket.addEventListener('open', () => {
            listenerSocketOpen = true
            startListenerPresenceHeartbeat()
            socket.send(JSON.stringify({
                type: 'subscribe',
                stationSlug: slug,
                clientId: listenerClientId,
                listening: isActivelyListening(),
                accessToken: radioAccessToken,
            }))
            flushPersonalStationGenres()
        })

        socket.addEventListener('message', (event) => {
            try {
                const payload = JSON.parse(String(event.data)) as {
                    type?: string
                    stationSlug?: string
                    count?: number
                    snapshot?: StationSnapshot
                    message?: string
                    overlay?: PrivateOverlayPayload
                    waveform?: TrackWaveformPayload
                    trackId?: number
                    entries?: RadioChatEntry[]
                    entry?: RadioChatEntry
                }
                if (payload.type === 'listener_count' && payload.stationSlug === slug) {
                    updateListenerCount(Number(payload.count || 0))
                    return
                }

                if (payload.type === 'viewer_state' && payload.stationSlug === slug) {
                    applyViewerState((payload as { viewer?: { id?: number; name?: string; email?: string } | null }).viewer || null)
                    return
                }

                if (payload.type === 'station_snapshot' && payload.stationSlug === slug && payload.snapshot) {
                    renderSnapshot(payload.snapshot)
                    return
                }

                if (payload.type === 'private_overlay' && payload.stationSlug === slug && payload.overlay) {
                    renderPrivateOverlay(payload.overlay)
                    return
                }

                if (payload.type === 'private_overlay_unauthorized' && payload.stationSlug === slug) {
                    setFeedbackVisible(false)
                    if (feedbackAuth) {
                        feedbackAuth.textContent = 'Sign in to load private radio feedback.'
                    }
                    return
                }

                if (payload.type === 'waveform_payload' && payload.stationSlug === slug && payload.waveform) {
                    const trackId = Number(payload.trackId || 0)
                    if (!trackId || waveformTrackId !== trackId) {
                        return
                    }

                    void applyWaveformPayload(trackId, payload.waveform, snapshot?.now_playing?.duration || 0)
                    return
                }

                 if (payload.type === 'chat_history' && payload.stationSlug === slug) {
                    setChatEntries(normalizeChatEntries(payload.entries))
                    updateChatStatus('Connected')
                    return
                }

                if (payload.type === 'chat_message' && payload.stationSlug === slug && payload.entry) {
                    appendChatEntry(payload.entry)
                    updateChatStatus('Connected')
                    return
                }

                if (payload.type === 'chat_unauthorized' && payload.stationSlug === slug) {
                    setChatVisible(false)
                    updateChatStatus('Sign in to join the chat.')
                    if (chatAuth) {
                        chatAuth.textContent = 'Sign in to chat and share segments with other listeners.'
                    }
                    return
                }

                if (payload.type === 'error' && payload.stationSlug === slug && payload.message) {
                    if (payload.message.toLowerCase().includes('unauthorized') || payload.message.toLowerCase().includes('access token')) {
                        void refreshRadioAccessTokenAndReconnect()
                    }
                    setError(payload.message)
                }
            } catch {
                // Ignore malformed websocket payloads from the listener channel.
            }
        })

        socket.addEventListener('close', () => {
            if (listenerSocket === socket) {
                listenerSocketOpen = false
                listenerSocket = null
            }
            personalWaveGenresRequestInFlight = false
            if (listenerPresenceTimer !== null) {
                window.clearInterval(listenerPresenceTimer)
                listenerPresenceTimer = null
            }
            if (!shuttingDown) {
                scheduleListenerReconnect()
            }
        })

        socket.addEventListener('error', () => {
            if (listenerSocket === socket) {
                listenerSocket.close()
            }
        })
    })()

    try {
        await listenerSocketConnectPromise
    } finally {
        listenerSocketConnectPromise = null
    }
}

async function attemptPlayback() {
    if (!audio) return

    try {
        await audio.play()
        clearError()
        autoplayBootstrapAttempted = true
    } catch {
        if (!autoplayBootstrapAttempted) {
            const restoreMuted = audio.muted
            const restoreVolume = audio.volume

            autoplayBootstrapAttempted = true
            audio.muted = true

            try {
                await audio.play()
                clearError()
                audio.muted = restoreMuted
                if (!restoreMuted && audio.volume === 0) {
                    audio.volume = restoreVolume > 0 ? restoreVolume : DEFAULT_RADIO_VOLUME
                }
                updateVolumeUi()
                return
            } catch {
                audio.muted = restoreMuted
                audio.volume = restoreVolume
            }
        }

        setError('Browser playback is blocked until you press Start radio.')
    } finally {
        updatePlayToggleLabel()
    }
}

async function togglePlayback() {
    if (!audio) return

    if (!audio.paused) {
        playbackIntent = false
        audio.pause()
        updatePlayToggleLabel()
        return
    }

    if (!snapshot?.now_playing) {
        playbackIntent = true
        if (snapshot && !hasPlayableQueue(snapshot)) {
            setError('No playable tracks available for this station.')
            updatePlayToggleLabel()
            return
        }
        requestStationSnapshot()
        setError('Waiting for live station snapshot...')
        return
    }

    playbackIntent = true
    if (!audio.currentSrc || Number(audio.dataset.trackId || '0') !== snapshot.now_playing.track_id) {
        loadTrackSource(snapshot.now_playing, true)
        if (authToken) {
            void loadFeedbackForTrack(snapshot.now_playing.track_id)
        }
        return
    }

    await attemptPlayback()
}

function initWaveform() {
    if (!waveformRoot) return

    regions = RegionsPlugin.create()

    wavesurfer = WaveSurfer.create({
        container: waveformRoot,
        waveColor: '#cbd5e1',
        progressColor: '#38bdf8',
        cursorColor: '#0f172a',
        height: 152,
        barWidth: 2,
        barGap: 2,
        barRadius: 3,
        normalize: true,
        interact: false,
        plugins: [regions],
    })

    wavesurfer.on('ready', () => {
        setWaveformStatus('', true)
        renderWaveformSegments()
        syncPlaybackChromeFromAudio()
    })

    wavesurfer.on('error', () => {
        setWaveformStatus('Failed to load waveform.')
    })

    updateRuntimeDiagnostics()
}

playToggleButton?.addEventListener('click', () => {
    void togglePlayback()
})

shareTrackButton?.addEventListener('click', () => {
    shareTrackWithFriends()
})

muteToggleButton?.addEventListener('click', () => {
    if (!audio) return

    if (audio.muted) {
        audio.muted = false
        if (audio.volume === 0) {
            audio.volume = previousVolume > 0 ? previousVolume : DEFAULT_RADIO_VOLUME
        }
    } else {
        previousVolume = audio.volume > 0 ? audio.volume : previousVolume
        audio.muted = true
    }

    if (!audio.muted) {
        persistVolumePreference(audio.volume)
    }

    updateVolumeUi()
})

volumeInput?.addEventListener('input', () => {
    if (!audio || !volumeInput) return

    const nextVolume = Math.max(0, Math.min(1, Number(volumeInput.value)))
    audio.volume = nextVolume
    audio.muted = nextVolume === 0

    if (nextVolume > 0) {
        previousVolume = nextVolume
    }

    persistVolumePreference(nextVolume)
    updateVolumeUi()
})

bitrateSelect?.addEventListener('change', () => {
    selectedBitrate = normalizeBitratePreference(bitrateSelect.value)
    persistBitratePreference(selectedBitrate)
    updateBitrateUi()
    publishRadioDiagnostics('bitrate-change', snapshot?.now_playing || null)

    if (snapshot?.now_playing) {
        syncAudio(snapshot.now_playing)
    }
})

transportSelect?.addEventListener('change', () => {
    selectedTransportMode = normalizeTransportPreference(transportSelect.value)
    persistTransportPreference(selectedTransportMode)
    nativeHlsFallbackTrackId = null
    updateTransportUi()
    publishRadioDiagnostics('transport-change', snapshot?.now_playing || null)

    if (!snapshot?.now_playing) {
        updateRuntimeDiagnostics()
        return
    }

    const current = snapshot.now_playing
    const preservedSeekTarget = audio && Number.isFinite(audio.currentTime) && audio.currentTime > 0
        ? Math.min(current.duration, Math.max(audio.currentTime, deriveLiveOffsetSeconds(current)))
        : deriveLiveOffsetSeconds(current)

    loadTrackSource(current, shouldAutoResumePlayback(), preservedSeekTarget)
})

document.addEventListener('click', (event) => {
    const target = event.target
    if (!(target instanceof HTMLElement)) return

    const stationToggle = target.closest('[data-radio-stations-toggle]')
    if (stationToggle instanceof HTMLElement) {
        stationsExpanded = stationToggle.getAttribute('data-radio-stations-toggle') === 'expand'
        renderStations(snapshot?.stations)
        return
    }

    const waveGenresToggle = target.closest('[data-radio-wave-genres-toggle]')
    if (waveGenresToggle instanceof HTMLElement) {
        waveGenresExpanded = waveGenresToggle.getAttribute('data-radio-wave-genres-toggle') === 'expand'
        renderWaveGenres(snapshot?.station)
        return
    }

    const waveStyleToggle = target.closest('[data-radio-wave-style]')
    if (waveStyleToggle instanceof HTMLElement) {
        const styleSlug = waveStyleToggle.getAttribute('data-radio-wave-style') || ''
        if (styleSlug) {
            togglePersonalWaveStyle(styleSlug)
        }
        return
    }

    const focusButton = target.closest('[data-radio-chat-focus-segment]')
    if (focusButton instanceof HTMLElement) {
        const segmentId = Number(focusButton.getAttribute('data-radio-chat-focus-segment') || '0')
        if (segmentId > 0) {
            focusSharedSegment(segmentId)
        }
        return
    }

    const shareButton = target.closest('[data-radio-share-segment]')
    if (shareButton instanceof HTMLElement) {
        const segmentId = Number(shareButton.getAttribute('data-radio-share-segment') || '0')
        if (segmentId > 0) {
            shareSegmentInChat(segmentId)
        }
        return
    }

    const directShareButton = target.closest('[data-radio-direct-share-segment]')
    if (directShareButton instanceof HTMLElement) {
        const segmentId = Number(directShareButton.getAttribute('data-radio-direct-share-segment') || '0')
        if (segmentId > 0) {
            shareSegmentWithFriends(segmentId)
        }
        return
    }

    const rateButton = target.closest('[data-radio-rate-kind]')
    if (!(rateButton instanceof HTMLElement)) return
    if (!authToken || feedbackTrackId === null) return

    const kind = rateButton.getAttribute('data-radio-rate-kind')
    const trackId = Number(rateButton.getAttribute('data-track-id') || '0')
    const rating = Number(rateButton.getAttribute('data-rating') || '0')
    const segmentId = Number(rateButton.getAttribute('data-segment-id') || '0')

    if (!trackId || !rating) return

    if (kind === 'track') {
        void submitTrackRating(trackId, rating).catch((err) => {
            setError(err instanceof Error ? err.message : 'Failed to rate track')
        })
        return
    }

    if (kind === 'segment' && segmentId) {
        void submitSegmentRating(trackId, segmentId, rating).catch((err) => {
            setError(err instanceof Error ? err.message : 'Failed to rate segment')
        })
    }
})

chatForm?.addEventListener('submit', (event) => {
    event.preventDefault()
    if (!chatInput) {
        return
    }

    sendChatMessage(chatInput.value)
    chatInput.value = ''
})

waveGenresPanel?.addEventListener('change', (event) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement) || !target.matches('[data-radio-wave-genre]')) {
        return
    }

    updatePersonalStationGenres()
})

audio?.addEventListener('loadedmetadata', () => {
    if (!audio) return

    if (pendingSeek !== null) {
        const duration = Number.isFinite(audio.duration) && audio.duration > 0
            ? audio.duration
            : (snapshot?.now_playing?.duration || pendingSeek)
        const maxTime = Math.max(0, duration - 1)
        audio.currentTime = Math.min(pendingSeek, maxTime)
        pendingSeek = null
    }

    syncPlaybackChromeFromAudio()
    updateVolumeUi()
    lastLiveSyncAt = Date.now()
    clearPlaybackRecoveryTimer()
    publishRadioDiagnostics('loadedmetadata', snapshot?.now_playing || null)
    if (shouldResumeAfterSourceLoad) {
        shouldResumeAfterSourceLoad = false
        void attemptPlayback()
        return
    }

    shouldResumeAfterSourceLoad = false
    updatePlayToggleLabel()
})

audio?.addEventListener('timeupdate', () => {
    syncPlaybackChromeFromAudio()
})

audio?.addEventListener('play', () => {
    clearError()
    playbackIntent = true
    lastLiveSyncAt = Date.now()
    clearPlaybackRecoveryTimer()
    syncPlaybackChromeFromAudio()
    publishRadioDiagnostics('play', snapshot?.now_playing || null)
    if (snapshot?.now_playing?.track_id && authToken) {
        void startListeningHistory(snapshot.now_playing.track_id)
        void loadFeedbackForTrack(snapshot.now_playing.track_id)
    }
    sendListenerPresence()
})

audio?.addEventListener('pause', () => {
    if (suppressPausePresence) {
        suppressPausePresence = false
        return
    }

    void completeListeningHistory(false)
    syncPlaybackChromeFromAudio()
    publishRadioDiagnostics('pause', snapshot?.now_playing || null)
    sendListenerPresence()
    if (!shouldAutoResumePlayback()) {
        unloadTrackSource()
    }
})

audio?.addEventListener('ended', () => {
    void completeListeningHistory(true)
    publishRadioDiagnostics('ended', snapshot?.now_playing || null)
    sendListenerPresence()
    requestStationSnapshot()
})

audio?.addEventListener('waiting', () => {
    publishRadioDiagnostics('waiting', snapshot?.now_playing || null)
    if (snapshot?.now_playing) {
        syncToLiveOffset(snapshot.now_playing)
        requestStationSnapshot()
    }

    if (hls) {
        hls.startLoad(Number.isFinite(audio.currentTime) ? audio.currentTime : -1)
        return
    }

    schedulePlaybackRecovery('waiting')
})

audio?.addEventListener('stalled', () => {
    publishRadioDiagnostics('stalled', snapshot?.now_playing || null)
    if (snapshot?.now_playing) {
        syncToLiveOffset(snapshot.now_playing)
        requestStationSnapshot()
    }

    if (hls) {
        hls.startLoad(Number.isFinite(audio.currentTime) ? audio.currentTime : -1)
        return
    }

    schedulePlaybackRecovery('stalled')
})

audio?.addEventListener('error', () => {
    publishRadioDiagnostics('audio-error', snapshot?.now_playing || null)
    if (snapshot?.now_playing) {
        const current = snapshot.now_playing
        requestStationSnapshot()
        const mediaErrorCode = audio.error?.code ?? null

        const shouldEscalateToNativeHls = (
            currentPlaybackMode === 'direct'
            && shouldAutoResumePlayback()
            && !Hls.isSupported()
            && canUseNativeHlsPlayback()
            && mediaErrorCode === MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED
            && nativeHlsFallbackTrackId !== current.track_id
        )

        if (shouldEscalateToNativeHls) {
            nativeHlsFallbackTrackId = current.track_id
            publishRadioDiagnostics('native-hls-fallback-armed', current)
            const fallbackSeekTarget = Number.isFinite(audio.currentTime) && audio.currentTime > 0
                ? Math.min(current.duration, Math.max(audio.currentTime, deriveLiveOffsetSeconds(current)))
                : deriveLiveOffsetSeconds(current)
            loadTrackSource(current, true, fallbackSeekTarget)
            return
        }
    }
    schedulePlaybackRecovery('audio-error')
})

audio?.addEventListener('volumechange', () => {
    updateVolumeUi()
})

document.addEventListener('visibilitychange', () => {
    sendListenerPresence()
    if (document.hidden) {
        void flushRadioDiagnostics(true)
        return
    }

    if (!document.hidden) {
        requestStationSnapshot()
    }
})

initWaveform()
updatePlayToggleLabel()
initializeVolumePreference()
initializeTransportPreference()
initializeBitratePreference()
applyPitchPreservation()
playbackIntent = true

void initializeAuth().finally(() => {
    if (!listenerSocket && (!isPersonalStation || Boolean(radioAccessToken))) {
        void connectListenerSocket()
    }

    if (isPersonalStation && !radioAccessToken) {
        setError('Sign in to open this personal radio station.')
    }

    renderStations(snapshot?.stations)
})

window.addEventListener('beforeunload', () => {
    shuttingDown = true
    void completeListeningHistory(false)
    void flushRadioDiagnostics(true)

    if (listenerReconnectTimer !== null) {
        window.clearTimeout(listenerReconnectTimer)
    }

    if (privateReconnectTimer !== null) {
        window.clearTimeout(privateReconnectTimer)
    }

    if (listenerPresenceTimer !== null) {
        window.clearInterval(listenerPresenceTimer)
    }

    closeListenerSocket()
    closePrivateSocket()
    destroyHls()
    clearWaveformSegments()
    wavesurfer?.destroy()
})
