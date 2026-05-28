import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { WebSocketServer, WebSocket } from 'ws'
import { createClient } from 'redis'

const wsPort = Number.parseInt(process.env.RADIO_WS_PORT || '8080', 10)
const radioAppOrigin = process.env.RADIO_APP_ORIGIN || 'http://app'
const redisHost = process.env.REDIS_HOST || 'redis'
const redisPort = Number.parseInt(process.env.REDIS_PORT || '6379', 10)
const redisPassword = process.env.REDIS_PASSWORD || ''
const redisPrefix = process.env.REDIS_PREFIX || 'ai_music_recommendations_database_'
const fallbackSnapshotIntervalMs = 60000
const heartbeatIntervalMs = 10000
const staleClientTimeoutMs = 25000
const hlsStorageRoot = '/var/www/html/storage/app/public_radio_hls'
const waveformStorageRoot = '/var/www/html/storage/app/public_radio_waveforms'
const chatHistoryLimit = 50
const chatMessageMaxLength = 400
const chatHistoryTtlSeconds = 604800

const redisUrl = redisPassword
    ? `redis://:${encodeURIComponent(redisPassword)}@${redisHost}:${redisPort}`
    : `redis://${redisHost}:${redisPort}`

const redis = createClient({ url: redisUrl })
await redis.connect()

const publicClients = new Map()
const stationClients = new Map()
const stationLoops = new Map()
const stationSnapshots = new Map()
const pendingStationSnapshots = new Map()
const publicClientSockets = new Map()
const prewarmedHlsTracks = new Set()
const pendingHlsWarmups = new Map()
const waveformPayloads = new Map()
const pendingWaveformPayloads = new Map()
const viewerStateCache = new Map()
const pendingViewerStates = new Map()
const pendingPrivateOverlays = new Map()

const privateClients = new Map()
const privateClientSockets = new Map()
const friendClients = new Map()
const friendUserSockets = new Map()

function isPersonalStationSlug(stationSlug) {
    return /^personal-\d+$/.test(normalizeStationSlug(stationSlug))
}

function normalizeStationSlug(rawSlug) {
    if (typeof rawSlug !== 'string') {
        return ''
    }

    return rawSlug.trim().toLowerCase()
}

function normalizeClientId(rawClientId) {
    if (typeof rawClientId !== 'string') {
        return ''
    }

    return rawClientId.trim()
}

function normalizeToken(rawToken) {
    if (typeof rawToken !== 'string') {
        return ''
    }

    return rawToken.trim()
}

function stationKey(stationSlug) {
    return `${redisPrefix}public_radio_listener_count:${stationSlug}`
}

function clientKey(stationSlug, clientId) {
    return `${stationSlug}:${clientId}`
}

function chatHistoryKey(stationSlug) {
    return `${redisPrefix}public_radio_chat:${stationSlug}`
}

function normalizeChatText(rawText) {
    if (typeof rawText !== 'string') {
        return ''
    }

    return rawText.trim().replace(/\s+/g, ' ').slice(0, chatMessageMaxLength)
}

function normalizeSegmentShare(rawSegment) {
    if (!rawSegment || typeof rawSegment !== 'object') {
        return null
    }

    const segment = rawSegment
    const trackId = Number(segment.trackId || 0)
    const segmentId = Number(segment.segmentId || 0)
    const start = Number(segment.start || 0)
    const end = Number(segment.end || 0)
    const label = typeof segment.label === 'string' ? segment.label.trim().slice(0, 80) : ''
    const trackTitle = typeof segment.trackTitle === 'string' ? segment.trackTitle.trim().slice(0, 160) : ''
    const artist = typeof segment.artist === 'string' ? segment.artist.trim().slice(0, 160) : ''

    if (trackId <= 0 || segmentId <= 0 || !Number.isFinite(start) || !Number.isFinite(end) || end <= start || !label || !trackTitle) {
        return null
    }

    return {
        trackId,
        trackTitle,
        artist,
        segmentId,
        label,
        start,
        end,
    }
}

function normalizeChatEntry(rawEntry) {
    if (!rawEntry || typeof rawEntry !== 'object') {
        return null
    }

    const entry = rawEntry
    if (typeof entry.id !== 'string' || typeof entry.stationSlug !== 'string' || typeof entry.createdAt !== 'string') {
        return null
    }

    if (!entry.user || typeof entry.user !== 'object' || !Number.isFinite(Number(entry.user.id))) {
        return null
    }

    if (entry.type === 'segment_share') {
        const segment = normalizeSegmentShare(entry.segment)
        if (!segment) {
            return null
        }

        return {
            id: entry.id,
            type: 'segment_share',
            stationSlug: entry.stationSlug,
            user: {
                id: Number(entry.user.id),
                name: typeof entry.user.name === 'string' ? entry.user.name : `User ${entry.user.id}`,
            },
            segment,
            createdAt: entry.createdAt,
        }
    }

    const text = normalizeChatText(entry.text)
    if (!text) {
        return null
    }

    return {
        id: entry.id,
        type: 'message',
        stationSlug: entry.stationSlug,
        user: {
            id: Number(entry.user.id),
            name: typeof entry.user.name === 'string' ? entry.user.name : `User ${entry.user.id}`,
        },
        text,
        createdAt: entry.createdAt,
    }
}

function parseChatEntry(rawEntry) {
    if (typeof rawEntry !== 'string' || rawEntry.trim() === '') {
        return null
    }

    try {
        return normalizeChatEntry(JSON.parse(rawEntry))
    } catch {
        return null
    }
}

async function loadChatHistory(stationSlug) {
    const history = await redis.lRange(chatHistoryKey(stationSlug), 0, chatHistoryLimit - 1)

    return history
        .map(parseChatEntry)
        .filter(Boolean)
        .reverse()
}

async function appendChatHistoryEntry(stationSlug, entry) {
    const normalizedEntry = normalizeChatEntry(entry)
    if (!normalizedEntry) {
        throw new Error('Invalid chat payload')
    }

    const key = chatHistoryKey(stationSlug)
    await redis.lPush(key, JSON.stringify(normalizedEntry))
    await redis.lTrim(key, 0, chatHistoryLimit - 1)
    await redis.expire(key, chatHistoryTtlSeconds)

    return normalizedEntry
}

function sendJson(socket, payload) {
    if (socket.readyState !== WebSocket.OPEN) return
    socket.send(JSON.stringify(payload))
}

function sendSocketError(socket, stationSlug, message) {
    sendJson(socket, {
        type: 'error',
        stationSlug,
        message,
    })
}

function buildStaticHlsMasterUrl(trackId) {
    return `/storage/public_radio_hls/${trackId}/current/master.m3u8`
}

function buildStaticWaveformUrl(trackId) {
    return `/storage/public_radio_waveforms/${trackId}/current.json`
}

function currentHlsMasterPath(trackId) {
    return `${hlsStorageRoot}/${trackId}/current/master.m3u8`
}

function currentWaveformPath(trackId) {
    return `${waveformStorageRoot}/${trackId}/current.json`
}

async function fileExists(path) {
    try {
        await fs.access(path)
        return true
    } catch {
        return false
    }
}

async function hasCurrentHlsPackage(trackId) {
    return Number.isFinite(trackId) && trackId > 0
        ? fileExists(currentHlsMasterPath(trackId))
        : false
}

async function readCurrentWaveformPayload(trackId) {
    if (!Number.isFinite(trackId) || trackId <= 0) {
        return null
    }

    const path = currentWaveformPath(trackId)
    if (!(await fileExists(path))) {
        return null
    }

    const payload = await fs.readFile(path, 'utf8')
    const decoded = JSON.parse(payload)
    if (!decoded || typeof decoded !== 'object') {
        return null
    }

    return decoded
}

function getStationSet(stationSlug) {
    if (!stationClients.has(stationSlug)) {
        stationClients.set(stationSlug, new Set())
    }

    return stationClients.get(stationSlug)
}

function stationClientCount(stationSlug) {
    const now = Date.now()
    const stationSet = stationClients.get(stationSlug) || new Set()
    const activeClientIds = new Set()

    for (const socket of stationSet) {
        const state = publicClients.get(socket)
        if (!state) continue
        if (!state.listening) continue
        if (!state.clientId) continue
        if (now - state.lastSeenAt > staleClientTimeoutMs) continue
        activeClientIds.add(state.clientId)
    }

    return activeClientIds.size
}

async function writeCount(stationSlug, count) {
    if (!stationSlug) return

    if (count > 0) {
        await redis.set(stationKey(stationSlug), String(count), { EX: 120 })
        return
    }

    await redis.del(stationKey(stationSlug))
}

async function broadcastCount(stationSlug) {
    if (!stationSlug) return

    const count = stationClientCount(stationSlug)
    await writeCount(stationSlug, count)

    const stationSet = stationClients.get(stationSlug) || new Set()
    for (const socket of stationSet) {
        sendJson(socket, {
            type: 'listener_count',
            stationSlug,
            count,
        })
    }
}

function pickStationAccessToken(stationSlug) {
    const stationSet = stationClients.get(stationSlug) || new Set()

    for (const socket of stationSet) {
        const state = publicClients.get(socket)
        if (!state?.accessToken) continue
        return state.accessToken
    }

    return ''
}

async function fetchStationSnapshot(stationSlug, accessToken = '') {
    const requestKey = `${stationSlug}:${accessToken}`
    const pending = pendingStationSnapshots.get(requestKey)
    if (pending) {
        return pending
    }

    const url = new URL(`${radioAppOrigin}/api/public/radio/${stationSlug}`)
    if (accessToken) {
        url.searchParams.set('access_token', accessToken)
    }

    const request = (async () => {
        const response = await fetch(url, {
            headers: { Accept: 'application/json' },
        })

        if (response.status === 401 || response.status === 403) {
            throw new Error('Unauthorized station access')
        }

        if (!response.ok) {
            throw new Error(`Failed to fetch station snapshot (${response.status})`)
        }

        const snapshot = await response.json()
        await prepareSnapshotAssets(snapshot)
        snapshot.listener_count = stationClientCount(stationSlug)

        return snapshot
    })()

    pendingStationSnapshots.set(requestKey, request)

    try {
        return await request
    } finally {
        pendingStationSnapshots.delete(requestKey)
    }
}

async function prewarmTrackHls(trackId) {
    if (!Number.isFinite(trackId) || trackId <= 0) {
        return false
    }

    if (await hasCurrentHlsPackage(trackId)) {
        prewarmedHlsTracks.add(trackId)
        return true
    }

    if (prewarmedHlsTracks.has(trackId)) {
        return true
    }

    const pending = pendingHlsWarmups.get(trackId)
    if (pending) {
        return pending
    }

    const request = (async () => {
        const response = await fetch(`${radioAppOrigin}/api/tracks/${trackId}/hls/master.m3u8`, {
            headers: { Accept: 'application/vnd.apple.mpegurl' },
        })

        if (!response.ok) {
            throw new Error(`Failed to prewarm HLS track ${trackId} (${response.status})`)
        }

        await response.arrayBuffer()
        prewarmedHlsTracks.add(trackId)
        return true
    })()

    pendingHlsWarmups.set(trackId, request)

    try {
        return await request
    } finally {
        pendingHlsWarmups.delete(trackId)
    }
}

async function fetchWaveformPayload(trackId) {
    if (!Number.isFinite(trackId) || trackId <= 0) {
        throw new Error('Waveform track id is required')
    }

    const localPayload = await readCurrentWaveformPayload(trackId)
    if (localPayload) {
        waveformPayloads.set(trackId, localPayload)
        return localPayload
    }

    if (waveformPayloads.has(trackId)) {
        return waveformPayloads.get(trackId)
    }

    const pending = pendingWaveformPayloads.get(trackId)
    if (pending) {
        return pending
    }

    const request = (async () => {
        const response = await fetch(`${radioAppOrigin}/api/tracks/${trackId}/waveform`, {
            headers: { Accept: 'application/json' },
        })

        if (!response.ok) {
            throw new Error(`Failed to load waveform for track ${trackId} (${response.status})`)
        }

        const payload = await response.json()
        waveformPayloads.set(trackId, payload)
        return payload
    })()

    pendingWaveformPayloads.set(trackId, request)

    try {
        return await request
    } finally {
        pendingWaveformPayloads.delete(trackId)
    }
}

function rewriteSnapshotTrack(track) {
    if (!track || typeof track !== 'object') {
        return track
    }

    const trackId = Number(track.track_id || 0)
    if (!trackId) {
        return track
    }

    return {
        ...track,
        hls_master_url: buildStaticHlsMasterUrl(trackId),
        waveform_url: buildStaticWaveformUrl(trackId),
    }
}

function comparableSnapshot(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') {
        return snapshot
    }

    const normalized = { ...snapshot }
    delete normalized.generated_at
    delete normalized.listener_count
    delete normalized.viewer

    return normalized
}

async function prepareSnapshotAssets(snapshot) {
    const currentTrackId = Number(snapshot?.now_playing?.track_id || 0)
    if (currentTrackId > 0) {
        const currentReady = await hasCurrentHlsPackage(currentTrackId)
        snapshot.now_playing = currentReady ? rewriteSnapshotTrack(snapshot.now_playing) : snapshot.now_playing
        if (!currentReady) {
            void prewarmTrackHls(currentTrackId).catch(() => {})
        }
        snapshot.now_playing_waveform = await fetchWaveformPayload(currentTrackId).catch(() => null)
    } else {
        snapshot.now_playing_waveform = null
    }

    if (Array.isArray(snapshot?.up_next)) {
        const rewritten = []
        for (const track of snapshot.up_next) {
            const trackId = Number(track?.track_id || 0)
            if (trackId > 0 && await hasCurrentHlsPackage(trackId)) {
                rewritten.push(rewriteSnapshotTrack(track))
                continue
            }

            rewritten.push(track)
        }
        snapshot.up_next = rewritten

        const nextTrackId = Number(snapshot.up_next[0]?.track_id || 0)
        if (nextTrackId > 0) {
            void prewarmTrackHls(nextTrackId).catch(() => {})
        }
    }

    if (Array.isArray(snapshot?.history)) {
        const rewrittenHistory = []
        for (const track of snapshot.history) {
            const trackId = Number(track?.track_id || 0)
            if (trackId > 0 && await hasCurrentHlsPackage(trackId)) {
                rewrittenHistory.push(rewriteSnapshotTrack(track))
                continue
            }

            rewrittenHistory.push(track)
        }
        snapshot.history = rewrittenHistory
    }
}

function nextSnapshotDelay(snapshot) {
    const remainingSeconds = Number(snapshot?.now_playing?.remaining_seconds || 0)
    if (!Number.isFinite(remainingSeconds) || remainingSeconds <= 0) {
        return fallbackSnapshotIntervalMs
    }

    if (remainingSeconds > 60) {
        return 60000
    }

    return Math.max(1250, Math.round((remainingSeconds * 1000) + 500))
}

function scheduleStationLoop(stationSlug, snapshot = null) {
    const existing = stationLoops.get(stationSlug)
    if (existing) {
        clearTimeout(existing)
    }

    const timer = setTimeout(() => {
        void broadcastSnapshot(stationSlug).then((nextSnapshot) => {
            scheduleStationLoop(stationSlug, nextSnapshot)
        }).catch((error) => {
            const stationSet = stationClients.get(stationSlug) || new Set()
            const message = error instanceof Error ? error.message : 'Failed to fetch station snapshot'
            for (const socket of stationSet) {
                sendSocketError(socket, stationSlug, message)
            }
            scheduleStationLoop(stationSlug, null)
        })
    }, nextSnapshotDelay(snapshot))

    stationLoops.set(stationSlug, timer)
}

async function broadcastSnapshot(stationSlug, force = false) {
    const stationSet = stationClients.get(stationSlug) || new Set()
    if (stationSet.size === 0) {
        return null
    }

    const accessToken = isPersonalStationSlug(stationSlug) ? pickStationAccessToken(stationSlug) : ''
    if (isPersonalStationSlug(stationSlug) && !accessToken) {
        throw new Error('Personal station access token is required')
    }

    const snapshot = await fetchStationSnapshot(stationSlug, accessToken)
    delete snapshot.viewer
    const serialized = JSON.stringify(comparableSnapshot(snapshot))

    if (!force && stationSnapshots.get(stationSlug) === serialized) {
        return snapshot
    }

    pushStationSnapshot(stationSlug, snapshot)

    return snapshot
}

function ensureStationLoop(stationSlug) {
    if (!stationSlug || stationLoops.has(stationSlug)) {
        return
    }

    scheduleStationLoop(stationSlug, null)
}

function stopStationLoopIfUnused(stationSlug) {
    const stationSet = stationClients.get(stationSlug)
    if (stationSet && stationSet.size > 0) {
        return
    }

    const timer = stationLoops.get(stationSlug)
    if (timer) {
        clearTimeout(timer)
        stationLoops.delete(stationSlug)
    }

    stationSnapshots.delete(stationSlug)
}

function detachPublicClient(socket, stationSlug) {
    if (!stationSlug) return

    const stationSet = stationClients.get(stationSlug)
    if (!stationSet) return

    stationSet.delete(socket)
    if (stationSet.size === 0) {
        stationClients.delete(stationSlug)
    }

    stopStationLoopIfUnused(stationSlug)
}

async function bindPublicSocketIdentity(socket, stationSlug, clientId) {
    if (!stationSlug || !clientId) {
        return
    }

    const nextKey = clientKey(stationSlug, clientId)
    const existingSocket = publicClientSockets.get(nextKey)
    if (existingSocket && existingSocket !== socket) {
        try {
            existingSocket.close(4001, 'superseded')
        } catch {
            existingSocket.terminate()
        }
    }

    publicClientSockets.set(nextKey, socket)
}

function unbindPublicSocketIdentity(socket, stationSlug, clientId) {
    if (!stationSlug || !clientId) {
        return
    }

    const key = clientKey(stationSlug, clientId)
    if (publicClientSockets.get(key) === socket) {
        publicClientSockets.delete(key)
    }
}

async function updatePublicClientState(socket, payload) {
    const state = publicClients.get(socket)
    if (!state) return

    const previousStationSlug = state.stationSlug
    const previousClientId = state.clientId
    const normalizedStationSlug = normalizeStationSlug(payload.stationSlug || previousStationSlug)
    const normalizedClientId = normalizeClientId(payload.clientId || previousClientId || state.id)
    const normalizedAccessToken = normalizeToken(payload.accessToken || state.accessToken)
    const accessTokenChanged = state.accessToken !== normalizedAccessToken

    if (!normalizedStationSlug) {
        return
    }

    if (isPersonalStationSlug(normalizedStationSlug)) {
        if (!normalizedAccessToken) {
            throw new Error('Personal station access token is required')
        }

        const needsAuthValidation = (
            previousStationSlug !== normalizedStationSlug
            || state.accessToken !== normalizedAccessToken
            || payload.type === 'subscribe'
            || payload.type === 'request_snapshot'
        )

        if (needsAuthValidation) {
            await fetchStationSnapshot(normalizedStationSlug, normalizedAccessToken)
        }
    }

    state.lastSeenAt = Date.now()
    state.listening = Boolean(payload.listening)
    state.accessToken = normalizedAccessToken
    if (accessTokenChanged || !normalizedAccessToken) {
        state.viewer = null
    }

    if (previousStationSlug !== normalizedStationSlug) {
        detachPublicClient(socket, previousStationSlug)
        getStationSet(normalizedStationSlug).add(socket)
        state.stationSlug = normalizedStationSlug
    }

    if (previousClientId && (previousClientId !== normalizedClientId || previousStationSlug !== normalizedStationSlug)) {
        unbindPublicSocketIdentity(socket, previousStationSlug, previousClientId)
    }

    state.clientId = normalizedClientId
    await bindPublicSocketIdentity(socket, normalizedStationSlug, normalizedClientId)
    ensureStationLoop(normalizedStationSlug)

    if (previousStationSlug && previousStationSlug !== normalizedStationSlug) {
        await broadcastCount(previousStationSlug)
    }

    await broadcastCount(normalizedStationSlug)

}

async function removePublicClient(socket) {
    const state = publicClients.get(socket)
    if (!state) return

    publicClients.delete(socket)
    unbindPublicSocketIdentity(socket, state.stationSlug, state.clientId)
    detachPublicClient(socket, state.stationSlug)

    if (state.stationSlug) {
        await broadcastCount(state.stationSlug)
    }
}

async function bindPrivateSocketIdentity(socket, stationSlug, clientId) {
    if (!stationSlug || !clientId) {
        return
    }

    const nextKey = clientKey(stationSlug, clientId)
    const existingSocket = privateClientSockets.get(nextKey)
    if (existingSocket && existingSocket !== socket) {
        try {
            existingSocket.close(4001, 'superseded')
        } catch {
            existingSocket.terminate()
        }
    }

    privateClientSockets.set(nextKey, socket)
}

function unbindPrivateSocketIdentity(socket, stationSlug, clientId) {
    if (!stationSlug || !clientId) {
        return
    }

    const key = clientKey(stationSlug, clientId)
    if (privateClientSockets.get(key) === socket) {
        privateClientSockets.delete(key)
    }
}

async function updatePrivateClientState(socket, payload) {
    const state = privateClients.get(socket)
    if (!state) return

    const previousStationSlug = state.stationSlug
    const previousClientId = state.clientId
    const normalizedStationSlug = normalizeStationSlug(payload.stationSlug || previousStationSlug)
    const normalizedClientId = normalizeClientId(payload.clientId || previousClientId || state.id)
    const normalizedToken = normalizeToken(payload.accessToken || state.token)

    if (!normalizedStationSlug) {
        return
    }

    state.lastSeenAt = Date.now()
    state.stationSlug = normalizedStationSlug
    state.clientId = normalizedClientId
    state.token = normalizedToken

    if (previousClientId && (previousClientId !== normalizedClientId || previousStationSlug !== normalizedStationSlug)) {
        unbindPrivateSocketIdentity(socket, previousStationSlug, previousClientId)
    }

    await bindPrivateSocketIdentity(socket, normalizedStationSlug, normalizedClientId)
}

async function fetchJson(url, headers) {
    const response = await fetch(url, { headers })

    if (response.status === 401 || response.status === 403) {
        return { unauthorized: true }
    }

    if (!response.ok) {
        throw new Error(`Request failed (${response.status})`)
    }

    return {
        unauthorized: false,
        payload: await response.json(),
    }
}

async function fetchAuthedJson(path, accessToken, init = {}) {
    if (!accessToken) {
        return { unauthorized: true }
    }

    const headers = {
        Accept: 'application/json',
        ...(init.body ? { 'Content-Type': 'application/json' } : {}),
        ...(init.headers || {}),
        Authorization: `Bearer ${accessToken}`,
    }

    const response = await fetch(`${radioAppOrigin}${path}`, {
        ...init,
        headers,
    })

    if (response.status === 401 || response.status === 403) {
        return { unauthorized: true }
    }

    if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.message || `Request failed (${response.status})`)
    }

    return {
        unauthorized: false,
        payload: await response.json(),
    }
}

function getFriendSocketSet(userId) {
    if (!friendUserSockets.has(userId)) {
        friendUserSockets.set(userId, new Set())
    }

    return friendUserSockets.get(userId)
}

function attachFriendSocketUser(socket, userId) {
    if (!Number.isFinite(userId) || userId <= 0) {
        return
    }

    getFriendSocketSet(userId).add(socket)
}

function detachFriendSocketUser(socket, userId) {
    if (!Number.isFinite(userId) || userId <= 0) {
        return
    }

    const socketSet = friendUserSockets.get(userId)
    if (!socketSet) {
        return
    }

    socketSet.delete(socket)
    if (socketSet.size === 0) {
        friendUserSockets.delete(userId)
    }
}

async function fetchAuthenticatedUser(accessToken) {
    const result = await fetchAuthedJson('/api/auth/me', accessToken)
    if (result.unauthorized) {
        return null
    }

    return result.payload?.user || null
}

async function fetchFriendState(accessToken) {
    return fetchAuthedJson('/api/friends/state', accessToken)
}

async function searchFriendUsers(accessToken, query) {
    const url = new URL(`${radioAppOrigin}/api/friends/search`)
    if (query) {
        url.searchParams.set('query', query)
    }

    const response = await fetch(url, {
        headers: {
            Accept: 'application/json',
            Authorization: `Bearer ${accessToken}`,
        },
    })

    if (response.status === 401 || response.status === 403) {
        return { unauthorized: true }
    }

    if (!response.ok) {
        const payload = await response.json().catch(() => null)
        throw new Error(payload?.message || `Request failed (${response.status})`)
    }

    return {
        unauthorized: false,
        payload: await response.json(),
    }
}

async function sendFriendRequest(accessToken, userId) {
    return fetchAuthedJson('/api/friends/requests', accessToken, {
        method: 'POST',
        body: JSON.stringify({ user_id: userId }),
    })
}

async function respondToFriendRequest(accessToken, friendshipId, action) {
    return fetchAuthedJson(`/api/friends/requests/${friendshipId}/respond`, accessToken, {
        method: 'POST',
        body: JSON.stringify({ action }),
    })
}

async function fetchFriendConversation(accessToken, userId) {
    return fetchAuthedJson(`/api/friends/conversations/${userId}`, accessToken)
}

async function sendFriendDirectMessage(accessToken, recipientId, text, share) {
    return fetchAuthedJson('/api/friends/messages', accessToken, {
        method: 'POST',
        body: JSON.stringify({
            recipient_id: recipientId,
            text,
            share,
        }),
    })
}

async function markFriendConversationRead(accessToken, userId) {
    return fetchAuthedJson(`/api/friends/conversations/${userId}/read`, accessToken, {
        method: 'POST',
    })
}

async function ensureFriendClientAuth(socket, payload = {}) {
    const state = friendClients.get(socket)
    if (!state) {
        return null
    }

    const nextToken = normalizeToken(payload.accessToken || state.accessToken)
    if (!nextToken) {
        throw new Error('Authentication token is required')
    }

    const previousToken = state.accessToken
    const previousUserId = Number(state.user?.id || 0)
    state.lastSeenAt = Date.now()
    state.accessToken = nextToken

    if (!state.user || previousToken !== nextToken) {
        if (previousUserId > 0) {
            detachFriendSocketUser(socket, previousUserId)
        }

        const viewer = await fetchAuthenticatedUser(nextToken)
        if (!viewer?.id) {
            state.user = null
            throw new Error('Authentication token is invalid')
        }

        state.user = viewer
        attachFriendSocketUser(socket, Number(viewer.id))
    }

    return state
}

async function sendFriendsBootstrap(socket) {
    const state = friendClients.get(socket)
    if (!state?.accessToken) {
        sendJson(socket, { type: 'friends_unauthorized' })
        return
    }

    const result = await fetchFriendState(state.accessToken)
    if (result.unauthorized) {
        const previousUserId = Number(state.user?.id || 0)
        if (previousUserId > 0) {
            detachFriendSocketUser(socket, previousUserId)
        }
        state.user = null
        sendJson(socket, { type: 'friends_unauthorized' })
        return
    }

    state.user = result.payload?.viewer || state.user
    sendJson(socket, {
        type: 'friends_bootstrap',
        state: result.payload,
    })
}

async function broadcastFriendsBootstrapToUser(userId) {
    const socketSet = friendUserSockets.get(Number(userId)) || new Set()
    for (const socket of socketSet) {
        try {
            await sendFriendsBootstrap(socket)
        } catch {
            sendJson(socket, { type: 'friends_error', message: 'Failed to refresh friends state' })
        }
    }
}

async function broadcastFriendsBootstrapToUsers(userIds) {
    const uniqueIds = [...new Set(userIds.map((userId) => Number(userId)).filter((userId) => userId > 0))]
    for (const userId of uniqueIds) {
        await broadcastFriendsBootstrapToUser(userId)
    }
}

async function sendConversationHistory(socket, userId) {
    const state = friendClients.get(socket)
    if (!state?.accessToken) {
        sendJson(socket, { type: 'friends_unauthorized' })
        return
    }

    const result = await fetchFriendConversation(state.accessToken, userId)
    if (result.unauthorized) {
        sendJson(socket, { type: 'friends_unauthorized' })
        return
    }

    sendJson(socket, {
        type: 'conversation_history',
        userId,
        conversationWith: result.payload?.conversation_with || null,
        messages: Array.isArray(result.payload?.messages) ? result.payload.messages : [],
    })
}

function broadcastDirectMessage(message) {
    const senderId = Number(message?.sender?.id || 0)
    const recipientId = Number(message?.recipient?.id || 0)
    const userIds = [senderId, recipientId].filter((userId) => userId > 0)

    for (const userId of userIds) {
        const socketSet = friendUserSockets.get(userId) || new Set()
        for (const socket of socketSet) {
            sendJson(socket, {
                type: 'direct_message',
                message,
            })
        }
    }
}

async function removeFriendClient(socket) {
    const state = friendClients.get(socket)
    if (!state) {
        return
    }

    friendClients.delete(socket)
    detachFriendSocketUser(socket, Number(state.user?.id || 0))
}

async function fetchPrivateOverlay(trackId, accessToken) {
    const url = new URL(`${radioAppOrigin}/api/radio/private-overlay/${trackId}`)
    url.searchParams.set('access_token', accessToken)

    const requestKey = `${trackId}:${accessToken}`
    const pending = pendingPrivateOverlays.get(requestKey)
    if (pending) {
        return pending
    }

    const request = (async () => {
        const overlayResult = await fetchJson(url, { Accept: 'application/json' })
        if (overlayResult.unauthorized) {
            return { unauthorized: true }
        }

        return {
            unauthorized: false,
            overlay: overlayResult.payload,
        }
    })()

    pendingPrivateOverlays.set(requestKey, request)

    try {
        return await request
    } finally {
        pendingPrivateOverlays.delete(requestKey)
    }
}

async function fetchViewerState(stationSlug, accessToken) {
    if (!accessToken) {
        return null
    }

    if (viewerStateCache.has(accessToken)) {
        return viewerStateCache.get(accessToken)
    }

    const requestKey = `${stationSlug}:${accessToken}`
    const pending = pendingViewerStates.get(requestKey)
    if (pending) {
        return pending
    }

    const request = (async () => {
        const snapshot = await fetchStationSnapshot(stationSlug, accessToken)
        const viewer = snapshot?.viewer && typeof snapshot.viewer === 'object' ? snapshot.viewer : null
        viewerStateCache.set(accessToken, viewer)
        return viewer
    })()

    pendingViewerStates.set(requestKey, request)

    try {
        return await request
    } finally {
        pendingViewerStates.delete(requestKey)
    }
}

async function sendViewerState(socket, stationSlug, accessToken) {
    const viewer = await fetchViewerState(stationSlug, accessToken).catch(() => null)
    const state = publicClients.get(socket)
    if (state) {
        state.viewer = viewer
    }
    sendJson(socket, {
        type: 'viewer_state',
        stationSlug,
        viewer,
    })
}

async function resolvePublicViewer(state, stationSlug) {
    if (!state?.accessToken) {
        state.viewer = null
        return null
    }

    const viewer = await fetchViewerState(stationSlug, state.accessToken).catch(() => null)
    state.viewer = viewer
    return viewer
}

async function sendChatHistory(socket, stationSlug) {
    const entries = await loadChatHistory(stationSlug)
    sendJson(socket, {
        type: 'chat_history',
        stationSlug,
        entries,
    })
}

async function broadcastChatEntry(stationSlug, entry) {
    const stationSet = stationClients.get(stationSlug) || new Set()

    for (const socket of stationSet) {
        const state = publicClients.get(socket)
        if (!state?.viewer?.id) {
            continue
        }

        sendJson(socket, {
            type: 'chat_message',
            stationSlug,
            entry,
        })
    }
}

async function updatePersonalStationGenres(stationSlug, accessToken, preferredGenres) {
    const url = new URL(`${radioAppOrigin}/api/radio/personal-stations/${stationSlug}/genres`)
    if (accessToken) {
        url.searchParams.set('access_token', accessToken)
    }

    const response = await fetch(url, {
        method: 'PUT',
        headers: {
            Accept: 'application/json',
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            preferred_genres: Array.isArray(preferredGenres) ? preferredGenres : [],
        }),
    })

    if (response.status === 401 || response.status === 403) {
        throw new Error('Unauthorized station access')
    }

    if (response.status === 404) {
        throw new Error('Personal station not found')
    }

    if (!response.ok) {
        throw new Error(`Failed to update personal station genres (${response.status})`)
    }

    const snapshot = await response.json()
    await prepareSnapshotAssets(snapshot)
    snapshot.listener_count = stationClientCount(stationSlug)

    return snapshot
}

function pushStationSnapshot(stationSlug, snapshot) {
    delete snapshot.viewer
    stationSnapshots.set(stationSlug, JSON.stringify(comparableSnapshot(snapshot)))

    const stationSet = stationClients.get(stationSlug) || new Set()
    for (const socket of stationSet) {
        sendJson(socket, {
            type: 'station_snapshot',
            stationSlug,
            snapshot,
        })
    }
}

async function sendOverlay(socket, stationSlug, trackId, accessToken) {
    if (!accessToken) {
        sendJson(socket, {
            type: 'private_overlay_unauthorized',
            stationSlug,
            trackId,
        })
        return
    }

    const result = await fetchPrivateOverlay(trackId, accessToken)
    if (result.unauthorized) {
        sendJson(socket, {
            type: 'private_overlay_unauthorized',
            stationSlug,
            trackId,
        })
        return
    }

    sendJson(socket, {
        type: 'private_overlay',
        stationSlug,
        trackId,
        overlay: result.overlay,
    })
}

async function sendPublicWaveform(socket, stationSlug, trackId) {
    const waveform = await fetchWaveformPayload(trackId)
    sendJson(socket, {
        type: 'waveform_payload',
        stationSlug,
        trackId,
        waveform,
    })
}

async function removePrivateClient(socket) {
    const state = privateClients.get(socket)
    if (!state) return

    privateClients.delete(socket)
    unbindPrivateSocketIdentity(socket, state.stationSlug, state.clientId)
}

function heartbeatClientMap(clientMap) {
    const now = Date.now()

    for (const [socket, state] of clientMap.entries()) {
        if (now - state.lastSeenAt > staleClientTimeoutMs) {
            socket.terminate()
            continue
        }

        if (!state.isAlive) {
            socket.terminate()
            continue
        }

        state.isAlive = false
        socket.ping()
    }
}

const server = createServer((request, response) => {
    response.writeHead(426, { 'Content-Type': 'text/plain' })
    response.end('Upgrade Required')
})

const publicWss = new WebSocketServer({ noServer: true })
const privateWss = new WebSocketServer({ noServer: true })
const friendsWss = new WebSocketServer({ noServer: true })

server.on('upgrade', (request, socket, head) => {
    const requestUrl = new URL(request.url || '/', 'http://localhost')

    if (requestUrl.pathname === '/ws/radio-listeners') {
        publicWss.handleUpgrade(request, socket, head, (ws) => {
            publicWss.emit('connection', ws, request, requestUrl)
        })
        return
    }

    if (requestUrl.pathname === '/ws/radio-private') {
        privateWss.handleUpgrade(request, socket, head, (ws) => {
            privateWss.emit('connection', ws, request, requestUrl)
        })
        return
    }

    if (requestUrl.pathname === '/ws/friends') {
        friendsWss.handleUpgrade(request, socket, head, (ws) => {
            friendsWss.emit('connection', ws, request, requestUrl)
        })
        return
    }

    socket.destroy()
})

publicWss.on('connection', async (socket, _request, requestUrl) => {
    const initialStationSlug = normalizeStationSlug(requestUrl.searchParams.get('station') || '')
    const initialClientId = normalizeClientId(requestUrl.searchParams.get('clientId') || randomUUID())
    const initialAccessToken = normalizeToken(requestUrl.searchParams.get('accessToken') || '')
    const state = {
        id: randomUUID(),
        // Start detached so the first subscribe/request_snapshot attaches this socket to the station fanout set.
        stationSlug: '',
        clientId: initialClientId,
        listening: false,
        accessToken: initialAccessToken,
        viewer: null,
        lastSeenAt: Date.now(),
        isAlive: true,
    }

    publicClients.set(socket, state)

    socket.on('pong', () => {
        const current = publicClients.get(socket)
        if (current) {
            current.isAlive = true
            current.lastSeenAt = Date.now()
        }
    })

    socket.on('message', async (message) => {
        let payload

        try {
            payload = JSON.parse(String(message))
        } catch {
            return
        }

        if (payload?.type === 'presence' || payload?.type === 'subscribe' || payload?.type === 'request_snapshot') {
            try {
                await updatePublicClientState(socket, payload)
                const state = publicClients.get(socket)
                const stationSlug = normalizeStationSlug(payload.stationSlug || state?.stationSlug || '')
                if (state?.accessToken) {
                    await sendViewerState(socket, stationSlug, state.accessToken)
                }
                const snapshot = await broadcastSnapshot(stationSlug, true)
                scheduleStationLoop(stationSlug, snapshot)
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to update radio socket state')
            }
            return
        }

        if (payload?.type === 'request_overlay') {
            try {
                await updatePublicClientState(socket, payload)
                const state = publicClients.get(socket)
                const stationSlug = normalizeStationSlug(payload.stationSlug || state?.stationSlug || '')
                const trackId = Number(payload.trackId || 0)
                if (trackId > 0 && state) {
                    await sendOverlay(socket, stationSlug, trackId, state.accessToken)
                }
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to load private overlay')
            }
            return
        }

        if (payload?.type === 'request_chat_history') {
            try {
                await updatePublicClientState(socket, payload)
                const state = publicClients.get(socket)
                const stationSlug = normalizeStationSlug(payload.stationSlug || state?.stationSlug || '')
                const viewer = await resolvePublicViewer(state, stationSlug)
                if (!viewer) {
                    sendJson(socket, {
                        type: 'chat_unauthorized',
                        stationSlug,
                    })
                    return
                }

                await sendChatHistory(socket, stationSlug)
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to load chat history')
            }
            return
        }

        if (payload?.type === 'chat_message') {
            try {
                await updatePublicClientState(socket, payload)
                const state = publicClients.get(socket)
                const stationSlug = normalizeStationSlug(payload.stationSlug || state?.stationSlug || '')
                const viewer = await resolvePublicViewer(state, stationSlug)
                if (!viewer) {
                    sendJson(socket, {
                        type: 'chat_unauthorized',
                        stationSlug,
                    })
                    return
                }

                const text = normalizeChatText(payload.text)
                if (!text) {
                    throw new Error('Chat message is empty')
                }

                const entry = await appendChatHistoryEntry(stationSlug, {
                    id: randomUUID(),
                    type: 'message',
                    stationSlug,
                    user: {
                        id: Number(viewer.id || 0),
                        name: typeof viewer.name === 'string' && viewer.name.trim() !== ''
                            ? viewer.name
                            : `User ${viewer.id}`,
                    },
                    text,
                    createdAt: new Date().toISOString(),
                })

                await broadcastChatEntry(stationSlug, entry)
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to send chat message')
            }
            return
        }

        if (payload?.type === 'chat_share_segment') {
            try {
                await updatePublicClientState(socket, payload)
                const state = publicClients.get(socket)
                const stationSlug = normalizeStationSlug(payload.stationSlug || state?.stationSlug || '')
                const viewer = await resolvePublicViewer(state, stationSlug)
                if (!viewer) {
                    sendJson(socket, {
                        type: 'chat_unauthorized',
                        stationSlug,
                    })
                    return
                }

                const segment = normalizeSegmentShare(payload.segment)
                if (!segment) {
                    throw new Error('Shared segment payload is invalid')
                }

                const entry = await appendChatHistoryEntry(stationSlug, {
                    id: randomUUID(),
                    type: 'segment_share',
                    stationSlug,
                    user: {
                        id: Number(viewer.id || 0),
                        name: typeof viewer.name === 'string' && viewer.name.trim() !== ''
                            ? viewer.name
                            : `User ${viewer.id}`,
                    },
                    segment,
                    createdAt: new Date().toISOString(),
                })

                await broadcastChatEntry(stationSlug, entry)
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to share segment in chat')
            }
            return
        }

        if (payload?.type === 'update_personal_station_genres') {
            try {
                await updatePublicClientState(socket, payload)
                const state = publicClients.get(socket)
                const stationSlug = normalizeStationSlug(payload.stationSlug || state?.stationSlug || '')
                if (!stationSlug || !state?.accessToken) {
                    throw new Error('Personal station access token is required')
                }

                const updatedSnapshot = await updatePersonalStationGenres(stationSlug, state.accessToken, payload.preferredGenres)
                pushStationSnapshot(stationSlug, updatedSnapshot)
                await broadcastCount(stationSlug)
                scheduleStationLoop(stationSlug, updatedSnapshot)
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to update personal station genres')
            }
            return
        }

        if (payload?.type === 'request_waveform') {
            try {
                const state = publicClients.get(socket)
                const stationSlug = normalizeStationSlug(payload.stationSlug || state?.stationSlug || '')
                const trackId = Number(payload.trackId || 0)
                if (trackId > 0) {
                    await sendPublicWaveform(socket, stationSlug, trackId)
                }
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to load waveform')
            }
            return
        }

        if (payload?.type === 'prewarm_track') {
            try {
                const trackId = Number(payload.trackId || 0)
                if (trackId > 0) {
                    await prewarmTrackHls(trackId)
                }
            } catch (error) {
                const stationSlug = normalizeStationSlug(payload.stationSlug || publicClients.get(socket)?.stationSlug || '')
                sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to prewarm track')
            }
        }
    })

    socket.on('close', async () => {
        try {
            await removePublicClient(socket)
        } catch {
            // Ignore cleanup errors during socket shutdown.
        }
    })

    socket.on('error', async () => {
        try {
            await removePublicClient(socket)
        } catch {
            // Ignore cleanup errors during socket shutdown.
        }
    })

    if (initialStationSlug) {
        try {
            await updatePublicClientState(socket, {
                type: 'request_snapshot',
                stationSlug: initialStationSlug,
                clientId: initialClientId,
                accessToken: initialAccessToken,
                listening: false,
            })
        } catch (error) {
            sendSocketError(socket, initialStationSlug, error instanceof Error ? error.message : 'Failed to fetch station snapshot')
        }
    }
})

privateWss.on('connection', async (socket, _request, requestUrl) => {
    const initialStationSlug = normalizeStationSlug(requestUrl.searchParams.get('station') || '')
    const initialClientId = normalizeClientId(requestUrl.searchParams.get('clientId') || randomUUID())
    const state = {
        id: randomUUID(),
        stationSlug: initialStationSlug,
        clientId: initialClientId,
        token: normalizeToken(requestUrl.searchParams.get('accessToken') || ''),
        lastSeenAt: Date.now(),
        isAlive: true,
    }

    privateClients.set(socket, state)

    socket.on('pong', () => {
        const current = privateClients.get(socket)
        if (current) {
            current.isAlive = true
            current.lastSeenAt = Date.now()
        }
    })

    socket.on('message', async (message) => {
        let payload

        try {
            payload = JSON.parse(String(message))
        } catch {
            return
        }

        if (payload?.type !== 'subscribe_private' && payload?.type !== 'request_overlay') {
            return
        }

        try {
            await updatePrivateClientState(socket, payload)
            const trackId = Number(payload.trackId || 0)
            if (payload.type === 'request_overlay' && trackId > 0) {
                const state = privateClients.get(socket)
                if (state) {
                    await sendOverlay(socket, state.stationSlug, trackId, state.token)
                }
            }
        } catch (error) {
            const stationSlug = normalizeStationSlug(payload.stationSlug || privateClients.get(socket)?.stationSlug || '')
            sendSocketError(socket, stationSlug, error instanceof Error ? error.message : 'Failed to update private radio socket state')
        }
    })

    socket.on('close', async () => {
        try {
            await removePrivateClient(socket)
        } catch {
            // Ignore cleanup errors during socket shutdown.
        }
    })

    socket.on('error', async () => {
        try {
            await removePrivateClient(socket)
        } catch {
            // Ignore cleanup errors during socket shutdown.
        }
    })
})

friendsWss.on('connection', async (socket, _request, requestUrl) => {
    const initialAccessToken = normalizeToken(requestUrl.searchParams.get('accessToken') || '')
    const state = {
        id: randomUUID(),
        accessToken: initialAccessToken,
        user: null,
        lastSeenAt: Date.now(),
        isAlive: true,
    }

    friendClients.set(socket, state)

    socket.on('pong', () => {
        const current = friendClients.get(socket)
        if (current) {
            current.isAlive = true
            current.lastSeenAt = Date.now()
        }
    })

    socket.on('message', async (message) => {
        let payload

        try {
            payload = JSON.parse(String(message))
        } catch {
            return
        }

        try {
            const clientState = await ensureFriendClientAuth(socket, payload)
            if (!clientState?.user?.id) {
                sendJson(socket, { type: 'friends_unauthorized' })
                return
            }

            if (payload?.type === 'bootstrap') {
                await sendFriendsBootstrap(socket)
                return
            }

            if (payload?.type === 'search_users') {
                const query = typeof payload.query === 'string' ? payload.query : ''
                const result = await searchFriendUsers(clientState.accessToken, query)
                if (result.unauthorized) {
                    sendJson(socket, { type: 'friends_unauthorized' })
                    return
                }

                sendJson(socket, {
                    type: 'friends_search_results',
                    query,
                    users: Array.isArray(result.payload?.users) ? result.payload.users : [],
                })
                return
            }

            if (payload?.type === 'send_friend_request') {
                const userId = Number(payload.userId || 0)
                if (!userId) {
                    throw new Error('Recipient user is required')
                }

                const result = await sendFriendRequest(clientState.accessToken, userId)
                if (result.unauthorized) {
                    sendJson(socket, { type: 'friends_unauthorized' })
                    return
                }

                const requesterId = Number(result.payload?.requester_id || clientState.user.id)
                const addresseeId = Number(result.payload?.addressee_id || userId)
                await broadcastFriendsBootstrapToUsers([requesterId, addresseeId])
                return
            }

            if (payload?.type === 'respond_friend_request') {
                const friendshipId = Number(payload.friendshipId || 0)
                const action = typeof payload.action === 'string' ? payload.action : ''
                if (!friendshipId || !action) {
                    throw new Error('Friend request action is required')
                }

                const result = await respondToFriendRequest(clientState.accessToken, friendshipId, action)
                if (result.unauthorized) {
                    sendJson(socket, { type: 'friends_unauthorized' })
                    return
                }

                const requesterId = Number(result.payload?.requester_id || 0)
                const addresseeId = Number(result.payload?.addressee_id || clientState.user.id)
                await broadcastFriendsBootstrapToUsers([requesterId, addresseeId])
                return
            }

            if (payload?.type === 'open_conversation') {
                const userId = Number(payload.userId || 0)
                if (!userId) {
                    throw new Error('Conversation user is required')
                }

                await sendConversationHistory(socket, userId)
                const result = await markFriendConversationRead(clientState.accessToken, userId)
                if (!result.unauthorized) {
                    await broadcastFriendsBootstrapToUsers([Number(clientState.user.id)])
                }
                return
            }

            if (payload?.type === 'mark_conversation_read') {
                const userId = Number(payload.userId || 0)
                if (!userId) {
                    throw new Error('Conversation user is required')
                }

                const result = await markFriendConversationRead(clientState.accessToken, userId)
                if (result.unauthorized) {
                    sendJson(socket, { type: 'friends_unauthorized' })
                    return
                }

                await broadcastFriendsBootstrapToUsers([Number(clientState.user.id)])
                return
            }

            if (payload?.type === 'send_direct_message') {
                const recipientId = Number(payload.userId || 0)
                const text = typeof payload.text === 'string' ? payload.text : null
                const share = payload.share && typeof payload.share === 'object' ? payload.share : null
                if (!recipientId) {
                    throw new Error('Message recipient is required')
                }

                const result = await sendFriendDirectMessage(clientState.accessToken, recipientId, text, share)
                if (result.unauthorized) {
                    sendJson(socket, { type: 'friends_unauthorized' })
                    return
                }

                const messagePayload = result.payload?.direct_message
                if (!messagePayload) {
                    throw new Error('Message payload is missing')
                }

                broadcastDirectMessage(messagePayload)
                await broadcastFriendsBootstrapToUsers([
                    Number(messagePayload.sender?.id || clientState.user.id),
                    Number(messagePayload.recipient?.id || recipientId),
                ])
                return
            }

            if (payload?.type === 'refresh_state') {
                await sendFriendsBootstrap(socket)
            }
        } catch (error) {
            sendJson(socket, {
                type: 'friends_error',
                message: error instanceof Error ? error.message : 'Failed to process friends socket event',
            })
        }
    })

    socket.on('close', async () => {
        try {
            await removeFriendClient(socket)
        } catch {
            // Ignore cleanup errors during socket shutdown.
        }
    })

    socket.on('error', async () => {
        try {
            await removeFriendClient(socket)
        } catch {
            // Ignore cleanup errors during socket shutdown.
        }
    })

    if (initialAccessToken) {
        try {
            await ensureFriendClientAuth(socket, { accessToken: initialAccessToken })
            await sendFriendsBootstrap(socket)
        } catch {
            sendJson(socket, { type: 'friends_unauthorized' })
        }
    }
})

const heartbeatTimer = setInterval(() => {
    heartbeatClientMap(publicClients)
    heartbeatClientMap(privateClients)
    heartbeatClientMap(friendClients)
}, heartbeatIntervalMs)

server.listen(wsPort, '0.0.0.0', () => {
    console.log(`[public-radio-ws] listening on 0.0.0.0:${wsPort}`)
})

async function shutdown() {
    clearInterval(heartbeatTimer)

    for (const timer of stationLoops.values()) {
        clearInterval(timer)
    }

    for (const socket of publicClients.keys()) {
        socket.terminate()
    }

    for (const socket of privateClients.keys()) {
        socket.terminate()
    }

    for (const socket of friendClients.keys()) {
        socket.terminate()
    }

    await redis.quit()
    server.close(() => {
        process.exit(0)
    })
}

process.on('SIGINT', () => {
    void shutdown()
})

process.on('SIGTERM', () => {
    void shutdown()
})
