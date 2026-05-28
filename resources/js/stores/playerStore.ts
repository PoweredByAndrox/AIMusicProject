/**
 * Player Store - Pinia store for audio player state management
 * 
 * Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.3, 2.4, 16.3
 */

import { defineStore } from 'pinia'
import { ref, computed, watch } from 'vue'
import { api } from '@/api/client'
import { AUTH_TOKEN_STORAGE_KEY } from '@/constants/auth'
import type { Track, TrackLyricsPayload, TrackPlaybackManifest, TrackPlaybackMode, TrackPlaybackStem } from '@/types'

// Toast notification state (simple reactive approach)
export const toastMessage = ref<{ text: string; type: 'error' | 'warning' | 'info' } | null>(null)

function showToast(text: string, type: 'error' | 'warning' | 'info' = 'warning') {
  toastMessage.value = { text, type }
  setTimeout(() => {
    toastMessage.value = null
  }, 4000)
}

export { showToast }

function createEmptyLyricsPayload(): TrackLyricsPayload {
  return {
    available: false,
    kind: 'none',
    format: null,
    text: null,
    cues: [],
    line_count: 0,
  }
}

function isTrackPlayable(track: Track): boolean {
  return track.analysis_status === 'completed'
}

function buildApiHeaders(): HeadersInit {
  const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
  return token
    ? {
        'Accept': 'application/json',
        'Authorization': `Bearer ${token}`,
      }
    : {
        'Accept': 'application/json',
      }
}

const DISLIKED_TRACK_MAX_RATING = 1

// API helper for recording plays
async function recordPlayStart(trackId: number, source: string): Promise<number | null> {
  try {
    // Map 'manual' to 'queue' since API only accepts: queue, playlist, radio, ai_dj
    const validSource = source === 'manual' ? 'queue' : source
    const response = await fetch(`/api/tracks/${trackId}/play`, {
      method: 'POST',
      headers: {
        ...buildApiHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ source: validSource }),
    })
    if (response.ok) {
      const data = await response.json()
      return data.history?.id || null
    }
  } catch (e) {
    console.error('Failed to record play:', e)
  }
  return null
}

async function recordPlayComplete(historyId: number, durationPlayed: number, completed: boolean): Promise<void> {
  try {
    await fetch(`/api/listening-history/${historyId}/complete`, {
      method: 'PUT',
      headers: {
        ...buildApiHeaders(),
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ duration_played: durationPlayed, completed }),
    })
  } catch (e) {
    console.error('Failed to record play completion:', e)
  }
}

export interface QueueItem {
  id: string
  track: Track
  addedAt: Date
  source: 'manual' | 'playlist' | 'radio' | 'ai_dj'
  originalIndex: number
}

export type RepeatMode = 'off' | 'all' | 'one'

export const usePlayerStore = defineStore('player', () => {
  // State
  const currentTrack = ref<Track | null>(null)
  const queue = ref<QueueItem[]>([])
  const currentIndex = ref<number>(-1)
  const isPlaying = ref(false)
  const currentTime = ref(0)
  const duration = ref(0)
  const volume = ref(1)
  const isMuted = ref(false)
  const shuffleMode = ref(false)
  const repeatMode = ref<RepeatMode>('off')
  const crossfadeEnabled = ref(false)
  const crossfadeDuration = ref(5) // seconds
  const skipDislikedSegmentsEnabled = ref(false)
  const smartRadioEnabled = ref(false)
  const aiDjEnabled = ref(false)
  const playbackManifest = ref<TrackPlaybackManifest | null>(null)
  const selectedPlaybackModeKey = ref('original')
  const stemMixerEnabled = ref(false)
  const stemEnabledByKey = ref<Record<string, boolean>>({})
  const lyricsPayload = ref<TrackLyricsPayload>(createEmptyLyricsPayload())
  const lyricsVisible = ref(false)
  
  // Listening history tracking
  const currentHistoryId = ref<number | null>(null)
  const playStartTime = ref<number>(0)
  
  // Radio queue management
  const radioRefreshInterval = ref<number | null>(null)
  
  // Original order for shuffle restore
  const originalOrder = ref<QueueItem[]>([])
  const trackRatingCache = ref<Record<number, number | null>>({})
  const trackRatingRequests = new Map<number, Promise<number | null>>()

  // Getters
  const hasNext = computed(() => {
    if (repeatMode.value === 'one') return true
    if (repeatMode.value === 'all' && queue.value.length > 0) return true
    return currentIndex.value < queue.value.length - 1
  })

  const hasPrevious = computed(() => {
    if (repeatMode.value === 'one') return true
    if (repeatMode.value === 'all' && queue.value.length > 0) return true
    return currentIndex.value > 0
  })

  const progress = computed(() => {
    if (duration.value === 0) return 0
    return (currentTime.value / duration.value) * 100
  })

  const queueLength = computed(() => queue.value.length)
  const availablePlaybackModes = computed<TrackPlaybackMode[]>(() => playbackManifest.value?.modes ?? [])
  const currentPlaybackMode = computed<TrackPlaybackMode | null>(() => (
    availablePlaybackModes.value.find((mode) => mode.key === selectedPlaybackModeKey.value)
    ?? availablePlaybackModes.value.find((mode) => mode.key === playbackManifest.value?.default_mode_key)
    ?? null
  ))
  const currentPlaybackEndpoint = computed(() => {
    if (!currentTrack.value) return ''
    return currentPlaybackMode.value?.stream_url ?? `/api/tracks/${currentTrack.value.id}/stream`
  })
  const availableStemSources = computed<TrackPlaybackStem[]>(() => playbackManifest.value?.stems ?? [])
  const enabledStemSources = computed<TrackPlaybackStem[]>(() => (
    availableStemSources.value.filter((stem) => stemEnabledByKey.value[stem.key] !== false)
  ))
  const canUseStemMixer = computed(() => availableStemSources.value.length > 0)
  const isStemMixerActive = computed(() => stemMixerEnabled.value && enabledStemSources.value.length > 0)
  const hasLyrics = computed(() => lyricsPayload.value.available)

  function clearPlaybackContext() {
    playbackManifest.value = null
    selectedPlaybackModeKey.value = 'original'
    stemMixerEnabled.value = false
    stemEnabledByKey.value = {}
    lyricsPayload.value = createEmptyLyricsPayload()
    lyricsVisible.value = false
  }

  async function loadPlaybackManifest(trackId: number): Promise<TrackPlaybackManifest | null> {
    try {
      const manifest = await api.getTrackPlaybackOptions(trackId)
      if (currentTrack.value?.id !== trackId) return null

      playbackManifest.value = manifest
      lyricsPayload.value = manifest.lyrics ?? createEmptyLyricsPayload()

      const preferredModeKey = manifest.modes.some((mode) => mode.key === selectedPlaybackModeKey.value)
        ? selectedPlaybackModeKey.value
        : manifest.default_mode_key

      selectedPlaybackModeKey.value = manifest.modes.some((mode) => mode.key === preferredModeKey)
        ? preferredModeKey
        : 'original'

      stemEnabledByKey.value = manifest.stems.reduce<Record<string, boolean>>((acc, stem) => {
        acc[stem.key] = stem.default_enabled !== false
        return acc
      }, {})

      if (manifest.stems.length === 0) {
        stemMixerEnabled.value = false
      }

      if (!manifest.lyrics?.available) {
        lyricsVisible.value = false
      }

      return manifest
    } catch (e) {
      console.error('Failed to load track playback options:', e)
      if (currentTrack.value?.id === trackId) {
        clearPlaybackContext()
      }
      return null
    }
  }

  function setPlaybackMode(modeKey: string) {
    if (!currentTrack.value) return
    if (modeKey === 'original') {
      selectedPlaybackModeKey.value = 'original'
      return
    }

    if (availablePlaybackModes.value.some((mode) => mode.key === modeKey)) {
      selectedPlaybackModeKey.value = modeKey
    }
  }

  function toggleStemSource(stemKey: string) {
    if (!availableStemSources.value.some((stem) => stem.key === stemKey)) return

    stemEnabledByKey.value = {
      ...stemEnabledByKey.value,
      [stemKey]: stemEnabledByKey.value[stemKey] === false,
    }

    if (enabledStemSources.value.length === 0) {
      stemMixerEnabled.value = false
    }
  }

  function setStemMixerEnabled(enabled: boolean) {
    stemMixerEnabled.value = enabled && canUseStemMixer.value && enabledStemSources.value.length > 0
  }

  function setLyricsVisible(visible: boolean) {
    lyricsVisible.value = visible && lyricsPayload.value.available
  }

  watch(() => currentTrack.value?.id ?? null, (nextTrackId, previousTrackId) => {
    if (nextTrackId === previousTrackId) return
    clearPlaybackContext()
  })

  // Actions
  function generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  }

  function rememberTrackRating(trackId: number, rating: number | null | undefined) {
    trackRatingCache.value = {
      ...trackRatingCache.value,
      [trackId]: typeof rating === 'number' ? rating : null,
    }
  }

  async function fetchTrackEffectiveRating(trackId: number, options: { force?: boolean } = {}): Promise<number | null> {
    if (options.force) {
      const nextCache = { ...trackRatingCache.value }
      delete nextCache[trackId]
      trackRatingCache.value = nextCache
      trackRatingRequests.delete(trackId)
    }

    if (Object.prototype.hasOwnProperty.call(trackRatingCache.value, trackId)) {
      return trackRatingCache.value[trackId] ?? null
    }

    const pendingRequest = trackRatingRequests.get(trackId)
    if (pendingRequest) return pendingRequest

    const request = api.getTrackRating(trackId)
      .then((response) => {
        const rating = typeof response.rating === 'number' ? response.rating : null
        rememberTrackRating(trackId, rating)
        return rating
      })
      .catch((e) => {
        console.error('Failed to load track rating for skip filter:', e)
        return null
      })
      .finally(() => {
        trackRatingRequests.delete(trackId)
      })

    trackRatingRequests.set(trackId, request)
    return request
  }

  function isDislikedTrackRating(rating: number | null): boolean {
    return typeof rating === 'number' && rating <= DISLIKED_TRACK_MAX_RATING
  }

  async function shouldSkipDislikedTrack(track: Track, options: { force?: boolean } = {}): Promise<boolean> {
    if (!skipDislikedSegmentsEnabled.value) return false
    const rating = await fetchTrackEffectiveRating(track.id, options)
    return isDislikedTrackRating(rating)
  }

  function showTrackAvailabilityToast(track: Track) {
    const statusMsg = track.analysis_status === 'failed'
      ? 'РћС€РёР±РєР° РѕР±СЂР°Р±РѕС‚РєРё С‚СЂРµРєР°'
      : 'РўСЂРµРє РµС‰С‘ РѕР±СЂР°Р±Р°С‚С‹РІР°РµС‚СЃСЏ'
    showToast(`${statusMsg}: "${track.title}"`, 'warning')
  }

  function ensureTrackPlayable(track: Track): boolean {
    if (isTrackPlayable(track)) return true
    showTrackAvailabilityToast(track)
    return false
  }

  function showDislikedTrackSkipToast(track: Track) {
    showToast(`Skipped 1-star track: "${track.title}"`, 'info')
  }

  function syncOriginalOrderSnapshot() {
    if (!shuffleMode.value) {
      originalOrder.value = [...queue.value]
    }
  }

  function insertTrackIntoQueue(track: Track, position: 'end' | 'next' = 'end', source: QueueItem['source'] = 'manual') {
    const item: QueueItem = {
      id: generateId(),
      track,
      addedAt: new Date(),
      source,
      originalIndex: queue.value.length,
    }

    if (position === 'next' && currentIndex.value >= 0) {
      queue.value.splice(currentIndex.value + 1, 0, item)
      queue.value.forEach((queuedItem, index) => {
        if (index > currentIndex.value + 1) {
          queuedItem.originalIndex = index
        }
      })
    } else {
      item.originalIndex = queue.value.length
      queue.value.push(item)
    }

    if (currentTrack.value === null && queue.value.length === 1) {
      currentIndex.value = 0
      currentTrack.value = track
    }

    syncOriginalOrderSnapshot()
    return item
  }

  async function addToQueueWithPolicy(track: Track, position: 'end' | 'next', source: QueueItem['source']) {
    if (!ensureTrackPlayable(track)) return false
    if (await shouldSkipDislikedTrack(track)) {
      showDislikedTrackSkipToast(track)
      return false
    }

    insertTrackIntoQueue(track, position, source)
    return true
  }

  async function enforceSkipDislikedTracks(trackId?: number): Promise<number> {
    if (!skipDislikedSegmentsEnabled.value) return 0

    const candidateTracks = new Map<number, Track>()
    if (currentTrack.value) {
      candidateTracks.set(currentTrack.value.id, currentTrack.value)
    }
    queue.value.forEach((item) => {
      candidateTracks.set(item.track.id, item.track)
    })

    if (typeof trackId === 'number' && !candidateTracks.has(trackId)) {
      return 0
    }

    const forceTrackIds = typeof trackId === 'number' ? new Set([trackId]) : new Set<number>()
    const ratingResults = await Promise.all(
      [...candidateTracks.values()].map(async (track) => ({
        track,
        shouldSkip: await shouldSkipDislikedTrack(track, { force: forceTrackIds.has(track.id) }),
      })),
    )

    const blockedTrackIds = new Set(
      ratingResults
        .filter((result) => result.shouldSkip)
        .map((result) => result.track.id),
    )

    if (blockedTrackIds.size === 0) return 0

    const removedCount = queue.value.filter((item) => blockedTrackIds.has(item.track.id)).length
    const previousCurrentTrackId = currentTrack.value?.id ?? null
    const previousCurrentIndex = currentIndex.value
    const previousWasPlaying = isPlaying.value
    const currentTrackBlocked = previousCurrentTrackId !== null && blockedTrackIds.has(previousCurrentTrackId)

    queue.value = queue.value.filter((item) => !blockedTrackIds.has(item.track.id))
    originalOrder.value = originalOrder.value.filter((item) => !blockedTrackIds.has(item.track.id))

    if (currentTrackBlocked) {
      if (queue.value.length === 0) {
        currentTrack.value = null
        currentIndex.value = -1
        isPlaying.value = false
        currentTime.value = 0
        duration.value = 0
      } else {
        const replacementIndex = Math.min(Math.max(previousCurrentIndex, 0), queue.value.length - 1)
        currentIndex.value = replacementIndex
        currentTrack.value = queue.value[replacementIndex].track
        currentTime.value = 0
        duration.value = 0

        if (previousWasPlaying) {
          isPlaying.value = true
          recordCurrentPlay()
        }
      }
    } else if (previousCurrentTrackId !== null) {
      const nextIndex = queue.value.findIndex((item) => item.track.id === previousCurrentTrackId)
      currentIndex.value = nextIndex
      currentTrack.value = nextIndex >= 0 ? queue.value[nextIndex].track : null
    }

    if (removedCount > 0) {
      showToast(`Removed ${removedCount} 1-star track${removedCount === 1 ? '' : 's'} from the queue.`, 'info')
    }

    return removedCount
  }

  function play() {
    if (currentTrack.value) {
      isPlaying.value = true
    }
  }

  function pause() {
    isPlaying.value = false
  }

  function stop() {
    isPlaying.value = false
    currentTime.value = 0
  }

  function next() {
    if (queue.value.length === 0) {
      // If radio mode is on, try to get tracks first
      if (smartRadioEnabled.value || aiDjEnabled.value) {
        refreshRadioQueue()
      }
      return
    }

    // Record completion of current track
    if (currentHistoryId.value !== null && currentTrack.value) {
      const durationPlayed = currentTime.value
      const completed = durationPlayed > 0 && duration.value > 0 && (durationPlayed / duration.value) > 0.9
      recordPlayComplete(currentHistoryId.value, durationPlayed, completed)
      currentHistoryId.value = null
    }

    if (repeatMode.value === 'one') {
      // Restart current track
      currentTime.value = 0
      play()
      recordCurrentPlay()
      return
    }

    let nextIndex = currentIndex.value + 1

    if (nextIndex >= queue.value.length) {
      if (repeatMode.value === 'all') {
        nextIndex = 0
      } else {
        // Check if radio mode - try to get more tracks
        if (smartRadioEnabled.value || aiDjEnabled.value) {
          refreshRadioQueue()
          // Check again after refresh
          if (nextIndex < queue.value.length) {
            currentIndex.value = nextIndex
            currentTrack.value = queue.value[nextIndex].track
            currentTime.value = 0
            play()
            recordCurrentPlay()
            return
          }
        }
        stop()
        return
      }
    }

    currentIndex.value = nextIndex
    currentTrack.value = queue.value[nextIndex].track
    currentTime.value = 0
    play()
    recordCurrentPlay()
    
    // Pre-fetch more tracks if radio mode is on and queue is getting low
    if ((smartRadioEnabled.value || aiDjEnabled.value) && queue.value.length - nextIndex <= 3) {
      refreshRadioQueue()
    }
  }

  function previous() {
    if (queue.value.length === 0) return

    // If more than 3 seconds into track, restart it
    if (currentTime.value > 3) {
      currentTime.value = 0
      return
    }

    if (repeatMode.value === 'one') {
      currentTime.value = 0
      play()
      return
    }

    let prevIndex = currentIndex.value - 1

    if (prevIndex < 0) {
      if (repeatMode.value === 'all') {
        prevIndex = queue.value.length - 1
      } else {
        currentTime.value = 0
        return
      }
    }

    currentIndex.value = prevIndex
    currentTrack.value = queue.value[prevIndex].track
    currentTime.value = 0
    play()
  }

  function seek(time: number) {
    currentTime.value = Math.max(0, Math.min(time, duration.value))
  }

  function setVolume(newVolume: number) {
    volume.value = Math.max(0, Math.min(1, newVolume))
    if (volume.value > 0) {
      isMuted.value = false
    }
  }

  function toggleMute() {
    isMuted.value = !isMuted.value
  }

  function addToQueue(track: Track, position: 'end' | 'next' = 'end', source: QueueItem['source'] = 'manual') {
    if (!skipDislikedSegmentsEnabled.value) {
      if (!ensureTrackPlayable(track)) return
      insertTrackIntoQueue(track, position, source)
      return
    }

    void addToQueueWithPolicy(track, position, source)
  }

  function removeFromQueue(itemId: string) {
    const index = queue.value.findIndex(q => q.id === itemId)
    if (index === -1) return

    queue.value.splice(index, 1)

    // Adjust current index if needed
    if (index < currentIndex.value) {
      currentIndex.value--
    } else if (index === currentIndex.value) {
      // Current track was removed
      if (queue.value.length === 0) {
        currentTrack.value = null
        currentIndex.value = -1
        stop()
      } else if (currentIndex.value >= queue.value.length) {
        currentIndex.value = queue.value.length - 1
        currentTrack.value = queue.value[currentIndex.value].track
      } else {
        currentTrack.value = queue.value[currentIndex.value].track
      }
    }

    // Update original order
    if (!shuffleMode.value) {
      originalOrder.value = [...queue.value]
    }
  }

  function reorderQueue(fromIndex: number, toIndex: number) {
    if (fromIndex === toIndex) return
    if (fromIndex < 0 || fromIndex >= queue.value.length) return
    if (toIndex < 0 || toIndex >= queue.value.length) return

    const [item] = queue.value.splice(fromIndex, 1)
    queue.value.splice(toIndex, 0, item)

    // Adjust current index
    if (fromIndex === currentIndex.value) {
      currentIndex.value = toIndex
    } else if (fromIndex < currentIndex.value && toIndex >= currentIndex.value) {
      currentIndex.value--
    } else if (fromIndex > currentIndex.value && toIndex <= currentIndex.value) {
      currentIndex.value++
    }

    // Update original order if not shuffled
    if (!shuffleMode.value) {
      originalOrder.value = [...queue.value]
    }
  }

  function clearQueue() {
    const current = currentIndex.value >= 0 ? queue.value[currentIndex.value] : null
    
    if (current && isPlaying.value) {
      // Keep only current track
      queue.value = [current]
      currentIndex.value = 0
      originalOrder.value = [current]
    } else {
      queue.value = []
      originalOrder.value = []
      currentIndex.value = -1
      currentTrack.value = null
      stop()
    }
  }

  function toggleShuffle() {
    if (shuffleMode.value) {
      // Restore original order
      const currentItem = currentIndex.value >= 0 ? queue.value[currentIndex.value] : null
      queue.value = [...originalOrder.value]
      
      // Find current track in restored order
      if (currentItem) {
        currentIndex.value = queue.value.findIndex(q => q.id === currentItem.id)
      }
    } else {
      // Save original order and shuffle
      originalOrder.value = [...queue.value]
      
      const currentItem = currentIndex.value >= 0 ? queue.value[currentIndex.value] : null
      
      // Fisher-Yates shuffle, keeping current track at current position
      const toShuffle = queue.value.filter((_, i) => i !== currentIndex.value)
      for (let i = toShuffle.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[toShuffle[i], toShuffle[j]] = [toShuffle[j], toShuffle[i]]
      }
      
      // Rebuild queue with current track at position 0
      if (currentItem) {
        queue.value = [currentItem, ...toShuffle]
        currentIndex.value = 0
      } else {
        queue.value = toShuffle
      }
    }
    
    shuffleMode.value = !shuffleMode.value
  }

  function setRepeatMode(mode: RepeatMode) {
    repeatMode.value = mode
  }

  function cycleRepeatMode() {
    const modes: RepeatMode[] = ['off', 'all', 'one']
    const currentModeIndex = modes.indexOf(repeatMode.value)
    repeatMode.value = modes[(currentModeIndex + 1) % modes.length]
  }

  function setCrossfade(enabled: boolean, durationSec?: number) {
    crossfadeEnabled.value = enabled
    if (durationSec !== undefined) {
      crossfadeDuration.value = Math.max(0, Math.min(12, durationSec))
    }
  }

  function setSkipDislikedSegments(enabled: boolean) {
    skipDislikedSegmentsEnabled.value = enabled
    if (enabled) {
      void enforceSkipDislikedTracks()
    }
  }

  function setSmartRadio(enabled: boolean) {
    smartRadioEnabled.value = enabled
    if (enabled) {
      startRadioMode()
      // Immediately fill queue if it's empty or low
      if (queue.value.length - currentIndex.value <= 3) {
        refreshRadioQueue()
      }
    } else if (!aiDjEnabled.value) {
      stopRadioMode()
    }
  }

  function setAiDj(enabled: boolean) {
    aiDjEnabled.value = enabled
    if (enabled) {
      startRadioMode()
      // Immediately fill queue if it's empty or low
      if (queue.value.length - currentIndex.value <= 3) {
        refreshRadioQueue()
      }
    } else if (!smartRadioEnabled.value) {
      stopRadioMode()
    }
  }

  function playTrack(track: Track, clearExisting = false) {
    if (!skipDislikedSegmentsEnabled.value) {
      if (!ensureTrackPlayable(track)) return

      if (clearExisting) {
        queue.value = []
        originalOrder.value = []
      }

      insertTrackIntoQueue(track, 'end', 'manual')

      const index = queue.value.findIndex(q => q.track.id === track.id)
      if (index >= 0) {
        currentIndex.value = index
        currentTrack.value = track
        currentTime.value = 0
        play()
        recordCurrentPlay()
      }
      return
    }

    void (async () => {
      if (!ensureTrackPlayable(track)) return
      if (await shouldSkipDislikedTrack(track)) {
        showDislikedTrackSkipToast(track)
        return
      }

      if (clearExisting) {
        queue.value = []
        originalOrder.value = []
      }

      insertTrackIntoQueue(track, 'end', 'manual')

      const index = queue.value.findIndex(q => q.track.id === track.id)
      if (index >= 0) {
        currentIndex.value = index
        currentTrack.value = track
        currentTime.value = 0
        play()
        recordCurrentPlay()
      }
    })()
  }

  function playPlaylist(tracks: Track[], startIndex = 0) {
    if (!skipDislikedSegmentsEnabled.value) {
      const playableTracks = tracks.filter(t => isTrackPlayable(t))
      const skippedCount = tracks.length - playableTracks.length

      if (skippedCount > 0) {
        showToast(`Пропущено ${skippedCount} треков (ещё обрабатываются)`, 'info')
      }

      if (playableTracks.length === 0) {
        showToast('Нет готовых треков для воспроизведения', 'warning')
        return
      }

      queue.value = []
      originalOrder.value = []

      playableTracks.forEach((track) => {
        insertTrackIntoQueue(track, 'end', 'playlist')
      })

      if (queue.value.length > 0) {
        currentIndex.value = Math.min(startIndex, queue.value.length - 1)
        currentTrack.value = queue.value[currentIndex.value].track
        currentTime.value = 0
        play()
      }
      return
    }

    void (async () => {
      const playableCandidates = tracks
        .map((track, index) => ({ track, originalIndex: index }))
        .filter(({ track }) => isTrackPlayable(track))
      const notReadyCount = tracks.length - playableCandidates.length

      const eligibility = await Promise.all(
        playableCandidates.map(async ({ track, originalIndex }) => ({
          track,
          originalIndex,
          shouldSkip: await shouldSkipDislikedTrack(track),
        })),
      )

      const playableTracks = eligibility.filter(({ shouldSkip }) => !shouldSkip)
      const skippedDislikedCount = eligibility.length - playableTracks.length

      if (notReadyCount > 0) {
        showToast(`Skipped ${notReadyCount} track${notReadyCount === 1 ? '' : 's'} that are not ready yet.`, 'info')
      }

      if (skippedDislikedCount > 0) {
        showToast(`Skipped ${skippedDislikedCount} 1-star track${skippedDislikedCount === 1 ? '' : 's'}.`, 'info')
      }

      if (playableTracks.length === 0) {
        showToast('No playable tracks are available for playback.', 'warning')
        return
      }

      queue.value = []
      originalOrder.value = []

      playableTracks.forEach(({ track }) => {
        insertTrackIntoQueue(track, 'end', 'playlist')
      })

      const nextIndex = playableTracks.findIndex(({ originalIndex }) => originalIndex >= startIndex)
      currentIndex.value = nextIndex >= 0 ? nextIndex : playableTracks.length - 1
      currentTrack.value = queue.value[currentIndex.value]?.track ?? null
      currentTime.value = 0
      play()
    })()
  }

  // Update time (called by audio engine)
  function updateTime(time: number) {
    currentTime.value = time
  }

  function updateDuration(dur: number) {
    duration.value = dur
  }

  function onTrackEnded() {
    // Record completion of previous track
    if (currentHistoryId.value !== null) {
      const durationPlayed = currentTime.value
      const completed = durationPlayed > 0 && duration.value > 0 && (durationPlayed / duration.value) > 0.9
      recordPlayComplete(currentHistoryId.value, durationPlayed, completed)
      currentHistoryId.value = null
    }
    next()
  }

  // Record play when track starts
  async function recordCurrentPlay() {
    if (!currentTrack.value) return
    
    // Complete previous play if any
    if (currentHistoryId.value !== null) {
      const durationPlayed = currentTime.value - playStartTime.value
      recordPlayComplete(currentHistoryId.value, durationPlayed, false)
    }
    
    // Get source from current queue item
    const currentItem = currentIndex.value >= 0 ? queue.value[currentIndex.value] : null
    const source = currentItem?.source || 'manual'
    
    // Record new play
    playStartTime.value = 0
    currentHistoryId.value = await recordPlayStart(currentTrack.value.id, source)
  }

  // Radio queue management
  async function refreshRadioQueue(replaceUpcoming = false) {
    if (!smartRadioEnabled.value && !aiDjEnabled.value) return
    
    try {
      if (replaceUpcoming) {
        queue.value = queue.value.filter((item, index) => (
          index <= currentIndex.value || (item.source !== 'radio' && item.source !== 'ai_dj')
        ))

        if (currentTrack.value) {
          currentIndex.value = queue.value.findIndex((item) => item.track.id === currentTrack.value?.id)
        }

        if (!shuffleMode.value) {
          originalOrder.value = [...queue.value]
        }
      }

      // Get recommendations from API
      const endpoint = aiDjEnabled.value && currentTrack.value 
        ? `/api/dj/compatible/${currentTrack.value.id}`
        : '/api/radio/recommendations'
      
      const response = await fetch(endpoint, {
        headers: buildApiHeaders(),
      })
      if (!response.ok) return
      
      const data = await response.json()
      
      // Parse response - AI DJ returns { compatible_tracks: [{track: Track, ...}] }
      // Radio returns { tracks: Track[] }
      let recommendedTracks: Track[] = []
      
      if (data.compatible_tracks && Array.isArray(data.compatible_tracks)) {
        // AI DJ format - extract track from each item
        recommendedTracks = data.compatible_tracks.map((item: any) => item.track).filter(Boolean)
      } else if (data.tracks && Array.isArray(data.tracks)) {
        // Radio format
        recommendedTracks = data.tracks
      }
      
      if (recommendedTracks.length === 0) return
      
      // Get IDs of tracks already in queue
      const queueTrackIds = new Set(queue.value.map(q => q.track.id))
      
      // Also exclude current track
      if (currentTrack.value) {
        queueTrackIds.add(currentTrack.value.id)
      }
      
      // Filter out tracks already in queue
      const candidateTracks = recommendedTracks.filter(t => !queueTrackIds.has(t.id) && isTrackPlayable(t))
      let newTracks = candidateTracks

      if (skipDislikedSegmentsEnabled.value && candidateTracks.length > 0) {
        const eligibility = await Promise.all(
          candidateTracks.map(async (track) => ({
            track,
            shouldSkip: await shouldSkipDislikedTrack(track),
          })),
        )
        newTracks = eligibility
          .filter(({ shouldSkip }) => !shouldSkip)
          .map(({ track }) => track)
      }
      
      // Calculate how many tracks to add (keep queue at ~5 tracks ahead)
      const tracksAhead = queue.value.length - currentIndex.value - 1
      const targetAhead = 5
      const tracksToAdd = replaceUpcoming
        ? Math.min(newTracks.length, targetAhead)
        : Math.min(newTracks.length, Math.max(0, targetAhead - tracksAhead))
      
      if (tracksToAdd <= 0) return
      
      // Add new tracks to queue
      const source = aiDjEnabled.value ? 'ai_dj' : 'radio'
      for (let i = 0; i < tracksToAdd; i++) {
        insertTrackIntoQueue(newTracks[i], 'end', source)
      }
      
      console.log(`🎵 Radio: Added ${tracksToAdd} tracks to queue (${tracksAhead} were ahead, target ${targetAhead})`)
    } catch (e) {
      console.error('Failed to refresh radio queue:', e)
    }
  }

  // Start radio mode with auto-refresh
  function startRadioMode() {
    if (radioRefreshInterval.value) {
      clearInterval(radioRefreshInterval.value)
    }
    
    // Initial refresh
    refreshRadioQueue()
    
    // Refresh every 30 seconds
    radioRefreshInterval.value = window.setInterval(() => {
      refreshRadioQueue()
    }, 30000)
  }

  // Stop radio mode
  function stopRadioMode() {
    if (radioRefreshInterval.value) {
      clearInterval(radioRefreshInterval.value)
      radioRefreshInterval.value = null
    }
  }

  function clearCurrentTrack() {
    currentTrack.value = null
    currentIndex.value = -1
    queue.value = []
    originalOrder.value = []
    isPlaying.value = false
    currentTime.value = 0
    duration.value = 0
  }

  return {
    // State
    currentTrack,
    queue,
    currentIndex,
    isPlaying,
    currentTime,
    duration,
    volume,
    isMuted,
    shuffleMode,
    repeatMode,
    crossfadeEnabled,
    crossfadeDuration,
    skipDislikedSegmentsEnabled,
    smartRadioEnabled,
    aiDjEnabled,
    playbackManifest,
    selectedPlaybackModeKey,
    stemMixerEnabled,
    stemEnabledByKey,
    lyricsPayload,
    lyricsVisible,
    
    // Getters
    hasNext,
    hasPrevious,
    progress,
    queueLength,
    availablePlaybackModes,
    currentPlaybackMode,
    currentPlaybackEndpoint,
    availableStemSources,
    enabledStemSources,
    canUseStemMixer,
    isStemMixerActive,
    hasLyrics,
    
    // Actions
    play,
    pause,
    stop,
    next,
    previous,
    seek,
    setVolume,
    toggleMute,
    addToQueue,
    removeFromQueue,
    reorderQueue,
    clearQueue,
    toggleShuffle,
    setRepeatMode,
    cycleRepeatMode,
    setCrossfade,
    setSkipDislikedSegments,
    setSmartRadio,
    setAiDj,
    playTrack,
    playPlaylist,
    rememberTrackRating,
    fetchTrackEffectiveRating,
    enforceSkipDislikedTracks,
    updateTime,
    updateDuration,
    onTrackEnded,
    clearCurrentTrack,
    refreshRadioQueue,
    startRadioMode,
    stopRadioMode,
    clearPlaybackContext,
    loadPlaybackManifest,
    setPlaybackMode,
    toggleStemSource,
    setStemMixerEnabled,
    setLyricsVisible,
  }
}, {
  persist: {
    key: 'music-player',
    paths: ['volume', 'isMuted', 'shuffleMode', 'repeatMode', 'crossfadeEnabled', 'crossfadeDuration', 'skipDislikedSegmentsEnabled', 'currentTrack', 'queue', 'currentIndex'],
  },
})
