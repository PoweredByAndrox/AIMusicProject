<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted, nextTick } from 'vue'
import { usePlayerStore, showToast } from '@/stores/playerStore'
import ArtistCreditsInline from './ArtistCreditsInline.vue'
import RatingWidget from './RatingWidget.vue'
import QueuePanel from './QueuePanel.vue'
import { api } from '@/api/client'

interface RatedSegmentSnapshot {
  id: number
  start: number
  end: number
  rating?: number | null
}

interface TrackFeedbackUpdatedDetail {
  trackId?: number
}

const AUTO_SKIP_MAX_RATING = 1
const AUTO_SKIP_OVERLAP_SEC = 1.2
const AUTO_SKIP_BLEND_SEC = 0.75
const AUTO_SKIP_OUTGOING_GAIN_RATIO = 0.76
const AUTO_SKIP_INCOMING_GAIN_RATIO = 0.14
const AUTO_SKIP_SEEK_PADDING_SEC = 0.04
const AUTO_SKIP_RESET_PADDING_SEC = 0.2

const playerStore = usePlayerStore()
const trackRating = ref<number | null>(null)
const isLoadingRating = ref(false)
const currentTrackSegments = ref<RatedSegmentSnapshot[]>([])
const isSelectingFragment = ref(false)
const fragmentStart = ref<number | null>(null)
const fragmentEnd = ref<number | null>(null)
const fragmentComment = ref('')
const showFragmentDialog = ref(false)
const isAutoSkippingSegment = ref(false)
const autoSkipCooldownUntil = ref(0)
const lastAutoSkippedRunKey = ref<string | null>(null)
const sortedTrackSegments = computed(() => (
  [...currentTrackSegments.value].sort((left, right) => left.start - right.start)
))

watch(() => playerStore.currentTrack, async (track) => {
  const trackId = track?.id ?? null
  currentTrackSegments.value = []
  autoSkipCooldownUntil.value = 0
  lastAutoSkippedRunKey.value = null
  isAutoSkippingSegment.value = false

  if (!trackId) {
    trackRating.value = null
    cancelFragmentSelection()
    return
  }

  try {
    isLoadingRating.value = true
    const [ratingResult, feedbackResult] = await Promise.allSettled([
      api.getTrackRating(trackId),
      api.getTrackFeedbackSnapshot(trackId),
    ])

    if (ratingResult.status === 'fulfilled') {
      trackRating.value = ratingResult.value.rating
      playerStore.rememberTrackRating(trackId, trackRating.value)
    } else {
      trackRating.value = null
    }

    currentTrackSegments.value = feedbackResult.status === 'fulfilled' && Array.isArray(feedbackResult.value.segments)
      ? feedbackResult.value.segments.map((segment) => ({
          id: segment.id,
          start: segment.start,
          end: segment.end,
          rating: segment.rating,
        }))
      : []
  } finally {
    isLoadingRating.value = false
  }

  cancelFragmentSelection()
}, { immediate: true })

const onRatingChange = async (rating: number | null) => {
  const trackId = playerStore.currentTrack?.id
  if (!trackId) return

  try {
    if (rating === null) {
      const response = await api.deleteTrackRating(trackId)
      trackRating.value = typeof response.feedback?.track_rating === 'number'
        ? response.feedback.track_rating
        : (typeof response.effective_rating === 'number' ? response.effective_rating : null)
    } else {
      const response = await api.setTrackRating(trackId, rating)
      trackRating.value = typeof response.feedback?.track_rating === 'number'
        ? response.feedback.track_rating
        : (typeof response.effective_rating === 'number' ? response.effective_rating : rating)
    }
    playerStore.rememberTrackRating(trackId, trackRating.value)
    void playerStore.enforceSkipDislikedTracks(trackId)
    showToast(rating ? `Rated ${rating} stars` : 'Rating removed', 'info')
  } catch (e) {
    console.error('Error saving rating:', e)
    showToast('Failed to save rating', 'error')
  }
}

function startFragmentSelection() {
  if (!playerStore.currentTrack) return
  isSelectingFragment.value = true
  fragmentStart.value = playerStore.currentTime
  fragmentEnd.value = null
  showToast('Choose the clip end point, then press Clip again.', 'info')
}

function setFragmentEnd() {
  if (fragmentStart.value === null) return
  fragmentEnd.value = playerStore.currentTime
  if (fragmentEnd.value < fragmentStart.value) {
    const temp = fragmentStart.value
    fragmentStart.value = fragmentEnd.value
    fragmentEnd.value = temp
  }
  showFragmentDialog.value = true
}

function cancelFragmentSelection() {
  isSelectingFragment.value = false
  fragmentStart.value = null
  fragmentEnd.value = null
  fragmentComment.value = ''
  showFragmentDialog.value = false
}

async function submitFragment(isLiked: boolean) {
  if (!playerStore.currentTrack || fragmentStart.value === null || fragmentEnd.value === null) return

  try {
    const response = await fetch(`/api/tracks/${playerStore.currentTrack.id}/fragments/rate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start_time: fragmentStart.value,
        end_time: fragmentEnd.value,
        is_liked: isLiked,
      }),
    })
    if (!response.ok) throw new Error('Failed to create fragment')
    const result = await response.json()

    if (fragmentComment.value.trim() && result.fragment?.id) {
      await fetch(`/api/fragments/${result.fragment.id}/comment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment: fragmentComment.value.trim() }),
      })
    }

    showToast(isLiked ? 'Clip saved as liked.' : 'Clip saved as disliked.', 'info')
    window.dispatchEvent(new CustomEvent('track-feedback-updated', {
      detail: { trackId: playerStore.currentTrack.id },
    }))
    cancelFragmentSelection()
  } catch (e) {
    console.error('Error creating fragment:', e)
    showToast('Failed to save clip feedback.', 'error')
  }
}

const audioRef = ref<HTMLAudioElement | null>(null)
const shadowAudioRef = ref<HTMLAudioElement | null>(null)
const audioContextRef = ref<AudioContext | null>(null)
const mediaSourceRef = ref<MediaElementAudioSourceNode | null>(null)
const gainNodeRef = ref<GainNode | null>(null)
const connectedAudioRef = ref<HTMLAudioElement | null>(null)
const shadowMediaSourceRef = ref<MediaElementAudioSourceNode | null>(null)
const shadowGainNodeRef = ref<GainNode | null>(null)
const shadowConnectedAudioRef = ref<HTMLAudioElement | null>(null)
const activeAudioSlot = ref<'main' | 'shadow'>('main')
const isSeeking = ref(false)
const seekValue = ref(0)
const stemAudioElements = new Map<string, HTMLAudioElement>()
const stemMediaSourceNodes = new Map<string, MediaElementAudioSourceNode>()
const stemGainNodes = new Map<string, GainNode>()
const stemConnectedAudioElements = new Map<string, HTMLAudioElement>()

function resolveBackendMediaUrl(path: string) {
  if (!path) return ''
  if (/^https?:\/\//i.test(path)) return path
  const origin = window.location.origin
  const backendUrl = origin.includes(':5173') ? origin.replace(':5173', ':8000') : origin
  return `${backendUrl}${path.startsWith('/') ? path : `/${path}`}`
}

const audioUrl = computed(() => {
  return resolveBackendMediaUrl(playerStore.currentPlaybackEndpoint)
})
const lyricsCueIndex = computed(() => {
  const cues = playerStore.lyricsPayload.cues
  if (!cues.length) return -1
  const currentMs = Math.round(playerStore.currentTime * 1000)
  return cues.findIndex((cue, index) => {
    const cueEnd = cue.end_ms ?? cues[index + 1]?.start_ms ?? Number.MAX_SAFE_INTEGER
    return currentMs >= cue.start_ms && currentMs < cueEnd
  })
})
const activeLyricsCue = computed(() => (
  lyricsCueIndex.value >= 0 ? playerStore.lyricsPayload.cues[lyricsCueIndex.value] : null
))
const previousLyricsCue = computed(() => (
  lyricsCueIndex.value > 0 ? playerStore.lyricsPayload.cues[lyricsCueIndex.value - 1] : null
))
const nextLyricsCue = computed(() => {
  if (lyricsCueIndex.value < 0) {
    return playerStore.lyricsPayload.cues[0] ?? null
  }
  return playerStore.lyricsPayload.cues[lyricsCueIndex.value + 1] ?? null
})

const formattedCurrentTime = computed(() => formatTime(playerStore.currentTime))
const formattedDuration = computed(() => formatTime(playerStore.duration))
const hasTrack = computed(() => Boolean(playerStore.currentTrack))
const trackMonogram = computed(() => {
  const title = playerStore.currentTrack?.title?.trim()
  if (!title) return 'MA'
  return title
    .split(/\s+/)
    .slice(0, 2)
    .map((word) => word[0]?.toUpperCase() ?? '')
    .join('')
    .slice(0, 2) || 'MA'
})
const trackSubtitle = computed(() => playerStore.currentTrack?.artist || 'Choose something from the library to start listening.')
const trackFacts = computed(() => {
  const track = playerStore.currentTrack
  if (!track) return []
  const facts: string[] = [track.source === 'youtube' ? 'YouTube' : 'Upload']
  if (track.genre) facts.push(track.genre)
  if (track.bpm) facts.push(`${Math.round(track.bpm)} BPM`)
  return facts.slice(0, 3)
})
const trackStatusLabel = computed(() => {
  const status = playerStore.currentTrack?.analysis_status
  if (!status) return 'Idle'
  switch (status) {
    case 'completed': return 'Ready'
    case 'processing': return 'Processing'
    case 'analyzing': return 'Analyzing'
    case 'failed': return 'Failed'
    default: return 'Queued'
  }
})
const fragmentSelectionLabel = computed(() => {
  if (!isSelectingFragment.value || fragmentStart.value === null) return ''
  return `Clip start ${formatTime(fragmentStart.value)}`
})
const volumeLabel = computed(() => {
  if (playerStore.isMuted || playerStore.volume === 0) return 'Muted'
  return `${Math.round(playerStore.volume * 100)}%`
})
const trackLoudnessGainDb = computed(() => {
  const gainDb = playerStore.currentTrack?.loudness_gain_db
  return Number.isFinite(gainDb) ? Number(gainDb) : 0
})
const effectiveVolume = computed(() => {
  if (playerStore.isMuted) return 0
  return playerStore.volume * dbToLinearGain(trackLoudnessGainDb.value)
})
const repeatIcon = computed(() => {
  switch (playerStore.repeatMode) {
    case 'one': return 'Track'
    case 'all': return 'Queue'
    default: return 'Loop'
  }
})

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

function dbToLinearGain(db: number): number {
  if (!Number.isFinite(db)) return 1
  return Math.pow(10, db / 20)
}

function resumeAudioContext() {
  if (audioContextRef.value?.state === 'suspended') {
    audioContextRef.value.resume().catch((e) => {
      console.error('Failed to resume audio context:', e)
    })
  }
}

function syncAudioGain() {
  const gain = Math.max(0, effectiveVolume.value)
  const primaryGainNode = getPrimaryGainNode()
  const secondaryGainNode = getSecondaryGainNode()
  if (!isAutoSkippingSegment.value && primaryGainNode) primaryGainNode.gain.value = playerStore.isStemMixerActive ? 0.0001 : gain
  if (!isAutoSkippingSegment.value && secondaryGainNode) secondaryGainNode.gain.value = 0
  if (audioRef.value) {
    audioRef.value.volume = gainNodeRef.value
      ? 1
      : (playerStore.isStemMixerActive ? 0 : (activeAudioSlot.value === 'main' ? Math.min(1, gain) : 0))
  }
  if (shadowAudioRef.value) {
    shadowAudioRef.value.volume = shadowGainNodeRef.value
      ? 1
      : (playerStore.isStemMixerActive ? 0 : (activeAudioSlot.value === 'shadow' ? Math.min(1, gain) : 0))
  }

  stemAudioElements.forEach((audio, stemKey) => {
    const stemEnabled = playerStore.stemEnabledByKey[stemKey] !== false
    const nextGain = playerStore.isStemMixerActive && stemEnabled ? gain : 0
    const gainNode = stemGainNodes.get(stemKey)

    if (gainNode) {
      gainNode.gain.value = nextGain
    } else {
      audio.volume = playerStore.isStemMixerActive && stemEnabled ? Math.min(1, gain) : 0
    }
  })
}

function getPrimaryAudio() {
  return activeAudioSlot.value === 'main' ? audioRef.value : shadowAudioRef.value
}

function getSecondaryAudio() {
  return activeAudioSlot.value === 'main' ? shadowAudioRef.value : audioRef.value
}

function getPrimaryGainNode() {
  return activeAudioSlot.value === 'main' ? gainNodeRef.value : shadowGainNodeRef.value
}

function getSecondaryGainNode() {
  return activeAudioSlot.value === 'main' ? shadowGainNodeRef.value : gainNodeRef.value
}

function isPrimaryAudioElement(audio: HTMLAudioElement | null | undefined) {
  return Boolean(audio) && audio === getPrimaryAudio()
}

function getStemLeaderAudio() {
  const leaderStem = playerStore.availableStemSources[0]
  if (!leaderStem) return null
  return stemAudioElements.get(leaderStem.key) ?? null
}

function isStemLeaderAudio(audio: HTMLAudioElement | null | undefined) {
  return Boolean(audio) && audio === getStemLeaderAudio()
}

function isTransportLeaderAudio(audio: HTMLAudioElement | null | undefined) {
  if (playerStore.isStemMixerActive) {
    return isStemLeaderAudio(audio)
  }

  return isPrimaryAudioElement(audio)
}

function attachStemAudioListeners(audio: HTMLAudioElement) {
  audio.addEventListener('loadedmetadata', onLoadedMetadata as EventListener)
  audio.addEventListener('canplaythrough', onCanPlay as EventListener)
  audio.addEventListener('timeupdate', onTimeUpdate as EventListener)
  audio.addEventListener('ended', onEnded as EventListener)
  audio.addEventListener('error', onAudioError as EventListener)
}

function detachStemAudioListeners(audio: HTMLAudioElement) {
  audio.removeEventListener('loadedmetadata', onLoadedMetadata as EventListener)
  audio.removeEventListener('canplaythrough', onCanPlay as EventListener)
  audio.removeEventListener('timeupdate', onTimeUpdate as EventListener)
  audio.removeEventListener('ended', onEnded as EventListener)
  audio.removeEventListener('error', onAudioError as EventListener)
}

function ensureAudioGraph() {
  if (!audioRef.value || typeof window === 'undefined') return
  const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AudioContextCtor) {
    syncAudioGain()
    return
  }

  if (!audioContextRef.value) audioContextRef.value = new AudioContextCtor()
  if (connectedAudioRef.value === audioRef.value && gainNodeRef.value) {
    syncAudioGain()
    return
  }

  mediaSourceRef.value?.disconnect()
  gainNodeRef.value?.disconnect()
  mediaSourceRef.value = audioContextRef.value.createMediaElementSource(audioRef.value)
  gainNodeRef.value = audioContextRef.value.createGain()
  mediaSourceRef.value.connect(gainNodeRef.value)
  gainNodeRef.value.connect(audioContextRef.value.destination)
  connectedAudioRef.value = audioRef.value
  audioRef.value.volume = 1
  syncAudioGain()
}

function ensureShadowAudioGraph() {
  if (!shadowAudioRef.value || typeof window === 'undefined') return
  const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AudioContextCtor) {
    syncAudioGain()
    return
  }

  if (!audioContextRef.value) audioContextRef.value = new AudioContextCtor()
  if (shadowConnectedAudioRef.value === shadowAudioRef.value && shadowGainNodeRef.value) {
    syncAudioGain()
    return
  }

  shadowMediaSourceRef.value?.disconnect()
  shadowGainNodeRef.value?.disconnect()
  shadowMediaSourceRef.value = audioContextRef.value.createMediaElementSource(shadowAudioRef.value)
  shadowGainNodeRef.value = audioContextRef.value.createGain()
  shadowMediaSourceRef.value.connect(shadowGainNodeRef.value)
  shadowGainNodeRef.value.connect(audioContextRef.value.destination)
  shadowConnectedAudioRef.value = shadowAudioRef.value
  shadowAudioRef.value.volume = 1
  syncAudioGain()
}

function ensureStemAudioGraph(stemKey: string, audio: HTMLAudioElement) {
  if (typeof window === 'undefined') return
  const AudioContextCtor = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
  if (!AudioContextCtor) {
    syncAudioGain()
    return
  }

  if (!audioContextRef.value) audioContextRef.value = new AudioContextCtor()
  if (stemConnectedAudioElements.get(stemKey) === audio && stemGainNodes.get(stemKey)) {
    syncAudioGain()
    return
  }

  stemMediaSourceNodes.get(stemKey)?.disconnect()
  stemGainNodes.get(stemKey)?.disconnect()

  const mediaSource = audioContextRef.value.createMediaElementSource(audio)
  const gainNode = audioContextRef.value.createGain()
  mediaSource.connect(gainNode)
  gainNode.connect(audioContextRef.value.destination)

  stemMediaSourceNodes.set(stemKey, mediaSource)
  stemGainNodes.set(stemKey, gainNode)
  stemConnectedAudioElements.set(stemKey, audio)
  audio.volume = 1
  syncAudioGain()
}

function destroyStemAudio(stemKey: string) {
  const audio = stemAudioElements.get(stemKey)
  if (audio) {
    audio.pause()
    detachStemAudioListeners(audio)
    audio.removeAttribute('src')
    audio.load()
  }

  stemMediaSourceNodes.get(stemKey)?.disconnect()
  stemGainNodes.get(stemKey)?.disconnect()
  stemMediaSourceNodes.delete(stemKey)
  stemGainNodes.delete(stemKey)
  stemConnectedAudioElements.delete(stemKey)
  stemAudioElements.delete(stemKey)
}

function syncStemMixerSources() {
  const requiredKeys = new Set(playerStore.availableStemSources.map((stem) => stem.key))

  Array.from(stemAudioElements.keys()).forEach((stemKey) => {
    if (!requiredKeys.has(stemKey)) {
      destroyStemAudio(stemKey)
    }
  })

  playerStore.availableStemSources.forEach((stem) => {
    const streamUrl = resolveBackendMediaUrl(stem.stream_url)
    let audio = stemAudioElements.get(stem.key)

    if (!audio) {
      audio = new Audio()
      audio.preload = 'auto'
      attachStemAudioListeners(audio)
      stemAudioElements.set(stem.key, audio)
    }

    if (audio.src !== streamUrl) {
      audio.pause()
      audio.src = streamUrl
      audio.load()
    }

    ensureStemAudioGraph(stem.key, audio)
  })
}

function pauseStemMixer() {
  stemAudioElements.forEach((audio) => {
    audio.pause()
  })
}

async function syncStemMixerPositions(targetTime: number) {
  const syncTasks: Array<Promise<void>> = []

  stemAudioElements.forEach((audio) => {
    if (Math.abs(audio.currentTime - targetTime) < 0.08) {
      return
    }

    audio.currentTime = targetTime
    syncTasks.push(waitForSeeked(audio, targetTime))
  })

  if (syncTasks.length > 0) {
    await Promise.allSettled(syncTasks)
  }
}

async function playStemMixer(startTime: number) {
  resumeAudioContext()
  syncStemMixerSources()
  await Promise.allSettled(Array.from(stemAudioElements.values()).map((audio) => waitForCanPlay(audio)))
  await syncStemMixerPositions(startTime)

  const playTasks = Array.from(stemAudioElements.values()).map((audio) => (
    audio.play().catch((e) => {
      throw e instanceof Error ? e : new Error('Failed to start stem audio')
    })
  ))

  await Promise.allSettled(playTasks)
}

function syncStemFollowerDrift() {
  const leaderAudio = getStemLeaderAudio()
  if (!leaderAudio) return

  stemAudioElements.forEach((audio, stemKey) => {
    if (audio === leaderAudio) {
      return
    }

    if (audio.paused || leaderAudio.paused) {
      return
    }

    if (Math.abs(audio.currentTime - leaderAudio.currentTime) > 0.18) {
      audio.currentTime = leaderAudio.currentTime
    }

    const gainNode = stemGainNodes.get(stemKey)
    const stemEnabled = playerStore.stemEnabledByKey[stemKey] !== false
    const nextGain = playerStore.isStemMixerActive && stemEnabled ? effectiveVolume.value : 0

    if (gainNode) {
      gainNode.gain.value = nextGain
    } else {
      audio.volume = playerStore.isStemMixerActive && stemEnabled ? Math.min(1, effectiveVolume.value) : 0
    }
  })
}

const shouldPlayOnLoad = ref(false)

function onLoadedMetadata(event: Event) {
  const audio = event.target as HTMLAudioElement | null
  if (!audio || !isTransportLeaderAudio(audio)) return
  playerStore.updateDuration(audio.duration)
}

function onCanPlay(event: Event) {
  const audio = event.target as HTMLAudioElement | null
  resumeAudioContext()
  syncAudioGain()
  if (!isTransportLeaderAudio(audio)) return
  if (shouldPlayOnLoad.value && audio) {
    shouldPlayOnLoad.value = false
    if (playerStore.isStemMixerActive) {
      void playStemMixer(playerStore.currentTime).catch(handlePlayError)
    } else {
      audio.play().catch(handlePlayError)
    }
  }
}

function onTimeUpdate(event: Event) {
  const audio = event.target as HTMLAudioElement | null
  if (!audio || !isTransportLeaderAudio(audio) || isSeeking.value) return
  playerStore.updateTime(audio.currentTime)
  if (playerStore.isStemMixerActive) {
    syncStemFollowerDrift()
  }
  void maybeAutoSkipDislikedSegment(audio.currentTime)
}

function onEnded(event: Event) {
  const audio = event.target as HTMLAudioElement | null
  if (!audio || !isTransportLeaderAudio(audio)) return
  playerStore.onTrackEnded()
}

function onSeekStart() {
  isSeeking.value = true
  seekValue.value = playerStore.currentTime
}

function onSeekInput(event: Event) {
  const target = event.target as HTMLInputElement
  seekValue.value = parseFloat(target.value)
}

function onSeekEnd() {
  resetAutoSkipState()

  if (playerStore.isStemMixerActive) {
    void syncStemMixerPositions(seekValue.value)
    playerStore.seek(seekValue.value)
    isSeeking.value = false
    return
  }

  const primaryAudio = getPrimaryAudio()
  if (primaryAudio) {
    primaryAudio.currentTime = seekValue.value
    playerStore.seek(seekValue.value)
  }
  isSeeking.value = false
}

function onVolumeChange(event: Event) {
  const target = event.target as HTMLInputElement
  playerStore.setVolume(parseFloat(target.value))
}

function handlePlayError(e: Error) {
  if (e.name === 'AbortError') return
  console.error('Play error:', e)
}

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms))
}

function resetAutoSkipState() {
  autoSkipCooldownUntil.value = 0
  lastAutoSkippedRunKey.value = null
  isAutoSkippingSegment.value = false

  const now = audioContextRef.value?.currentTime
  if (typeof now === 'number') {
    gainNodeRef.value?.gain.cancelScheduledValues(now)
    shadowGainNodeRef.value?.gain.cancelScheduledValues(now)
  }

  getSecondaryAudio()?.pause()
  syncAudioGain()
}

function isDislikedSegment(segment: RatedSegmentSnapshot | undefined) {
  return Boolean(
    segment
    && typeof segment.rating === 'number'
    && segment.rating <= AUTO_SKIP_MAX_RATING
    && segment.end > segment.start,
  )
}

function getTriggeredDislikedRun(currentTime: number) {
  for (let index = 0; index < sortedTrackSegments.value.length; index += 1) {
    const candidate = sortedTrackSegments.value[index]
    if (!isDislikedSegment(candidate)) continue

    let runEndIndex = index
    while (runEndIndex < sortedTrackSegments.value.length - 1 && isDislikedSegment(sortedTrackSegments.value[runEndIndex + 1])) {
      runEndIndex += 1
    }

    const runStart = candidate
    const runEnd = sortedTrackSegments.value[runEndIndex]
    const triggerAt = Math.max(0, runStart.start - AUTO_SKIP_OVERLAP_SEC)
    const cutoffAt = Math.max(triggerAt, runEnd.end - AUTO_SKIP_SEEK_PADDING_SEC)

    if (currentTime >= triggerAt && currentTime < cutoffAt) {
      return {
        key: `${runStart.id}:${runEnd.id}`,
        start: runStart.start,
        end: runEnd.end,
        triggerAt,
        segmentIds: sortedTrackSegments.value.slice(index, runEndIndex + 1).map((segment) => segment.id),
      }
    }

    index = runEndIndex
  }

  return null
}

function waitForCanPlay(audio: HTMLAudioElement) {
  if (audio.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
    return Promise.resolve()
  }

  return new Promise<void>((resolve, reject) => {
    const onReady = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error('Audio element failed to become ready'))
    }
    const cleanup = () => {
      audio.removeEventListener('canplay', onReady)
      audio.removeEventListener('error', onError)
    }

    audio.addEventListener('canplay', onReady, { once: true })
    audio.addEventListener('error', onError, { once: true })
  })
}

function waitForSeeked(audio: HTMLAudioElement, targetTime: number) {
  if (Math.abs(audio.currentTime - targetTime) < 0.08) {
    return Promise.resolve()
  }

  return new Promise<void>((resolve, reject) => {
    const onSeeked = () => {
      cleanup()
      resolve()
    }
    const onError = () => {
      cleanup()
      reject(new Error('Audio element failed while seeking'))
    }
    const cleanup = () => {
      audio.removeEventListener('seeked', onSeeked)
      audio.removeEventListener('error', onError)
    }

    audio.addEventListener('seeked', onSeeked, { once: true })
    audio.addEventListener('error', onError, { once: true })
  })
}

function syncCurrentTrackSegments(trackId: number, segments: Array<{ id: number; start: number; end: number; rating?: number | null }> | undefined) {
  if (playerStore.currentTrack?.id !== trackId) return
  currentTrackSegments.value = Array.isArray(segments)
    ? segments.map((segment) => ({
        id: segment.id,
        start: segment.start,
        end: segment.end,
        rating: segment.rating,
      }))
    : []
}

async function refreshCurrentTrackSegments(trackId: number) {
  try {
    const snapshot = await api.getTrackFeedbackSnapshot(trackId)
    syncCurrentTrackSegments(trackId, snapshot.segments)
  } catch (e) {
    console.error('Failed to refresh current track feedback:', e)
  }
}

function toggleSkipDislikedSegments() {
  const nextEnabled = !playerStore.skipDislikedSegmentsEnabled
  playerStore.setSkipDislikedSegments(nextEnabled)
  autoSkipCooldownUntil.value = 0
  lastAutoSkippedRunKey.value = null
  showToast(
    nextEnabled
      ? 'Auto-skip for 1-star tracks and segments enabled.'
      : 'Auto-skip for 1-star tracks and segments disabled.',
    'info',
  )
}

async function hardSeekSkip(targetTime: number) {
  if (playerStore.isStemMixerActive) {
    const clampedTarget = Math.max(0, Math.min(targetTime, playerStore.duration || targetTime))
    await syncStemMixerPositions(clampedTarget)
    playerStore.seek(clampedTarget)
    return
  }

  const primaryAudio = getPrimaryAudio()
  if (!primaryAudio) return

  const clampedTarget = Math.max(0, Math.min(targetTime, primaryAudio.duration || playerStore.duration || targetTime))
  primaryAudio.currentTime = clampedTarget
  playerStore.seek(clampedTarget)
}

async function overlapSkipToTime(targetTime: number) {
  if (playerStore.isStemMixerActive) {
    await hardSeekSkip(targetTime)
    return
  }

  const mainAudio = getPrimaryAudio()
  const shadowAudio = getSecondaryAudio()

  if (!mainAudio || !shadowAudio) {
    await hardSeekSkip(targetTime)
    return
  }

  ensureAudioGraph()
  ensureShadowAudioGraph()

  const mainGainNode = getPrimaryGainNode()
  const shadowGainNode = getSecondaryGainNode()

  if (!mainGainNode || !shadowGainNode || !audioContextRef.value) {
    await hardSeekSkip(targetTime)
    return
  }

  const clampedTarget = Math.max(0, Math.min(targetTime, mainAudio.duration || playerStore.duration || targetTime))
  const targetGain = Math.max(0.0001, effectiveVolume.value)
  const outgoingGain = Math.max(0.0001, targetGain * AUTO_SKIP_OUTGOING_GAIN_RATIO)
  const incomingGain = Math.max(0.0001, targetGain * AUTO_SKIP_INCOMING_GAIN_RATIO)

  if (shadowAudio.src !== audioUrl.value) {
    shadowAudio.src = audioUrl.value
    shadowAudio.load()
  }

  await waitForCanPlay(shadowAudio)
  shadowAudio.pause()
  shadowAudio.currentTime = clampedTarget
  await waitForSeeked(shadowAudio, clampedTarget)

  const now = audioContextRef.value.currentTime
  const blendEndsAt = now + Math.min(AUTO_SKIP_BLEND_SEC, AUTO_SKIP_OVERLAP_SEC)
  mainGainNode.gain.cancelScheduledValues(now)
  shadowGainNode.gain.cancelScheduledValues(now)
  mainGainNode.gain.setValueAtTime(outgoingGain, now)
  mainGainNode.gain.linearRampToValueAtTime(0.0001, blendEndsAt)
  shadowGainNode.gain.setValueAtTime(incomingGain, now)
  shadowGainNode.gain.linearRampToValueAtTime(targetGain, blendEndsAt)
  await shadowAudio.play().catch((e) => {
    throw e instanceof Error ? e : new Error('Failed to start shadow audio')
  })

  await wait(AUTO_SKIP_OVERLAP_SEC * 1000)

  const handoffTime = shadowAudio.currentTime
  const handoffNow = audioContextRef.value.currentTime
  mainGainNode.gain.cancelScheduledValues(handoffNow)
  shadowGainNode.gain.cancelScheduledValues(handoffNow)
  mainGainNode.gain.setValueAtTime(0, handoffNow)
  shadowGainNode.gain.setValueAtTime(targetGain, handoffNow)
  mainAudio.pause()
  activeAudioSlot.value = activeAudioSlot.value === 'main' ? 'shadow' : 'main'
  playerStore.seek(handoffTime)
  syncAudioGain()
}

async function maybeAutoSkipDislikedSegment(currentTime: number) {
  if (!playerStore.skipDislikedSegmentsEnabled || !playerStore.isPlaying || isSeeking.value || isSelectingFragment.value || showFragmentDialog.value) {
    return
  }
  const primaryAudio = getPrimaryAudio()
  if (!primaryAudio || isAutoSkippingSegment.value) return

  const activeRun = getTriggeredDislikedRun(currentTime)

  if (!activeRun) {
    if (lastAutoSkippedRunKey.value !== null && currentTime >= autoSkipCooldownUntil.value + AUTO_SKIP_RESET_PADDING_SEC) {
      lastAutoSkippedRunKey.value = null
    }
    return
  }

  if (currentTime < autoSkipCooldownUntil.value || activeRun.key === lastAutoSkippedRunKey.value) {
    return
  }

  isAutoSkippingSegment.value = true
  lastAutoSkippedRunKey.value = activeRun.key

  try {
    const targetTime = Math.min(
      primaryAudio.duration || playerStore.duration || activeRun.end,
      activeRun.end + AUTO_SKIP_SEEK_PADDING_SEC,
    )
    autoSkipCooldownUntil.value = targetTime
    await overlapSkipToTime(targetTime)
  } catch (e) {
    console.error('Failed to auto-skip disliked segment:', e)
  } finally {
    isAutoSkippingSegment.value = false
    syncAudioGain()
  }
}

function onTrackFeedbackUpdated(event: Event) {
  const customEvent = event as CustomEvent<TrackFeedbackUpdatedDetail>
  const trackId = customEvent.detail?.trackId
  if (!trackId) return
  void playerStore.enforceSkipDislikedTracks(trackId)
  if (playerStore.currentTrack?.id !== trackId) return
  void refreshCurrentTrackSegments(trackId)
}

function onAudioError(event: Event) {
  const audio = event.target as HTMLAudioElement
  if (!isTransportLeaderAudio(audio)) return
  const error = audio?.error
  console.error('Audio load error:', error?.message || 'Unknown error', 'Code:', error?.code)

  const trackTitle = playerStore.currentTrack?.title || 'Track'
  let message = `Unable to play "${trackTitle}".`
  if (error) {
    switch (error.code) {
      case MediaError.MEDIA_ERR_ABORTED:
        return
      case MediaError.MEDIA_ERR_NETWORK:
        message = `Unable to play "${trackTitle}" due to a network issue.`
        break
      case MediaError.MEDIA_ERR_DECODE:
        message = `Unable to decode "${trackTitle}".`
        break
      case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
        message = `Audio source for "${trackTitle}" is not available.`
        break
    }
  }

  showToast(message, 'error')
  localStorage.removeItem('player_currentTrack')
  localStorage.removeItem('player_queue')
  playerStore.clearCurrentTrack()
}

function syncPlayerSources(nextUrl: string, options: { resumeAt?: number } = {}) {
  if (!audioRef.value || !playerStore.currentTrack || !nextUrl) {
    return
  }

  const resumeAt = typeof options.resumeAt === 'number' && options.resumeAt > 0
    ? options.resumeAt
    : null

  resetAutoSkipState()
  activeAudioSlot.value = 'main'
  ensureAudioGraph()
  ensureShadowAudioGraph()
  syncAudioGain()
  shouldPlayOnLoad.value = playerStore.isPlaying

  const primaryAudio = audioRef.value
  primaryAudio.pause()

  if (resumeAt !== null) {
    primaryAudio.addEventListener('loadedmetadata', () => {
      if (!audioRef.value) return
      const cappedTime = Math.min(resumeAt, audioRef.value.duration || resumeAt)
      audioRef.value.currentTime = cappedTime
      playerStore.seek(cappedTime)
    }, { once: true })
  }

  primaryAudio.src = nextUrl
  primaryAudio.load()

  if (shadowAudioRef.value) {
    shadowAudioRef.value.pause()
    shadowAudioRef.value.currentTime = 0
    shadowAudioRef.value.src = nextUrl
    shadowAudioRef.value.load()
  }
}

watch(() => playerStore.isPlaying, (playing) => {
  if (!playerStore.currentTrack) {
    return
  }

  if (playerStore.isStemMixerActive) {
    if (playing) {
      resumeAudioContext()
      void playStemMixer(playerStore.currentTime).catch(handlePlayError)
    } else {
      pauseStemMixer()
    }
    return
  }

  const primaryAudio = getPrimaryAudio()
  if (primaryAudio) {
    if (playing) {
      resumeAudioContext()
      primaryAudio.play().catch(handlePlayError)
    } else {
      audioRef.value?.pause()
      shadowAudioRef.value?.pause()
    }
  }
})

watch(() => playerStore.currentTrack, async (newTrack) => {
  if (!newTrack) {
    playerStore.clearPlaybackContext()
    return
  }

  await playerStore.loadPlaybackManifest(newTrack.id)
  await nextTick()
  syncStemMixerSources()

  if (audioRef.value && playerStore.currentTrack?.id === newTrack.id) {
    if (playerStore.isStemMixerActive) {
      pauseStemMixer()
      if (playerStore.isPlaying) {
        void playStemMixer(playerStore.currentTime).catch(handlePlayError)
      }
    } else {
      syncPlayerSources(audioUrl.value)
    }
  }
})

watch(audioUrl, (nextUrl, previousUrl) => {
  if (playerStore.isStemMixerActive) {
    return
  }

  if (!nextUrl || nextUrl === previousUrl || !audioRef.value || !playerStore.currentTrack) {
    return
  }

  if (audioRef.value.src === nextUrl && (!shadowAudioRef.value || shadowAudioRef.value.src === nextUrl)) {
    return
  }

  syncPlayerSources(nextUrl, { resumeAt: playerStore.currentTime })
})

watch(effectiveVolume, () => {
  syncAudioGain()
})

watch(() => playerStore.currentTime, (time) => {
  if (playerStore.isStemMixerActive) {
    const leaderAudio = getStemLeaderAudio()
    if (leaderAudio && !isSeeking.value && Math.abs(leaderAudio.currentTime - time) > 1) {
      resetAutoSkipState()
      void syncStemMixerPositions(time)
    }
    return
  }

  const primaryAudio = getPrimaryAudio()
  if (primaryAudio && !isSeeking.value && Math.abs(primaryAudio.currentTime - time) > 1) {
    resetAutoSkipState()
    primaryAudio.currentTime = time
  }
})

watch(() => playerStore.isStemMixerActive, (enabled) => {
  if (!playerStore.currentTrack) {
    return
  }

  syncStemMixerSources()
  syncAudioGain()

  if (enabled) {
    audioRef.value?.pause()
    shadowAudioRef.value?.pause()
    if (playerStore.isPlaying) {
      void playStemMixer(playerStore.currentTime).catch(handlePlayError)
    }
    return
  }

  pauseStemMixer()
  if (audioUrl.value) {
    syncPlayerSources(audioUrl.value, { resumeAt: playerStore.currentTime })
  }
})

watch(() => playerStore.stemEnabledByKey, () => {
  syncAudioGain()
}, { deep: true })

onMounted(() => {
  const mountPlaybackContext = async () => {
    if (!audioRef.value) {
      return
    }

    activeAudioSlot.value = 'main'
    ensureAudioGraph()
    ensureShadowAudioGraph()
    syncAudioGain()
    if (playerStore.currentTrack) {
      await playerStore.loadPlaybackManifest(playerStore.currentTrack.id)
    }
    syncStemMixerSources()
    if (playerStore.currentTrack && audioUrl.value) {
      if (playerStore.isStemMixerActive) {
        if (playerStore.isPlaying) {
          await playStemMixer(playerStore.currentTime).catch(handlePlayError)
        }
      } else {
        syncPlayerSources(audioUrl.value)
      }
    }
  }

  void mountPlaybackContext()
  window.addEventListener('track-feedback-updated', onTrackFeedbackUpdated as EventListener)
  if (playerStore.skipDislikedSegmentsEnabled) {
    void playerStore.enforceSkipDislikedTracks()
  }
})

onUnmounted(() => {
  window.removeEventListener('track-feedback-updated', onTrackFeedbackUpdated as EventListener)
  Array.from(stemAudioElements.keys()).forEach((stemKey) => {
    destroyStemAudio(stemKey)
  })
  mediaSourceRef.value?.disconnect()
  gainNodeRef.value?.disconnect()
  shadowMediaSourceRef.value?.disconnect()
  shadowGainNodeRef.value?.disconnect()
  audioContextRef.value?.close().catch(() => undefined)
  mediaSourceRef.value = null
  gainNodeRef.value = null
  shadowMediaSourceRef.value = null
  shadowGainNodeRef.value = null
  audioContextRef.value = null
  connectedAudioRef.value = null
  shadowConnectedAudioRef.value = null
})
</script>

<template>
  <div class="fixed bottom-2 left-2 right-2 z-50 sm:bottom-3 sm:left-3 sm:right-3">
    <audio
      v-if="playerStore.currentTrack"
      ref="audioRef"
      :src="audioUrl"
      @loadedmetadata="onLoadedMetadata"
      @canplaythrough="onCanPlay"
      @timeupdate="onTimeUpdate"
      @ended="onEnded"
      @error="onAudioError"
      preload="auto"
    />
    <audio
      v-if="playerStore.currentTrack"
      ref="shadowAudioRef"
      class="hidden"
      @loadedmetadata="onLoadedMetadata"
      @canplaythrough="onCanPlay"
      @timeupdate="onTimeUpdate"
      @ended="onEnded"
      @error="onAudioError"
      preload="auto"
      aria-hidden="true"
      tabindex="-1"
    />

    <div
      v-if="playerStore.lyricsVisible && playerStore.hasLyrics"
      class="player-lyrics-card mx-auto mb-2 max-w-7xl text-slate-100"
    >
      <template v-if="playerStore.lyricsPayload.kind === 'timed_text'">
        <p class="player-lyrics-line player-lyrics-line--ghost">{{ previousLyricsCue?.text || ' ' }}</p>
        <p class="player-lyrics-line player-lyrics-line--active">{{ activeLyricsCue?.text || nextLyricsCue?.text || '...' }}</p>
        <p class="player-lyrics-line player-lyrics-line--ghost">{{ nextLyricsCue?.text || ' ' }}</p>
      </template>
      <pre v-else class="player-lyrics-pre">{{ playerStore.lyricsPayload.text }}</pre>
    </div>

    <div class="player-dock mx-auto max-w-7xl overflow-hidden rounded-[2rem] text-slate-100">
      <div class="px-3 pt-2.5 sm:px-5 sm:pt-3">
        <div class="player-dock__meter relative overflow-hidden rounded-full">
          <div
            class="player-dock__meter-fill absolute inset-y-0 left-0 rounded-full"
            :style="{ width: `${playerStore.progress}%` }"
          />
          <input
            type="range"
            min="0"
            :max="playerStore.duration || 100"
            :value="isSeeking ? seekValue : playerStore.currentTime"
            step="0.1"
            class="player-progress-range absolute inset-0 h-full w-full cursor-pointer appearance-none bg-transparent"
            @mousedown="onSeekStart"
            @touchstart="onSeekStart"
            @input="onSeekInput"
            @mouseup="onSeekEnd"
            @touchend="onSeekEnd"
          />
        </div>
      </div>

      <div class="grid gap-2 px-3 pb-2.5 pt-2 sm:px-5 sm:pt-2.5 xl:grid-cols-[minmax(0,1.45fr)_minmax(14rem,0.86fr)_minmax(13rem,0.62fr)] xl:items-center">
        <section class="player-track-card">
          <div class="player-track-card__cover">{{ trackMonogram }}</div>
          <div class="min-w-0 flex-1">
            <div class="flex flex-wrap items-center gap-2">
              <span class="player-meta-pill player-meta-pill--neutral">
                {{ hasTrack ? 'Now playing' : 'Player idle' }}
              </span>
              <span
                class="player-meta-pill"
                :class="{
                  'player-meta-pill--ready': playerStore.currentTrack?.analysis_status === 'completed',
                  'player-meta-pill--working': playerStore.currentTrack?.analysis_status === 'processing' || playerStore.currentTrack?.analysis_status === 'analyzing',
                  'player-meta-pill--queued': playerStore.currentTrack?.analysis_status === 'pending',
                  'player-meta-pill--failed': playerStore.currentTrack?.analysis_status === 'failed',
                  'player-meta-pill--idle': !playerStore.currentTrack,
                }"
              >
                {{ trackStatusLabel }}
              </span>
              <span v-for="fact in trackFacts" :key="fact" class="player-fact-chip">
                {{ fact }}
              </span>
            </div>

            <div class="mt-1.5 truncate text-[1.02rem] font-semibold tracking-tight text-white sm:text-[1.15rem]">
              {{ playerStore.currentTrack?.title || 'No track selected' }}
            </div>
            <div class="mt-1 truncate text-[0.84rem] text-slate-300/76">
              <ArtistCreditsInline
                :credits="playerStore.currentTrack?.artist_credits"
                :fallback="trackSubtitle"
                link-class="transition hover:text-white"
                empty-text="Choose something from the library to start listening."
              />
            </div>

            <div class="mt-2.5 flex flex-wrap items-center gap-2">
              <div v-if="hasTrack" class="player-rating-shell">
                <RatingWidget
                  v-model="trackRating"
                  size="sm"
                  @rate="onRatingChange"
                  @clear="onRatingChange(null)"
                />
              </div>
              <div v-if="fragmentSelectionLabel" class="player-inline-note">
                {{ fragmentSelectionLabel }}
              </div>
              <div v-if="isLoadingRating" class="player-inline-note player-inline-note--muted">
                Loading rating
              </div>
            </div>

            <div
              v-if="playerStore.availablePlaybackModes.length > 1 || playerStore.canUseStemMixer || playerStore.hasLyrics"
              class="player-session-switches mt-2.5"
            >
              <button
                v-for="mode in playerStore.availablePlaybackModes"
                :key="mode.key"
                class="player-chip-button"
                :class="{ 'player-chip-button--active': !playerStore.isStemMixerActive && playerStore.selectedPlaybackModeKey === mode.key }"
                @click="playerStore.setStemMixerEnabled(false); playerStore.setPlaybackMode(mode.key)"
                :title="mode.description || mode.label"
              >
                {{ mode.label }}
              </button>
              <button
                v-if="playerStore.canUseStemMixer"
                class="player-chip-button"
                :class="{ 'player-chip-button--active': playerStore.isStemMixerActive }"
                @click="playerStore.setStemMixerEnabled(!playerStore.isStemMixerActive)"
                title="Enable realtime stem mixer"
              >
                Stem mix
              </button>
              <button
                v-if="playerStore.hasLyrics"
                class="player-chip-button"
                :class="{ 'player-chip-button--active': playerStore.lyricsVisible }"
                @click="playerStore.setLyricsVisible(!playerStore.lyricsVisible)"
                title="Show or hide lyrics"
              >
                Lyrics
              </button>
            </div>

            <div v-if="playerStore.isStemMixerActive && playerStore.availableStemSources.length" class="player-stem-switches mt-2">
              <button
                v-for="stem in playerStore.availableStemSources"
                :key="stem.key"
                class="player-chip-button"
                :class="{ 'player-chip-button--active': playerStore.stemEnabledByKey[stem.key] !== false }"
                @click="playerStore.toggleStemSource(stem.key)"
                :title="stem.description || stem.label"
              >
                {{ stem.label }}
              </button>
            </div>
          </div>
        </section>

        <section class="player-transport-card">
          <div class="player-transport-card__header">
            <button
              @click="playerStore.toggleShuffle()"
              class="player-chip-button"
              :class="{ 'player-chip-button--active': playerStore.shuffleMode }"
              title="Shuffle queue"
            >
              Mix
            </button>
            <button
              @click="playerStore.cycleRepeatMode()"
              class="player-chip-button"
              :class="{ 'player-chip-button--active': playerStore.repeatMode !== 'off' }"
              :title="`Repeat mode: ${playerStore.repeatMode}`"
            >
              {{ repeatIcon }}
            </button>
            <button
              v-if="hasTrack"
              @click="isSelectingFragment ? setFragmentEnd() : startFragmentSelection()"
              class="player-chip-button"
              :class="{ 'player-chip-button--accent': isSelectingFragment }"
              :title="isSelectingFragment ? 'Set clip end point' : 'Rate a clip from the current track'"
            >
              {{ isSelectingFragment ? 'Set end' : 'Clip' }}
            </button>
            <button
              v-if="hasTrack"
              @click="toggleSkipDislikedSegments"
              class="player-chip-button"
              :class="{ 'player-chip-button--active': playerStore.skipDislikedSegmentsEnabled }"
              title="Skip 1-star tracks and auto-skip 1-star segments"
            >
              Skip 1★
            </button>
          </div>

          <div class="player-transport-card__core">
            <button
              @click="playerStore.previous()"
              :disabled="!playerStore.hasPrevious"
              class="player-nav-button"
              title="Previous track"
            >
              Prev
            </button>
            <button
              @click="playerStore.isPlaying ? playerStore.pause() : playerStore.play()"
              :disabled="!hasTrack"
              class="player-play-button"
              title="Play or pause"
            >
              {{ playerStore.isPlaying ? 'Pause' : 'Play' }}
            </button>
            <button
              @click="playerStore.next()"
              :disabled="!playerStore.hasNext"
              class="player-nav-button"
              title="Next track"
            >
              Next
            </button>
          </div>

          <div class="player-transport-card__footer">
            <span>{{ formattedCurrentTime }}</span>
            <span class="player-transport-card__divider" />
            <span>{{ formattedDuration }}</span>
            <span class="player-transport-card__progress">
              {{ Math.round(playerStore.progress) }}%
            </span>
          </div>
        </section>

        <section class="player-utility-card">
          <div class="player-utility-card__row">
            <QueuePanel />
            <div class="player-volume-shell">
              <button
                @click="playerStore.toggleMute()"
                class="player-volume-button"
                title="Mute or unmute"
              >
                {{ playerStore.isMuted || playerStore.volume === 0 ? 'Off' : 'Vol' }}
              </button>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                :value="playerStore.volume"
                @input="onVolumeChange"
                class="player-volume-range h-2 w-full cursor-pointer appearance-none bg-transparent"
              />
              <span class="player-volume-value">{{ volumeLabel }}</span>
            </div>
          </div>
        </section>
      </div>
    </div>
  </div>

  <Teleport to="body">
    <Transition name="fade">
      <div
        v-if="showFragmentDialog"
        class="fixed inset-0 z-[100] flex items-center justify-center bg-slate-950/80 px-4 backdrop-blur-md"
        @click.self="cancelFragmentSelection"
      >
        <div class="player-modal w-full max-w-xl">
          <div class="flex items-start justify-between gap-4">
            <div>
              <p class="player-modal__kicker">Clip feedback</p>
              <h3 class="player-modal__title">Rate this section</h3>
              <p class="player-modal__copy">
                {{ formatTime(fragmentStart || 0) }} - {{ formatTime(fragmentEnd || 0) }}
                ({{ ((fragmentEnd || 0) - (fragmentStart || 0)).toFixed(1) }}s)
              </p>
            </div>
            <button
              @click="cancelFragmentSelection"
              class="player-modal__close"
            >
              Close
            </button>
          </div>

          <div class="mt-6">
            <label class="player-modal__label" for="fragment-comment">
              What stands out in this clip?
            </label>
            <textarea
              id="fragment-comment"
              v-model="fragmentComment"
              placeholder="Example: the drop hits hard, the transition drags, the vocal hook stands out..."
              class="player-modal__textarea"
              maxlength="500"
            />
            <p class="player-modal__meta">{{ fragmentComment.length }}/500</p>
          </div>

          <div class="mt-6 flex flex-col gap-3 sm:flex-row">
            <button
              @click="submitFragment(true)"
              class="player-modal__action player-modal__action--primary"
            >
              Save as liked clip
            </button>
            <button
              @click="submitFragment(false)"
              class="player-modal__action player-modal__action--danger"
            >
              Save as disliked clip
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.player-lyrics-card {
  border: 1px solid rgba(148, 163, 184, 0.16);
  border-radius: 1.4rem;
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 30%),
    linear-gradient(180deg, rgba(15, 23, 42, 0.94) 0%, rgba(15, 23, 42, 0.9) 100%);
  box-shadow:
    0 16px 36px rgba(15, 23, 42, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.06);
  padding: 0.95rem 1.15rem;
  backdrop-filter: blur(10px);
}

.player-lyrics-line {
  margin: 0;
  text-align: center;
  line-height: 1.45;
}

.player-lyrics-line--ghost {
  color: rgba(148, 163, 184, 0.78);
  font-size: 0.8rem;
}

.player-lyrics-line--active {
  margin: 0.22rem 0;
  color: #f8fafc;
  font-size: 1.05rem;
  font-weight: 800;
  letter-spacing: 0.01em;
}

.player-lyrics-pre {
  margin: 0;
  max-height: 10rem;
  overflow-y: auto;
  white-space: pre-wrap;
  color: #e2e8f0;
  font-size: 0.84rem;
  line-height: 1.55;
}

.player-dock {
  border: 1px solid rgba(148, 163, 184, 0.16);
  background:
    radial-gradient(circle at 14% 18%, rgba(56, 189, 248, 0.12), transparent 22%),
    radial-gradient(circle at 86% 20%, rgba(249, 115, 22, 0.08), transparent 18%),
    linear-gradient(180deg, rgba(15, 23, 42, 0.93) 0%, rgba(15, 23, 42, 0.9) 100%);
  box-shadow:
    0 18px 42px rgba(15, 23, 42, 0.26),
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    inset 0 -1px 0 rgba(15, 23, 42, 0.35);
  backdrop-filter: blur(10px);
}

.player-dock__meter {
  height: 0.32rem;
  background: rgba(148, 163, 184, 0.12);
}

.player-dock__meter-fill {
  background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 48%, #67e8f9 100%);
  box-shadow: 0 0 16px rgba(56, 189, 248, 0.48);
}

.player-track-card,
.player-transport-card,
.player-utility-card {
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
}

.player-track-card {
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 0.72rem;
  padding: 0.1rem 0;
}

.player-track-card__cover {
  display: flex;
  height: 3.1rem;
  width: 3.1rem;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  border-radius: 0.95rem;
  background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 58%, #4338ca 100%);
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.2);
  color: #ffffff;
  font-size: 1.08rem;
  font-weight: 800;
  letter-spacing: 0.1em;
}

.player-transport-card {
  padding: 0.1rem 0;
}

.player-transport-card__header,
.player-transport-card__core {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: center;
  gap: 0.7rem;
}

.player-transport-card__core {
  margin-top: 0.55rem;
}

.player-transport-card__footer {
  margin-top: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.45rem;
  color: #cbd5e1;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}

.player-transport-card__divider {
  height: 0.3rem;
  width: 0.3rem;
  border-radius: 999px;
  background: rgba(148, 163, 184, 0.55);
}

.player-transport-card__progress {
  margin-left: 0.2rem;
  color: #7dd3fc;
}

.player-utility-card {
  display: flex;
  min-width: 0;
  align-items: stretch;
  padding: 0.1rem 0;
}

.player-utility-card__row {
  display: flex;
  width: 100%;
  flex-direction: column;
  align-items: flex-start;
  align-content: flex-start;
  gap: 0.6rem;
}

.player-meta-pill,
.player-fact-chip,
.player-inline-note,
.player-chip-button,
.player-nav-button,
.player-volume-button,
.player-modal__close,
.player-modal__action {
  transition:
    transform 0.18s ease,
    border-color 0.18s ease,
    background-color 0.18s ease,
    box-shadow 0.18s ease,
    color 0.18s ease;
}

.player-meta-pill {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  padding: 0.3rem 0.62rem;
  font-size: 0.62rem;
  font-weight: 800;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.player-meta-pill--neutral,
.player-meta-pill--idle {
  border: 1px solid rgba(148, 163, 184, 0.14);
  background: rgba(255, 255, 255, 0.05);
  color: #cbd5e1;
}

.player-meta-pill--ready {
  border: 1px solid rgba(52, 211, 153, 0.2);
  background: rgba(16, 185, 129, 0.16);
  color: #d1fae5;
}

.player-meta-pill--working {
  border: 1px solid rgba(96, 165, 250, 0.22);
  background: rgba(37, 99, 235, 0.16);
  color: #dbeafe;
}

.player-meta-pill--queued {
  border: 1px solid rgba(251, 191, 36, 0.22);
  background: rgba(245, 158, 11, 0.16);
  color: #fef3c7;
}

.player-meta-pill--failed {
  border: 1px solid rgba(248, 113, 113, 0.24);
  background: rgba(239, 68, 68, 0.16);
  color: #fee2e2;
}

.player-fact-chip,
.player-inline-note {
  display: inline-flex;
  align-items: center;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
  padding: 0.32rem 0.64rem;
  color: #cbd5e1;
  font-size: 0.64rem;
  font-weight: 800;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.player-inline-note {
  border-color: rgba(168, 85, 247, 0.2);
  background: rgba(168, 85, 247, 0.12);
  color: #f5d0fe;
}

.player-inline-note--muted {
  border-color: rgba(148, 163, 184, 0.14);
  background: rgba(255, 255, 255, 0.04);
  color: #94a3b8;
}

.player-rating-shell,
.player-volume-shell {
  border-radius: 1rem;
  border: 1px solid rgba(255, 255, 255, 0.06);
  background: rgba(255, 255, 255, 0.035);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
}

.player-rating-shell {
  padding: 0.4rem 0.58rem;
}

.player-session-switches,
.player-stem-switches {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.player-volume-shell {
  display: flex;
  align-items: center;
  width: 100%;
  align-self: stretch;
  flex: 1;
  min-width: 0;
  gap: 0.55rem;
  padding: 0.44rem 0.56rem;
}

.player-chip-button,
.player-nav-button,
.player-volume-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.07);
  background: rgba(255, 255, 255, 0.04);
  color: #e2e8f0;
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.15em;
  text-transform: uppercase;
}

.player-chip-button {
  min-height: 2rem;
  padding: 0.44rem 0.72rem;
}

.player-nav-button {
  min-height: 2.3rem;
  min-width: 3.8rem;
  padding: 0.52rem 0.78rem;
}

.player-chip-button:hover:not(:disabled),
.player-nav-button:hover:not(:disabled),
.player-volume-button:hover:not(:disabled) {
  transform: translateY(-1px);
  border-color: rgba(125, 211, 252, 0.24);
  background: rgba(255, 255, 255, 0.08);
}

.player-chip-button--active {
  border-color: rgba(56, 189, 248, 0.26);
  background: rgba(14, 165, 233, 0.16);
  color: #e0f2fe;
}

.player-chip-button--accent {
  border-color: rgba(217, 70, 239, 0.28);
  background: rgba(168, 85, 247, 0.18);
  color: #fae8ff;
}

.player-nav-button:disabled,
.player-chip-button:disabled,
.player-volume-button:disabled,
.player-play-button:disabled {
  opacity: 0.42;
  cursor: not-allowed;
}

.player-play-button {
  display: inline-flex;
  min-height: 2.8rem;
  min-width: 5.75rem;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid rgba(125, 211, 252, 0.26);
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  color: #0f172a;
  box-shadow:
    0 10px 24px rgba(14, 165, 233, 0.18),
    inset 0 1px 0 rgba(255, 255, 255, 0.9);
  font-size: 0.76rem;
  font-weight: 900;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  transition:
    transform 0.18s ease,
    box-shadow 0.18s ease,
    filter 0.18s ease;
}

.player-play-button:hover:not(:disabled) {
  transform: translateY(-1px) scale(1.01);
  box-shadow:
    0 14px 30px rgba(14, 165, 233, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.96);
  filter: brightness(1.02);
}

.player-volume-button {
  min-width: 3.4rem;
  padding: 0.42rem 0.58rem;
}

.player-volume-value {
  display: inline-flex;
  min-width: 3.25rem;
  justify-content: center;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.06);
  background: rgba(255, 255, 255, 0.04);
  padding: 0.38rem 0.54rem;
  color: #cbd5e1;
  font-size: 0.66rem;
  font-weight: 800;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.player-progress-range,
.player-volume-range {
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
}

.player-volume-range {
  flex: 1;
  min-width: 7rem;
}

.player-volume-range::-webkit-slider-runnable-track {
  height: 0.38rem;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(125, 211, 252, 0.24), rgba(148, 163, 184, 0.18));
}

.player-volume-range::-moz-range-track {
  height: 0.38rem;
  border-radius: 999px;
  background: linear-gradient(90deg, rgba(125, 211, 252, 0.24), rgba(148, 163, 184, 0.18));
}

.player-progress-range::-webkit-slider-thumb,
.player-volume-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 999px;
  border: 3px solid #38bdf8;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(14, 165, 233, 0.42);
}

.player-progress-range::-moz-range-thumb,
.player-volume-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border: 3px solid #38bdf8;
  border-radius: 999px;
  background: #ffffff;
  box-shadow: 0 8px 24px rgba(14, 165, 233, 0.42);
}

.player-modal {
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 1.8rem;
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 28%),
    linear-gradient(180deg, rgba(15, 23, 42, 0.98) 0%, rgba(2, 6, 23, 0.99) 100%);
  box-shadow:
    0 34px 90px rgba(2, 6, 23, 0.55),
    inset 0 1px 0 rgba(255, 255, 255, 0.06);
  padding: 1.6rem;
  color: #e2e8f0;
}

.player-modal__kicker,
.player-modal__label,
.player-modal__meta {
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.22em;
  text-transform: uppercase;
}

.player-modal__kicker,
.player-modal__label {
  color: #7dd3fc;
}

.player-modal__title {
  margin-top: 0.8rem;
  color: #ffffff;
  font-size: 1.9rem;
  font-weight: 700;
  letter-spacing: -0.03em;
}

.player-modal__copy {
  margin-top: 0.65rem;
  color: rgba(226, 232, 240, 0.78);
  font-size: 0.94rem;
  line-height: 1.6;
}

.player-modal__close {
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.05);
  padding: 0.7rem 0.95rem;
  color: #e2e8f0;
}

.player-modal__textarea {
  width: 100%;
  min-height: 8rem;
  border-radius: 1.2rem;
  border: 1px solid rgba(148, 163, 184, 0.16);
  background: rgba(255, 255, 255, 0.05);
  padding: 1rem;
  color: #f8fafc;
  resize: none;
  outline: none;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.player-modal__textarea:focus {
  border-color: rgba(56, 189, 248, 0.28);
  box-shadow:
    0 0 0 4px rgba(56, 189, 248, 0.12),
    inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.player-modal__meta {
  margin-top: 0.65rem;
  color: rgba(203, 213, 225, 0.62);
}

.player-modal__action {
  flex: 1;
  justify-content: center;
  border-radius: 1rem;
  padding: 0.95rem 1rem;
}

.player-modal__action--primary {
  border: 1px solid rgba(96, 165, 250, 0.24);
  background: linear-gradient(135deg, #2563eb 0%, #0ea5e9 100%);
  color: #ffffff;
}

.player-modal__action--danger {
  border: 1px solid rgba(248, 113, 113, 0.2);
  background: rgba(239, 68, 68, 0.12);
  color: #fee2e2;
}

.player-modal__close:hover,
.player-modal__action:hover {
  transform: translateY(-1px);
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

@media (max-width: 1279px) {
  .player-track-card,
  .player-utility-card {
    padding: 0.15rem 0;
  }
}

@media (max-width: 767px) {
  .player-track-card {
    flex-direction: row;
    align-items: flex-start;
    gap: 0.68rem;
  }

  .player-track-card__cover {
    height: 2.85rem;
    width: 2.85rem;
    border-radius: 0.82rem;
    font-size: 0.94rem;
  }

  .player-fact-chip {
    display: none;
  }

  .player-transport-card__header {
    flex-wrap: nowrap;
    justify-content: flex-start;
    overflow-x: auto;
    padding-bottom: 0.2rem;
    scrollbar-width: none;
  }

  .player-transport-card__header::-webkit-scrollbar {
    display: none;
  }

  .player-transport-card__core {
    justify-content: stretch;
    gap: 0.45rem;
  }

  .player-nav-button,
  .player-play-button {
    flex: 1;
    min-width: 0;
  }

  .player-nav-button {
    padding: 0.48rem 0.62rem;
  }

  .player-play-button {
    min-height: 2.6rem;
    min-width: 5.1rem;
  }

  .player-volume-shell {
    width: 100%;
    min-width: 0;
    gap: 0.42rem;
    padding: 0.38rem 0.46rem;
  }

  .player-utility-card__row {
    flex-direction: row;
    align-items: center;
    gap: 0.5rem;
  }

  .player-volume-button {
    min-width: 2.9rem;
    padding-inline: 0.52rem;
  }

  .player-volume-range {
    min-width: 4.5rem;
  }

  .player-volume-value {
    min-width: 2.65rem;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
  }
}

@media (max-width: 639px) {
  .player-dock {
    border-radius: 1.35rem;
  }

  .player-meta-pill {
    font-size: 0.58rem;
    letter-spacing: 0.12em;
  }

  .player-chip-button {
    min-height: 1.9rem;
    padding: 0.38rem 0.62rem;
    font-size: 0.62rem;
  }

  .player-transport-card__footer {
    font-size: 0.68rem;
    letter-spacing: 0.05em;
  }
}

@media (min-width: 1280px) {
  .player-transport-card,
  .player-utility-card {
    border-left: 1px solid rgba(148, 163, 184, 0.1);
    padding-left: 0.8rem;
  }
}
</style>
