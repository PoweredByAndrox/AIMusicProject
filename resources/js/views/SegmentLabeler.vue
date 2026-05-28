<script setup lang="ts">
/**
 * SegmentLabeler - UI for labeling track segments
 * Hybrid approach: auto-detected segments + manual adjustment
 * Now with real-time segment highlighting and quick rating
 * Synced with global player
 */
import { ref, onMounted, onUnmounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api from '@/api/client'
import WaveformWithFragments from '@/components/WaveformWithFragments.vue'
import { usePlayerStore } from '@/stores/playerStore'
import type { Track, Fragment } from '@/types'

interface Segment {
  id: number
  start: number
  end: number
  type: string
  display_label?: string
  importance: number
  is_liked: boolean | null
  rating: number | null
}

interface TrackFeedbackSnapshot {
  track_id: number
  track_rating?: number | null
  explicit_track_rating?: number | null
  computed_track_rating?: number | null
  preference_score?: number | null
  sentiment?: string | null
  fragments?: Fragment[]
  segments?: Segment[]
}

const WAVEFORM_SEGMENT_ID_OFFSET = 1000000000

const route = useRoute()
const router = useRouter()
const playerStore = usePlayerStore()

const track = ref<Track | null>(null)
const segments = ref<Segment[]>([])
const savedFragments = ref<Fragment[]>([])
const feedback = ref<TrackFeedbackSnapshot | null>(null)
const loading = ref(true)
const saving = ref(false)
const analyzing = ref(false)
const analysisStatus = ref<{
  status: string
  progress: number
  message: string
  segments_created?: number
} | null>(null)
let statusPollInterval: ReturnType<typeof setInterval> | null = null

// Playback tracking - sync with global player
const waveformRef = ref<InstanceType<typeof WaveformWithFragments> | null>(null)

const trackId = computed(() => Number(route.params.id))

// Check if this track is playing in global player
const isTrackInGlobalPlayer = computed(() => {
  return playerStore.currentTrack?.id === trackId.value
})

const waveformLocalTime = ref(0)
const currentTime = computed(() => {
  return isTrackInGlobalPlayer.value ? playerStore.currentTime : waveformLocalTime.value
})
const isPlaying = computed(() => {
  return isTrackInGlobalPlayer.value && playerStore.isPlaying
})

// Current segment being played
const currentSegment = computed(() => {
  if (!isPlaying.value || segments.value.length === 0) return null
  return segments.value.find(seg => 
    currentTime.value >= seg.start && currentTime.value < seg.end
  ) || null
})

// Handle time updates from local waveform
const onTimeUpdate = (time: number) => {
  waveformLocalTime.value = time
}

// When play is requested from waveform - load track into global player
const onPlayRequested = () => {
  if (track.value) {
    // Load track into global player if not already there
    if (!isTrackInGlobalPlayer.value) {
      playerStore.playTrack(track.value, false)
    } else {
      playerStore.play()
    }
  }
}

// When seek is requested from waveform - sync with global player
const onSeekRequested = (time: number) => {
  if (!track.value) return

  if (isTrackInGlobalPlayer.value) {
    playerStore.seek(time)
    return
  }

  playerStore.playTrack(track.value, false)
  setTimeout(() => {
    if (playerStore.currentTrack?.id !== track.value?.id) return
    playerStore.seek(time)
  }, 100)
}

const onPauseRequested = () => {
  playerStore.pause()
}

// Play a specific segment through the global player only
const playSegment = (segment: Segment) => {
  if (!track.value) return

  if (isTrackInGlobalPlayer.value) {
    playerStore.seek(segment.start)
    playerStore.play()
    return
  }

  playerStore.playTrack(track.value, false)
  setTimeout(() => {
    if (playerStore.currentTrack?.id !== track.value?.id) return
    playerStore.seek(segment.start)
    playerStore.play()
  }, 100)
}

// Load track into global player
const loadInGlobalPlayer = () => {
  if (track.value && !isTrackInGlobalPlayer.value) {
    playerStore.playTrack(track.value, false)
  }
}

// Scroll to current segment
const scrollToSegment = (segmentId: number) => {
  const el = document.getElementById(`segment-${segmentId}`)
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

// Watch for segment changes to auto-scroll
watch(currentSegment, (newSeg, oldSeg) => {
  if (newSeg && newSeg.id !== oldSeg?.id) {
    scrollToSegment(newSeg.id)
  }
})

// Sync waveform position with global player
watch(() => playerStore.currentTime, (time) => {
  if (isTrackInGlobalPlayer.value && waveformRef.value && Math.abs(waveformLocalTime.value - time) > 0.5) {
    waveformRef.value.syncSeekTo(time)
  }
})

// Sync play/pause state with global player
watch(() => playerStore.isPlaying, (playing) => {
  if (isTrackInGlobalPlayer.value && waveformRef.value) {
    if (playing) {
      waveformRef.value.syncPlay()
    } else {
      waveformRef.value.syncPause()
    }
  }
})

watch(() => playerStore.currentTrack?.id, (currentTrackId) => {
  if (!waveformRef.value) return

  if (currentTrackId === trackId.value) {
    waveformRef.value.syncSeekTo(playerStore.currentTime)
    return
  }

  waveformRef.value.syncPause()
})

// Load track and segments
const loadTrack = async () => {
  loading.value = true
  try {
    const response = await api.getTrack(trackId.value)
    track.value = response
    
    // Load auto-detected segments from patterns
    await loadSegments()
  } catch (error) {
    console.error('Failed to load track:', error)
  } finally {
    loading.value = false
  }
}

const loadSegments = async () => {
  try {
    const data = await api.getTrackFeedbackSnapshot(trackId.value)
    applyFeedbackSnapshot(data)
  } catch (error) {
    console.error('Failed to load segments:', error)
    feedback.value = null
    segments.value = []
  }
}

const applyFeedbackSnapshot = (snapshot: TrackFeedbackSnapshot | null | undefined) => {
  feedback.value = snapshot ?? null
  segments.value = Array.isArray(snapshot?.segments) ? snapshot.segments : []
  savedFragments.value = Array.isArray(snapshot?.fragments) ? snapshot.fragments : []
}

// Analyze track to get segments
const analyzeTrack = async () => {
  analyzing.value = true
  analysisStatus.value = { status: 'queued', progress: 0, message: 'Starting analysis...' }
  
  try {
    await api.analyzeTrackSegments(trackId.value)
    startStatusPolling()
  } catch (error) {
    console.error('Failed to analyze:', error)
    analysisStatus.value = { status: 'failed', progress: 0, message: 'Failed to start analysis' }
    analyzing.value = false
  }
}

// Poll for analysis status
const startStatusPolling = () => {
  if (statusPollInterval) {
    clearInterval(statusPollInterval)
  }
  
  statusPollInterval = setInterval(async () => {
    try {
      const data = await api.getTrackSegmentAnalysisStatus(trackId.value)
      analysisStatus.value = data

      if (data.status === 'completed' || data.status === 'failed' || data.status === 'idle') {
        stopStatusPolling()
        analyzing.value = false

        if (data.status === 'completed') {
          await loadSegments()
        }
      }
    } catch (error) {
      console.error('Failed to get status:', error)
    }
  }, 1000) // Poll every second
}

const stopStatusPolling = () => {
  if (statusPollInterval) {
    clearInterval(statusPollInterval)
    statusPollInterval = null
  }
}

// Rate a segment with stars (0-5, step 0.5)
const rateSegment = async (segment: Segment, rating: number) => {
  segment.rating = rating
  saving.value = true
  try {
    const response = await api.rateSegment(segment.id, rating)
    if (response.feedback) {
      applyFeedbackSnapshot(response.feedback)
    } else {
      feedback.value = {
        ...(feedback.value ?? { track_id: trackId.value }),
        track_id: trackId.value,
        track_rating: typeof response.track_rating === 'number' ? response.track_rating : (feedback.value?.track_rating ?? null),
        computed_track_rating: typeof response.computed_track_rating === 'number' ? response.computed_track_rating : (feedback.value?.computed_track_rating ?? null),
        sentiment: response.sentiment ?? feedback.value?.sentiment ?? null,
        segments: segments.value,
        fragments: savedFragments.value,
      }
    }
    window.dispatchEvent(new CustomEvent('track-feedback-updated', {
      detail: { trackId: trackId.value },
    }))
  } catch (error) {
    console.error('Failed to rate segment:', error)
  } finally {
    saving.value = false
  }
}

const getSegmentDisplayLabel = (segment: Segment): string => {
  return segment.display_label || segment.type.replace(/_/g, ' ').toUpperCase()
}

// Convert segments to fragments for waveform display
const fragmentsForWaveform = computed(() => {
  const autoSegments = segments.value.map(seg => ({
    id: seg.id,
    track_id: trackId.value,
    start_time: seg.start,
    end_time: seg.end,
    is_liked: seg.is_liked,
    name: getSegmentDisplayLabel(seg),
    comment: `${getSegmentDisplayLabel(seg)} (${formatTime(seg.start)} - ${formatTime(seg.end)})`
  }))

  const merged = [
    ...savedFragments.value,
    ...autoSegments.map((segment) => ({
      ...segment,
      id: segment.id + WAVEFORM_SEGMENT_ID_OFFSET,
    })),
  ]

  return merged.sort((left, right) => left.start_time - right.start_time)
})

// Handle fragment creation from waveform
const onFragmentCreated = async (data: { start_time: number; end_time: number; is_liked: boolean }) => {
  saving.value = true
  try {
    await api.createFragment(trackId.value, {
      start_time: data.start_time,
      end_time: data.end_time,
      is_liked: data.is_liked
    })
    // Reload segments
    await loadSegments()
  } catch (error) {
    console.error('Failed to create fragment:', error)
  } finally {
    saving.value = false
  }
}

// Format time
const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// Get segment type color
const getTypeColor = (type: string): string => {
  const colors: Record<string, string> = {
    'drop_high': 'bg-red-500',
    'drop_bass': 'bg-orange-500',
    'melodic': 'bg-blue-500',
    'verse': 'bg-gray-500',
    'buildup': 'bg-yellow-500',
    'break': 'bg-purple-500',
    'ambient': 'bg-teal-500',
    'intro': 'bg-gray-400',
    'outro': 'bg-gray-400',
  }
  return colors[type] || 'bg-gray-500'
}

// Stats
const stats = computed(() => {
  const rated = segments.value.filter(s => s.rating !== null).length
  const unrated = segments.value.filter(s => s.rating === null).length
  const avgRating = rated > 0 
    ? segments.value.filter(s => s.rating !== null).reduce((sum, s) => sum + (s.rating || 0), 0) / rated 
    : 0
  return { rated, unrated, total: segments.value.length, avgRating }
})

const playbackProgress = computed(() => {
  if (!track.value?.duration) return 0
  return Math.max(0, Math.min(100, (currentTime.value / track.value.duration) * 100))
})

const currentSegmentIndex = computed(() => {
  if (!currentSegment.value) return -1
  return segments.value.findIndex((segment) => segment.id === currentSegment.value?.id)
})

const averageSegmentDuration = computed(() => {
  if (segments.value.length === 0) return 0
  const totalDuration = segments.value.reduce((sum, segment) => sum + (segment.end - segment.start), 0)
  return totalDuration / segments.value.length
})

// Calculate overall track rating from segments
const trackRating = computed(() => {
  if (typeof feedback.value?.track_rating === 'number') {
    return feedback.value.track_rating
  }

  const ratedSegments = segments.value.filter(s => s.rating !== null)
  if (ratedSegments.length === 0) return null
  
  // Weighted average: higher importance segments count more
  let totalWeight = 0
  let weightedSum = 0
  for (const seg of ratedSegments) {
    const weight = seg.importance || 1
    weightedSum += (seg.rating || 0) * weight
    totalWeight += weight
  }
  return totalWeight > 0 ? weightedSum / totalWeight : null
})

const computedTrackRating = computed(() => (
  typeof feedback.value?.computed_track_rating === 'number'
    ? feedback.value.computed_track_rating
    : trackRating.value
))

const preferenceScore = computed(() => (
  typeof feedback.value?.preference_score === 'number'
    ? Math.round(feedback.value.preference_score * 100)
    : null
))

const getImportanceLabel = (importance: number): string => {
  return `Impact ${importance || 1}`
}

const getImportanceClasses = (importance: number): string => {
  if (importance >= 3) return 'bg-amber-100 text-amber-800 border border-amber-200'
  if (importance === 2) return 'bg-sky-100 text-sky-800 border border-sky-200'
  return 'bg-slate-100 text-slate-700 border border-slate-200'
}

const getSegmentTimelineStyle = (segment: Segment) => {
  const totalDuration = track.value?.duration || 0
  if (totalDuration <= 0) {
    return {
      marginLeft: '0%',
      width: '0%',
    }
  }

  const start = Math.max(0, Math.min(100, (segment.start / totalDuration) * 100))
  const width = Math.max(1.5, Math.min(100 - start, ((segment.end - segment.start) / totalDuration) * 100))

  return {
    marginLeft: `${start}%`,
    width: `${width}%`,
  }
}

onMounted(() => {
  loadTrack()
})

onUnmounted(() => {
  stopStatusPolling()
})

watch(() => route.params.id, () => {
  stopStatusPolling()
  loadTrack()
})
</script>

<template>
  <div class="grid min-w-0 gap-6 xl:grid-cols-[0.92fr_1.18fr]">
    <section class="relative min-w-0 overflow-hidden rounded-[2rem] bg-gradient-to-br from-slate-950 via-sky-900 to-cyan-600 p-8 text-white shadow-2xl lg:p-10">
      <div class="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.28),transparent_32%),radial-gradient(circle_at_bottom_left,rgba(34,211,238,0.28),transparent_28%)]"></div>
      <div class="relative space-y-6">
        <button
          @click="router.back()"
          class="inline-flex items-center gap-2 text-sm font-medium text-cyan-100/80 transition hover:text-white"
        >
          <span>←</span>
          <span>Back to track</span>
        </button>

        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.35em] text-cyan-100/80">Segment Workspace</p>
          <h1 class="mt-6 text-4xl font-semibold tracking-tight lg:text-5xl">
            {{ track ? `${track.artist || 'Unknown'} - ${track.title}` : 'Segment workspace' }}
          </h1>
          <p class="mt-5 max-w-2xl text-sm leading-6 text-sky-50/85 lg:text-base">
            Review the section map, keep playback aligned with the main player, and correct weak detections directly from the waveform.
          </p>
        </div>

        <div v-if="track" class="grid gap-3 sm:grid-cols-2">
          <div class="rounded-2xl border border-white/15 bg-white/10 px-4 py-4 backdrop-blur">
            <p class="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-100/70">Duration</p>
            <p class="mt-3 text-2xl font-semibold text-white">{{ formatTime(track.duration) }}</p>
          </div>
          <div class="rounded-2xl border border-white/15 bg-white/10 px-4 py-4 backdrop-blur">
            <p class="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-100/70">Detected sections</p>
            <p class="mt-3 text-2xl font-semibold text-white">{{ segments.length }}</p>
          </div>
          <div class="rounded-2xl border border-white/15 bg-white/10 px-4 py-4 backdrop-blur">
            <p class="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-100/70">Track score</p>
            <p class="mt-3 text-2xl font-semibold text-white">
              {{ trackRating !== null ? `${trackRating.toFixed(1)} ★` : '--' }}
            </p>
            <p class="mt-1 text-sm text-cyan-50/80">
              Computed: {{ computedTrackRating !== null ? `${computedTrackRating.toFixed(1)} ★` : '--' }}
            </p>
          </div>
          <div class="rounded-2xl border border-white/15 bg-white/10 px-4 py-4 backdrop-blur">
            <p class="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-100/70">Fit / average</p>
            <p class="mt-3 text-2xl font-semibold text-white">
              {{ preferenceScore !== null ? `${preferenceScore}%` : '--' }}
            </p>
            <p class="mt-1 text-sm text-cyan-50/80">
              Avg section: {{ averageSegmentDuration > 0 ? `${averageSegmentDuration.toFixed(1)}s` : '--' }}
            </p>
          </div>
        </div>

        <div class="rounded-2xl border border-white/15 bg-white/10 px-5 py-5 backdrop-blur">
          <div class="flex items-start justify-between gap-4">
            <div>
              <p class="text-xs font-semibold uppercase tracking-[0.28em] text-cyan-100/70">Current section</p>
              <template v-if="currentSegment">
                <h2 class="mt-3 text-2xl font-semibold text-white">
                  {{ getSegmentDisplayLabel(currentSegment) }}
                </h2>
                <p class="mt-2 text-sm text-cyan-50/85">
                  {{ formatTime(currentSegment.start) }} - {{ formatTime(currentSegment.end) }}
                  / {{ (currentSegment.end - currentSegment.start).toFixed(1) }}s
                </p>
                <div class="mt-4 flex flex-wrap gap-2">
                  <span
                    class="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium"
                    :class="getImportanceClasses(currentSegment.importance)"
                  >
                    {{ getImportanceLabel(currentSegment.importance) }}
                  </span>
                  <span class="inline-flex items-center rounded-full border border-white/15 bg-white/10 px-3 py-1 text-xs font-medium text-white">
                    {{ currentSegment.rating !== null ? `${currentSegment.rating.toFixed(1)} ★` : 'Unrated' }}
                  </span>
                </div>
              </template>
              <template v-else>
                <h2 class="mt-3 text-2xl font-semibold text-white">Waiting for playback</h2>
                <p class="mt-2 text-sm text-cyan-50/85">
                  Start playback from the waveform or open the track in the main player to follow the active section here.
                </p>
              </template>
            </div>

            <div class="rounded-full border border-white/15 bg-white/10 px-4 py-2 text-sm font-medium text-white">
              {{ stats.rated }} rated / {{ stats.unrated }} pending
            </div>
          </div>
        </div>

        <div class="flex flex-col gap-3 sm:flex-row">
          <button
            v-if="!isTrackInGlobalPlayer"
            @click="loadInGlobalPlayer"
            class="rounded-full border border-white/20 bg-white/10 px-5 py-3 text-sm font-semibold text-white transition hover:bg-white/15"
          >
            Sync with player
          </button>
          <div
            v-else
            class="inline-flex items-center justify-center gap-2 rounded-full border border-emerald-200/30 bg-emerald-50/10 px-5 py-3 text-sm font-semibold text-emerald-50"
          >
            <span class="h-2 w-2 rounded-full bg-emerald-300"></span>
            <span>Synced with player</span>
          </div>
          <button
            @click="analyzeTrack"
            :disabled="analyzing"
            class="rounded-full bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {{ analyzing ? 'Analyzing...' : 'Re-analyze sections' }}
          </button>
        </div>

        <Transition name="fade">
          <div
            v-if="analyzing && analysisStatus"
            class="rounded-2xl border border-amber-200/30 bg-amber-50/10 px-4 py-4 text-sm text-amber-50 backdrop-blur"
          >
            <div class="flex items-center justify-between gap-4">
              <div>
                <p class="font-semibold">Rebuilding section map</p>
                <p class="mt-1 text-amber-50/90">{{ analysisStatus.message }}</p>
              </div>
              <div class="text-right text-lg font-semibold">{{ analysisStatus.progress }}%</div>
            </div>
            <div class="mt-3 h-2 overflow-hidden rounded-full bg-black/20">
              <div
                class="h-full rounded-full bg-white/80 transition-all duration-300"
                :style="{ width: `${analysisStatus.progress}%` }"
              ></div>
            </div>
          </div>
        </Transition>
      </div>
    </section>

    <section class="min-w-0 rounded-[2rem] border border-slate-200 bg-white p-8 shadow-xl lg:p-10">
      <div v-if="loading" class="py-20 text-center text-slate-500">
        Loading track workspace...
      </div>

      <template v-else-if="track">
        <div class="flex flex-col gap-6">
          <div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <p class="text-sm font-semibold uppercase tracking-[0.28em] text-sky-700">Waveform Studio</p>
              <h2 class="mt-3 text-3xl font-semibold tracking-tight text-slate-900">Playback and section review</h2>
              <p class="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
                Drag on the waveform to create a custom section, click any detected section to jump there, and keep ratings tied to the current feedback snapshot.
              </p>
            </div>

            <div class="rounded-2xl bg-slate-50 px-4 py-4 text-sm text-slate-600">
              <div class="flex items-center justify-between gap-6">
                <span class="font-medium text-slate-700">Playback progress</span>
                <span class="font-semibold text-slate-900">{{ Math.round(playbackProgress) }}%</span>
              </div>
              <div class="mt-3 h-2 overflow-hidden rounded-full bg-slate-200">
                <div
                  class="h-full rounded-full bg-sky-600 transition-all duration-200"
                  :style="{ width: `${playbackProgress}%` }"
                ></div>
              </div>
              <div class="mt-3 flex items-center justify-between text-xs uppercase tracking-[0.24em] text-slate-500">
                <span>{{ formatTime(currentTime) }}</span>
                <span>{{ formatTime(track.duration) }}</span>
              </div>
            </div>
          </div>

          <div class="min-w-0 overflow-hidden rounded-[1.75rem] border border-slate-200 bg-slate-50 p-4">
            <WaveformWithFragments
              ref="waveformRef"
              v-if="track.file_path"
              :audio-url="`/api/tracks/${track.id}/stream`"
              :fragments="fragmentsForWaveform"
              :duration="track.duration"
              :track-id="track.id"
              :external-audio="true"
              @fragment-created="onFragmentCreated"
              @time-update="onTimeUpdate"
              @play-requested="onPlayRequested"
              @pause-requested="onPauseRequested"
              @seek="onSeekRequested"
            />
          </div>

          <div class="rounded-2xl bg-slate-50 px-4 py-4 text-sm text-slate-600">
            <p class="font-semibold text-slate-900">How this page works</p>
            <div class="mt-3 grid gap-3 lg:grid-cols-3">
              <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                Click a section card or the waveform to jump playback without leaving the page.
              </div>
              <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                Drag across the waveform to mark a custom section when the auto-detected cut is weak.
              </div>
              <div class="rounded-2xl border border-slate-200 bg-white px-4 py-3">
                Rate sections directly and keep explicit track score versus computed score visible at the same time.
              </div>
            </div>
          </div>

          <div class="grid gap-4 lg:grid-cols-[0.72fr_1.28fr]">
            <div class="rounded-[1.75rem] border border-slate-200 bg-slate-50 p-5">
              <p class="text-sm font-semibold uppercase tracking-[0.28em] text-sky-700">Current Section</p>
              <template v-if="currentSegment">
                <div class="mt-4 space-y-4">
                  <div class="flex items-center gap-2">
                    <span :class="['h-3 w-3 rounded-full', getTypeColor(currentSegment.type)]"></span>
                    <span class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
                      Section {{ currentSegmentIndex + 1 }}
                    </span>
                  </div>
                  <div>
                    <h3 class="text-2xl font-semibold tracking-tight text-slate-900">
                      {{ getSegmentDisplayLabel(currentSegment) }}
                    </h3>
                    <p class="mt-2 text-sm leading-6 text-slate-600">
                      {{ formatTime(currentSegment.start) }} - {{ formatTime(currentSegment.end) }}
                      / {{ (currentSegment.end - currentSegment.start).toFixed(1) }}s
                    </p>
                  </div>
                  <div class="flex flex-wrap gap-2">
                    <span
                      class="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium"
                      :class="getImportanceClasses(currentSegment.importance)"
                    >
                      {{ getImportanceLabel(currentSegment.importance) }}
                    </span>
                    <span class="inline-flex items-center rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-700">
                      {{ currentSegment.rating !== null ? `${currentSegment.rating.toFixed(1)} ★` : 'Unrated' }}
                    </span>
                  </div>
                  <button
                    @click="playSegment(currentSegment)"
                    class="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 transition hover:border-slate-300 hover:bg-white"
                  >
                    Jump to this section
                  </button>
                </div>
              </template>
              <template v-else>
                <div class="mt-4 rounded-2xl border border-dashed border-slate-200 bg-white px-4 py-5 text-sm leading-6 text-slate-600">
                  Start playback from the waveform or the player to surface the active section here.
                </div>
              </template>
            </div>

            <div>
              <div class="flex items-center justify-between gap-4">
                <div>
                  <p class="text-sm font-semibold uppercase tracking-[0.28em] text-sky-700">Detected Sections</p>
                  <h3 class="mt-3 text-3xl font-semibold tracking-tight text-slate-900">{{ segments.length }} sections</h3>
                </div>
                <div class="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600">
                  {{ stats.rated }} rated / {{ stats.unrated }} pending
                </div>
              </div>

              <div v-if="segments.length === 0" class="mt-6 rounded-2xl border border-slate-200 bg-slate-50 px-4 py-5 text-sm leading-6 text-slate-600">
                No sections are stored for this track yet. Re-run analysis to rebuild the section map.
              </div>

              <div v-else class="segment-scroll mt-6 space-y-3 max-h-[70vh] overflow-y-auto pr-1">
                <article
                  v-for="(segment, index) in segments"
                  :key="segment.id"
                  :id="`segment-${segment.id}`"
                  @click="playSegment(segment)"
                  :class="[
                    'rounded-2xl border px-4 py-4 transition-all cursor-pointer',
                    currentSegment?.id === segment.id
                      ? 'border-sky-300 bg-sky-50 shadow-lg'
                      : 'border-slate-200 bg-slate-50 hover:border-slate-300 hover:bg-white'
                  ]"
                >
                  <div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div class="min-w-0 flex-1">
                      <div class="flex flex-wrap items-center gap-2">
                        <span :class="['h-2.5 w-2.5 rounded-full', getTypeColor(segment.type)]"></span>
                        <span class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">
                          Section {{ index + 1 }}
                        </span>
                        <span
                          class="inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-medium"
                          :class="getImportanceClasses(segment.importance)"
                        >
                          {{ getImportanceLabel(segment.importance) }}
                        </span>
                      </div>
                      <h4 class="mt-3 text-xl font-semibold text-slate-900">
                        {{ getSegmentDisplayLabel(segment) }}
                      </h4>
                      <p class="mt-1 text-sm text-slate-600">
                        {{ formatTime(segment.start) }} - {{ formatTime(segment.end) }}
                        / {{ (segment.end - segment.start).toFixed(1) }}s
                      </p>

                      <div class="mt-4 h-1.5 overflow-hidden rounded-full bg-slate-200">
                        <div
                          class="h-full rounded-full bg-sky-600"
                          :style="getSegmentTimelineStyle(segment)"
                        ></div>
                      </div>
                    </div>

                    <div class="flex flex-col items-start gap-3 lg:items-end">
                      <button
                        class="rounded-full border border-slate-200 px-4 py-2 text-xs font-medium text-slate-700 transition hover:border-slate-300 hover:bg-white"
                      >
                        Play section
                      </button>

                      <div class="flex items-center gap-1" @click.stop>
                        <button
                          v-for="star in [1, 2, 3, 4, 5]"
                          :key="star"
                          @click="rateSegment(segment, star)"
                          :disabled="saving"
                          class="text-xl leading-none transition-transform hover:scale-110 disabled:cursor-not-allowed disabled:opacity-50"
                          :class="segment.rating !== null && segment.rating >= star ? 'text-amber-400' : 'text-slate-300 hover:text-amber-300'"
                        >
                          {{ segment.rating !== null && segment.rating >= star ? '★' : '☆' }}
                        </button>
                        <span class="ml-2 min-w-[52px] text-right text-sm font-medium text-slate-700">
                          {{ segment.rating !== null ? segment.rating.toFixed(1) : '--' }}
                        </span>
                      </div>

                      <div class="text-xs font-medium uppercase tracking-[0.2em] text-slate-500">
                        <span v-if="currentSegment?.id === segment.id" class="text-sky-700">Active now</span>
                        <span v-else-if="segment.rating !== null">Rated section</span>
                        <span v-else>Waiting for rating</span>
                      </div>
                    </div>
                  </div>
                </article>
              </div>
            </div>
          </div>
        </div>
      </template>
    </section>
  </div>
</template>


<style scoped>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}

.segment-scroll {
  scroll-behavior: smooth;
}

.segment-scroll::-webkit-scrollbar {
  width: 7px;
}

.segment-scroll::-webkit-scrollbar-track {
  border-radius: 999px;
  background: rgba(15, 23, 42, 0.8);
}

.segment-scroll::-webkit-scrollbar-thumb {
  border-radius: 999px;
  background: linear-gradient(180deg, rgba(56, 189, 248, 0.7), rgba(139, 92, 246, 0.7));
}

.segment-scroll::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, rgba(56, 189, 248, 0.9), rgba(139, 92, 246, 0.9));
}
</style>
