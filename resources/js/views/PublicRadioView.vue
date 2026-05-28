<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

type RadioStation = {
  slug: string
  name: string
  description: string
  public_url: string
  embed_url: string
}

type RadioTrack = {
  track_id: number
  title: string
  artist: string | null
  duration: number
  started_at: string
  ends_at: string
  offset_seconds: number
  remaining_seconds: number
  stream_url: string
}

type RadioSnapshot = {
  station: RadioStation
  now_playing: RadioTrack | null
  up_next: RadioTrack[]
  history: RadioTrack[]
  generated_at: string
}

const route = useRoute()
const station = ref<RadioSnapshot | null>(null)
const loading = ref(true)
const error = ref('')
const playbackBlocked = ref(false)
const audioRef = ref<HTMLAudioElement | null>(null)
const pendingSeekSeconds = ref<number | null>(null)

let refreshTimer: number | null = null

const slug = computed(() => String(route.params.slug ?? 'test'))
const isEmbed = computed(() => Boolean(route.meta.embed))
const nowPlaying = computed(() => station.value?.now_playing ?? null)
const progressPercent = computed(() => {
  const current = nowPlaying.value
  if (!current || current.duration <= 0) return 0
  return Math.min(100, Math.max(0, (current.offset_seconds / current.duration) * 100))
})

async function fetchStation() {
  loading.value = true

  try {
    const response = await fetch(`/api/public/radio/${slug.value}`, {
      headers: { Accept: 'application/json' },
    })

    if (!response.ok) {
      const payload = await response.json().catch(() => null)
      throw new Error(payload?.message || 'Failed to load radio station')
    }

    station.value = await response.json() as RadioSnapshot
    error.value = ''
  } catch (err) {
    error.value = err instanceof Error ? err.message : 'Failed to load radio station'
  } finally {
    loading.value = false
  }
}

async function attemptPlayback() {
  if (!audioRef.value) return

  try {
    await audioRef.value.play()
    playbackBlocked.value = false
  } catch {
    playbackBlocked.value = true
  }
}

function syncAudioToStation() {
  if (!audioRef.value || !nowPlaying.value) return

  const audio = audioRef.value
  const current = nowPlaying.value
  const desiredSrc = new URL(current.stream_url, window.location.origin).toString()
  const currentTrackId = Number(audio.dataset.trackId || 0)

  if (currentTrackId !== current.track_id || audio.src !== desiredSrc) {
    audio.dataset.trackId = String(current.track_id)
    pendingSeekSeconds.value = current.offset_seconds
    audio.src = current.stream_url
    audio.load()
    return
  }

  const driftSeconds = Math.abs(audio.currentTime - current.offset_seconds)
  if (driftSeconds > 3) {
    audio.currentTime = current.offset_seconds
  }

  if (audio.paused && !playbackBlocked.value) {
    void attemptPlayback()
  }
}

function handleLoadedMetadata() {
  if (!audioRef.value) return

  if (pendingSeekSeconds.value !== null) {
    const maxTime = Math.max(0, (audioRef.value.duration || pendingSeekSeconds.value) - 1)
    audioRef.value.currentTime = Math.min(pendingSeekSeconds.value, maxTime)
    pendingSeekSeconds.value = null
  }

  void attemptPlayback()
}

function handleEnded() {
  void fetchStation()
}

function formatTime(totalSeconds: number): string {
  if (!Number.isFinite(totalSeconds) || totalSeconds <= 0) return '0:00'
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = Math.floor(totalSeconds % 60)
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

function startRefreshLoop() {
  if (refreshTimer !== null) {
    window.clearInterval(refreshTimer)
  }

  refreshTimer = window.setInterval(() => {
    void fetchStation()
  }, 15000)
}

watch(() => route.params.slug, () => {
  station.value = null
  playbackBlocked.value = false
  void fetchStation()
})

watch(nowPlaying, () => {
  syncAudioToStation()
})

onMounted(() => {
  void fetchStation()
  startRefreshLoop()
})

onBeforeUnmount(() => {
  if (refreshTimer !== null) {
    window.clearInterval(refreshTimer)
  }
})
</script>

<template>
  <div :class="isEmbed ? 'min-h-screen bg-transparent p-4' : 'min-h-screen bg-[radial-gradient(circle_at_top,_rgba(125,211,252,0.28),_transparent_42%),linear-gradient(180deg,#eff6ff_0%,#f8fafc_48%,#e2e8f0_100%)] p-4 md:p-8'">
    <div :class="isEmbed ? 'mx-auto max-w-3xl' : 'w-full'">
      <section :class="isEmbed ? 'surface-card !rounded-[1.8rem] !p-5' : 'surface-card overflow-hidden !rounded-[2rem] !p-0'">
        <div :class="isEmbed ? 'flex flex-col gap-5' : 'border-b border-slate-200/80 bg-slate-50/80 px-6 py-6 md:px-8'">
          <div class="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <p class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Public Radio</p>
              <h1 class="mt-2 text-3xl font-semibold text-slate-950">
                {{ station?.station.name ?? 'Loading station' }}
              </h1>
              <p class="mt-3 max-w-2xl text-sm leading-6 text-slate-600">
                {{ station?.station.description ?? 'Shared station link backed by server-side schedule state.' }}
              </p>
            </div>

            <div v-if="station && !isEmbed" class="grid gap-3 text-sm text-slate-600">
              <div class="rounded-[1.2rem] border border-slate-200 bg-white/80 px-4 py-3">
                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Public URL</div>
                <div class="mt-2 break-all font-medium text-slate-950">{{ station.station.public_url }}</div>
              </div>
              <div class="rounded-[1.2rem] border border-slate-200 bg-white/80 px-4 py-3">
                <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Embed URL</div>
                <div class="mt-2 break-all font-medium text-slate-950">{{ station.station.embed_url }}</div>
              </div>
            </div>
          </div>
        </div>

        <div :class="isEmbed ? 'grid gap-5' : 'grid gap-6 px-6 py-6 md:grid-cols-[minmax(0,1.4fr)_minmax(22rem,0.9fr)] md:px-8 md:py-8'">
          <div class="grid gap-5">
            <div class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5 shadow-[0_20px_60px_rgba(15,23,42,0.08)]">
              <div class="flex items-center justify-between gap-4">
                <div>
                  <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Now Playing</div>
                  <div class="mt-2 text-2xl font-semibold text-slate-950">
                    {{ nowPlaying?.title ?? 'Waiting for station queue' }}
                  </div>
                  <div class="mt-2 text-sm text-slate-600">
                    {{ nowPlaying?.artist || 'Unknown artist' }}
                  </div>
                </div>

                <div class="rounded-[1.2rem] bg-sky-100 px-4 py-3 text-right">
                  <div class="text-xs font-semibold uppercase tracking-[0.18em] text-sky-700">Shared Offset</div>
                  <div class="mt-2 text-lg font-semibold text-sky-900">
                    {{ formatTime(nowPlaying?.offset_seconds ?? 0) }}
                  </div>
                </div>
              </div>

              <div class="mt-5">
                <div class="h-2 overflow-hidden rounded-full bg-slate-200">
                  <div class="h-full rounded-full bg-sky-500 transition-[width] duration-300" :style="{ width: `${progressPercent}%` }"></div>
                </div>
                <div class="mt-2 flex items-center justify-between text-xs font-medium text-slate-500">
                  <span>{{ formatTime(nowPlaying?.offset_seconds ?? 0) }}</span>
                  <span>{{ formatTime(nowPlaying?.duration ?? 0) }}</span>
                </div>
              </div>

              <audio
                ref="audioRef"
                class="mt-5 w-full"
                controls
                preload="auto"
                @loadedmetadata="handleLoadedMetadata"
                @ended="handleEnded"
              />

              <div class="mt-4 flex flex-wrap items-center gap-3 text-sm">
                <button v-if="playbackBlocked" type="button" class="action-primary" @click="attemptPlayback">
                  Start radio
                </button>
                <span class="badge-soft">
                  {{ nowPlaying ? `${formatTime(nowPlaying.remaining_seconds)} remaining` : 'Awaiting next track' }}
                </span>
              </div>
            </div>

            <div v-if="error" class="rounded-[1.5rem] border border-rose-200 bg-rose-50/80 px-5 py-4 text-sm font-medium text-rose-700">
              {{ error }}
            </div>

            <div v-else-if="loading && !station" class="rounded-[1.5rem] border border-slate-200 bg-white/80 px-5 py-4 text-sm text-slate-600">
              Loading shared station state...
            </div>
          </div>

          <div v-if="!isEmbed" class="grid gap-5">
            <section class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5">
              <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Up Next</div>
              <div v-if="station?.up_next.length" class="mt-4 grid gap-3">
                <article v-for="track in station.up_next" :key="`next-${track.track_id}-${track.started_at}`" class="rounded-[1.15rem] border border-slate-200 bg-slate-50/85 px-4 py-3">
                  <div class="font-semibold text-slate-950">{{ track.title }}</div>
                  <div class="mt-1 text-sm text-slate-500">{{ track.artist || 'Unknown artist' }}</div>
                  <div class="mt-2 text-xs font-medium uppercase tracking-[0.14em] text-slate-400">
                    starts {{ new Date(track.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }}
                  </div>
                </article>
              </div>
              <p v-else class="mt-4 text-sm text-slate-500">No tracks queued yet.</p>
            </section>

            <section class="rounded-[1.5rem] border border-slate-200 bg-white/85 p-5">
              <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Recent History</div>
              <div v-if="station?.history.length" class="mt-4 grid gap-3">
                <article v-for="track in station.history" :key="`history-${track.track_id}-${track.started_at}`" class="rounded-[1.15rem] border border-slate-200 bg-slate-50/85 px-4 py-3">
                  <div class="font-semibold text-slate-950">{{ track.title }}</div>
                  <div class="mt-1 text-sm text-slate-500">{{ track.artist || 'Unknown artist' }}</div>
                  <div class="mt-2 text-xs font-medium uppercase tracking-[0.14em] text-slate-400">
                    played at {{ new Date(track.started_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }}
                  </div>
                </article>
              </div>
              <p v-else class="mt-4 text-sm text-slate-500">History will appear after the station advances.</p>
            </section>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>
