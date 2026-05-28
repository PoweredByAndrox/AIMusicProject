import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { api } from '@/api/client'
import { usePlayerStore } from './playerStore'
import type { Track } from '@/types'

function createTrack(id: number): Track {
  return {
    id,
    title: `Track ${id}`,
    artist: `Artist ${id}`,
    album: null,
    year: null,
    duration: 180,
    file_path: `/audio/track-${id}.mp3`,
    source: 'upload',
    youtube_url: null,
    genre: null,
    genre_tags: null,
    bpm: null,
    key: null,
    mood: null,
    mood_tags: null,
    has_vocals: null,
    is_acoustic: null,
    loudness_integrated_lufs: null,
    loudness_true_peak_db: null,
    loudness_gain_db: null,
    loudness_target_lufs: null,
    is_liked: false,
    manually_edited: false,
    embedding_path: null,
    analysis_status: 'completed',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  }
}

async function flushPlayerStore() {
  await Promise.resolve()
  await new Promise((resolve) => setTimeout(resolve, 0))
  await Promise.resolve()
}

describe('Player Store - Skip 1-star tracks', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.restoreAllMocks()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ history: { id: 1 } }),
    }))
  })

  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('does not add a 1-star track to the queue when skip is enabled', async () => {
    const store = usePlayerStore()
    vi.spyOn(api, 'getTrackRating').mockResolvedValue({ rating: 1 })

    store.setSkipDislikedSegments(true)
    store.addToQueue(createTrack(101), 'end')

    await flushPlayerStore()

    expect(store.queue).toHaveLength(0)
    expect(store.currentTrack).toBeNull()
  })

  it('does not replace the current track with a blocked 1-star track', async () => {
    const store = usePlayerStore()
    vi.spyOn(api, 'getTrackRating').mockImplementation(async (trackId: number) => ({
      rating: trackId === 202 ? 1 : 5,
    }))

    const safeTrack = createTrack(201)
    const blockedTrack = createTrack(202)

    store.setSkipDislikedSegments(true)
    store.playTrack(safeTrack, false)
    await flushPlayerStore()

    store.playTrack(blockedTrack, false)
    await flushPlayerStore()

    expect(store.currentTrack?.id).toBe(safeTrack.id)
    expect(store.queue.map((item) => item.track.id)).toEqual([safeTrack.id])
  })

  it('filters 1-star tracks out of playlists before playback starts', async () => {
    const store = usePlayerStore()
    vi.spyOn(api, 'getTrackRating').mockImplementation(async (trackId: number) => ({
      rating: trackId === 302 ? 1 : 5,
    }))

    const tracks = [createTrack(301), createTrack(302), createTrack(303)]

    store.setSkipDislikedSegments(true)
    store.playPlaylist(tracks, 1)
    await flushPlayerStore()

    expect(store.queue.map((item) => item.track.id)).toEqual([301, 303])
    expect(store.currentTrack?.id).toBe(303)
    expect(store.currentIndex).toBe(1)
  })

  it('removes an already playing 1-star track when skip is turned on', async () => {
    const store = usePlayerStore()
    const blockedTrack = createTrack(401)
    const safeTrack = createTrack(402)

    store.playTrack(blockedTrack, true)
    store.addToQueue(safeTrack, 'end')
    store.play()
    store.rememberTrackRating(blockedTrack.id, 1)
    store.rememberTrackRating(safeTrack.id, 5)

    store.setSkipDislikedSegments(true)
    await flushPlayerStore()

    expect(store.queue.map((item) => item.track.id)).toEqual([safeTrack.id])
    expect(store.currentTrack?.id).toBe(safeTrack.id)
    expect(store.currentIndex).toBe(0)
  })
})
