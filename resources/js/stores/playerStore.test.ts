/**
 * Property-Based Tests for Player Store
 * 
 * **Validates: Requirements 1.1, 1.4, 1.5, 2.1, 2.2**
 */

import { describe, it, expect } from 'vitest'
import { setActivePinia, createPinia } from 'pinia'
import fc from 'fast-check'
import { usePlayerStore } from './playerStore'
import type { Track } from '@/types'

// Helper to create a track with a specific ID
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
    bpm: null,
    key: null,
    mood: null,
    has_vocals: null,
    is_acoustic: null,
    is_liked: false,
    manually_edited: false,
    embedding_path: null,
    analysis_status: 'completed',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  }
}

// Arbitrary for generating an array of tracks with unique IDs
const uniqueTracksArbitrary = (minLength: number, maxLength: number): fc.Arbitrary<Track[]> =>
  fc.uniqueArray(fc.integer({ min: 1, max: 100000 }), { minLength, maxLength })
    .map((ids: number[]) => ids.map((id: number) => createTrack(id)))

// Arbitrary for a single track
const singleTrackArbitrary = (): fc.Arbitrary<Track> =>
  fc.integer({ min: 1, max: 100000 }).map((id: number) => createTrack(id))

// Helper to create a fresh store for each property iteration
function createFreshStore() {
  setActivePinia(createPinia())
  return usePlayerStore()
}

describe('Player Store - Property Tests', () => {
  /**
   * Property 1: Инварианты очереди воспроизведения
   * **Validates: Requirements 1.1, 1.4, 1.5**
   */
  describe('Property 1: Queue Invariants', () => {
    /**
     * Property 1.1: Adding track to end increases queue length by 1
     * and track is at the last position
     */
    it('adding track to end increases length by 1 and track is at last position', () => {
      fc.assert(
        fc.property(
          uniqueTracksArbitrary(0, 20),
          singleTrackArbitrary(),
          (initialTracks: Track[], newTrack: Track) => {
            const store = createFreshStore()
            
            // Add initial tracks
            initialTracks.forEach((track: Track) => store.addToQueue(track, 'end'))
            const initialLength = store.queue.length
            
            // Add new track
            store.addToQueue(newTrack, 'end')
            
            // Verify: length increased by 1
            expect(store.queue.length).toBe(initialLength + 1)
            
            // Verify: track is at last position
            const lastItem = store.queue[store.queue.length - 1]
            expect(lastItem.track.id).toBe(newTrack.id)
            expect(lastItem.track.title).toBe(newTrack.title)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * Property 1.2: Removing track decreases queue length by 1
     * and track is absent from queue
     */
    it('removing track decreases length by 1 and track is absent', () => {
      fc.assert(
        fc.property(
          uniqueTracksArbitrary(1, 20),
          (tracks: Track[]) => {
            const store = createFreshStore()
            
            // Add all tracks
            tracks.forEach((track: Track) => store.addToQueue(track, 'end'))
            const initialLength = store.queue.length
            
            // Pick a random track to remove
            const indexToRemove = Math.floor(Math.random() * store.queue.length)
            const itemToRemove = store.queue[indexToRemove]
            const itemId = itemToRemove.id
            
            // Remove the track
            store.removeFromQueue(itemId)
            
            // Verify: length decreased by 1
            expect(store.queue.length).toBe(initialLength - 1)
            
            // Verify: track is absent
            const found = store.queue.find((q) => q.id === itemId)
            expect(found).toBeUndefined()
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * Property 1.3: Clearing queue leaves only current track (if playing)
     * or empties queue completely (if not playing)
     */
    it('clearing queue leaves only current track if playing, otherwise empties', () => {
      fc.assert(
        fc.property(
          uniqueTracksArbitrary(2, 20),
          fc.boolean(),
          (tracks: Track[], shouldPlay: boolean) => {
            const store = createFreshStore()
            
            // Add all tracks
            tracks.forEach((track: Track) => store.addToQueue(track, 'end'))
            
            if (shouldPlay && store.queue.length > 0) {
              // Start playing
              store.play()
              const currentTrackBefore = store.currentTrack
              
              // Clear queue
              store.clearQueue()
              
              // Verify: only current track remains
              expect(store.queue.length).toBe(1)
              expect(store.queue[0].track.id).toBe(currentTrackBefore?.id)
            } else {
              // Not playing - stop first
              store.stop()
              
              // Clear queue
              store.clearQueue()
              
              // Verify: queue is empty
              expect(store.queue.length).toBe(0)
            }
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * Property 1.4: Adding track "play next" inserts after current
     */
    it('adding track as "next" inserts after current position', () => {
      fc.assert(
        fc.property(
          uniqueTracksArbitrary(2, 10),
          singleTrackArbitrary(),
          (initialTracks: Track[], newTrack: Track) => {
            const store = createFreshStore()
            
            // Add initial tracks and start playing
            initialTracks.forEach((track: Track) => store.addToQueue(track, 'end'))
            store.play()
            
            const currentIndex = store.currentIndex
            const initialLength = store.queue.length
            
            // Add new track as "next"
            store.addToQueue(newTrack, 'next')
            
            // Verify: length increased by 1
            expect(store.queue.length).toBe(initialLength + 1)
            
            // Verify: new track is at currentIndex + 1
            const insertedItem = store.queue[currentIndex + 1]
            expect(insertedItem.track.id).toBe(newTrack.id)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  /**
   * Property 2: Round-trip для режима Shuffle
   * **Validates: Requirements 2.1, 2.2**
   */
  describe('Property 2: Shuffle Round-trip', () => {
    /**
     * Property 2.1: Shuffle preserves all tracks (only changes order)
     */
    it('shuffle preserves all tracks', () => {
      fc.assert(
        fc.property(
          uniqueTracksArbitrary(2, 20),
          (tracks: Track[]) => {
            const store = createFreshStore()
            
            // Add all tracks
            tracks.forEach((track: Track) => store.addToQueue(track, 'end'))
            
            // Get track IDs before shuffle
            const trackIdsBefore = store.queue.map((q) => q.track.id).sort((a, b) => a - b)
            const lengthBefore = store.queue.length
            
            // Enable shuffle
            store.toggleShuffle()
            
            // Get track IDs after shuffle
            const trackIdsAfter = store.queue.map((q) => q.track.id).sort((a, b) => a - b)
            const lengthAfter = store.queue.length
            
            // Verify: same number of tracks
            expect(lengthAfter).toBe(lengthBefore)
            
            // Verify: same tracks exist (sorted IDs match)
            expect(trackIdsAfter).toEqual(trackIdsBefore)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * Property 2.2: Shuffle then unshuffle restores original order
     */
    it('shuffle then unshuffle restores original order', () => {
      fc.assert(
        fc.property(
          uniqueTracksArbitrary(2, 20),
          (tracks: Track[]) => {
            const store = createFreshStore()
            
            // Add all tracks
            tracks.forEach((track: Track) => store.addToQueue(track, 'end'))
            
            // Record original order (by queue item IDs)
            const originalOrder = store.queue.map((q) => q.id)
            
            // Enable shuffle
            store.toggleShuffle()
            expect(store.shuffleMode).toBe(true)
            
            // Disable shuffle (restore)
            store.toggleShuffle()
            expect(store.shuffleMode).toBe(false)
            
            // Get restored order
            const restoredOrder = store.queue.map((q) => q.id)
            
            // Verify: order is restored
            expect(restoredOrder).toEqual(originalOrder)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    /**
     * Property 2.3: Current track position is maintained during shuffle
     */
    it('current track remains accessible after shuffle', () => {
      fc.assert(
        fc.property(
          uniqueTracksArbitrary(3, 15),
          (tracks: Track[]) => {
            const store = createFreshStore()
            
            // Add all tracks and start playing
            tracks.forEach((track: Track) => store.addToQueue(track, 'end'))
            store.play()
            
            const currentTrackId = store.currentTrack?.id
            
            // Enable shuffle
            store.toggleShuffle()
            
            // Verify: current track is still the same
            expect(store.currentTrack?.id).toBe(currentTrackId)
            
            // Verify: current track is at index 0 after shuffle
            expect(store.currentIndex).toBe(0)
            expect(store.queue[0].track.id).toBe(currentTrackId)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })
  })
})


/**
 * Property 7: Загрузка плейлиста в очередь
 * **Validates: Requirements 5.7**
 */
describe('Property 7: Playlist Loading into Queue', () => {
  /**
   * Property 7.1: Loading playlist adds exactly N tracks to queue
   */
  it('loading playlist adds exactly N tracks to queue', () => {
    fc.assert(
      fc.property(
        uniqueTracksArbitrary(1, 20),
        (tracks: Track[]) => {
          const store = createFreshStore()
          
          // Load playlist
          store.playPlaylist(tracks)
          
          // Verify: queue has exactly N tracks
          expect(store.queue.length).toBe(tracks.length)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 7.2: Playlist order is preserved in queue
   */
  it('playlist order is preserved in queue', () => {
    fc.assert(
      fc.property(
        uniqueTracksArbitrary(2, 15),
        (tracks: Track[]) => {
          const store = createFreshStore()
          
          // Load playlist
          store.playPlaylist(tracks)
          
          // Verify: order is preserved
          for (let i = 0; i < tracks.length; i++) {
            expect(store.queue[i].track.id).toBe(tracks[i].id)
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 7.3: All queue items have source='playlist'
   */
  it('all queue items have source=playlist', () => {
    fc.assert(
      fc.property(
        uniqueTracksArbitrary(1, 10),
        (tracks: Track[]) => {
          const store = createFreshStore()
          
          // Load playlist
          store.playPlaylist(tracks)
          
          // Verify: all items have source='playlist'
          for (const item of store.queue) {
            expect(item.source).toBe('playlist')
          }
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })

  /**
   * Property 7.4: Loading playlist starts playback
   */
  it('loading playlist starts playback at specified index', () => {
    fc.assert(
      fc.property(
        uniqueTracksArbitrary(3, 10),
        fc.integer({ min: 0, max: 9 }),
        (tracks: Track[], startIndex: number) => {
          const store = createFreshStore()
          
          // Clamp startIndex to valid range
          const validStartIndex = Math.min(startIndex, tracks.length - 1)
          
          // Load playlist with start index
          store.playPlaylist(tracks, validStartIndex)
          
          // Verify: current track is at start index
          expect(store.currentIndex).toBe(validStartIndex)
          expect(store.currentTrack?.id).toBe(tracks[validStartIndex].id)
          expect(store.isPlaying).toBe(true)
          
          return true
        }
      ),
      { numRuns: 100 }
    )
  })
})
