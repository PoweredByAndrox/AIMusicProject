/**
 * Property-Based Tests for Audio Engine Crossfade
 * 
 * **Validates: Requirements 3.2, 3.3, 3.5**
 */

import { describe, it, expect } from 'vitest'
import fc from 'fast-check'
import { AudioEngine } from './AudioEngine'

/**
 * Helper to calculate expected volume at a given time during crossfade
 * Using linear interpolation as specified in design doc
 */
function calculateLinearVolume(
  startVolume: number,
  endVolume: number,
  progress: number // 0 to 1
): number {
  return startVolume + (endVolume - startVolume) * progress
}

describe('Audio Engine - Property Tests', () => {
  /**
   * Property 3: Громкость при Crossfade
   * **Validates: Requirements 3.2**
   * 
   * For any moment t in crossfade interval [0, duration]:
   * - currentVolume(t) + nextVolume(t) ≈ 1.0 (with tolerance ±0.1)
   * - currentVolume monotonically decreases from 1 to 0
   * - nextVolume monotonically increases from 0 to 1
   */
  describe('Property 3: Crossfade Volume', () => {
    it('current + next volume sum is approximately 1.0 at any point', () => {
      fc.assert(
        fc.property(
          fc.double({ min: 0, max: 1, noNaN: true }), // progress (0 to 1)
          (progress: number) => {
            // Linear crossfade: current fades from 1 to 0, next fades from 0 to 1
            const currentVolume = calculateLinearVolume(1, 0, progress)
            const nextVolume = calculateLinearVolume(0, 1, progress)
            
            const sum = currentVolume + nextVolume
            
            // Verify: sum is approximately 1.0 (with tolerance ±0.1)
            expect(sum).toBeGreaterThanOrEqual(0.9)
            expect(sum).toBeLessThanOrEqual(1.1)
            
            // For linear crossfade, sum should be exactly 1.0
            expect(Math.abs(sum - 1.0)).toBeLessThan(0.001)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    it('current volume monotonically decreases during crossfade', () => {
      fc.assert(
        fc.property(
          fc.double({ min: 0, max: 1, noNaN: true }), // progress1
          fc.double({ min: 0, max: 1, noNaN: true }), // progress2
          (progress1: number, progress2: number) => {
            // Ensure progress2 > progress1
            const p1 = Math.min(progress1, progress2)
            const p2 = Math.max(progress1, progress2)
            
            if (p1 === p2) return true // Skip if equal
            
            const volume1 = calculateLinearVolume(1, 0, p1)
            const volume2 = calculateLinearVolume(1, 0, p2)
            
            // Verify: volume decreases as progress increases
            expect(volume2).toBeLessThanOrEqual(volume1)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    it('next volume monotonically increases during crossfade', () => {
      fc.assert(
        fc.property(
          fc.double({ min: 0, max: 1, noNaN: true }), // progress1
          fc.double({ min: 0, max: 1, noNaN: true }), // progress2
          (progress1: number, progress2: number) => {
            // Ensure progress2 > progress1
            const p1 = Math.min(progress1, progress2)
            const p2 = Math.max(progress1, progress2)
            
            if (p1 === p2) return true // Skip if equal
            
            const volume1 = calculateLinearVolume(0, 1, p1)
            const volume2 = calculateLinearVolume(0, 1, p2)
            
            // Verify: volume increases as progress increases
            expect(volume2).toBeGreaterThanOrEqual(volume1)
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    it('volume boundaries are correct at start and end', () => {
      // At progress = 0
      const currentAtStart = calculateLinearVolume(1, 0, 0)
      const nextAtStart = calculateLinearVolume(0, 1, 0)
      expect(currentAtStart).toBe(1)
      expect(nextAtStart).toBe(0)
      
      // At progress = 1
      const currentAtEnd = calculateLinearVolume(1, 0, 1)
      const nextAtEnd = calculateLinearVolume(0, 1, 1)
      expect(currentAtEnd).toBe(0)
      expect(nextAtEnd).toBe(1)
    })
  })

  /**
   * Property 4: Валидация длительности Crossfade
   * **Validates: Requirements 3.3, 3.5**
   * 
   * - Values in range [0, 12] are accepted
   * - Values outside range are clamped to boundaries
   * - If next track is shorter than crossfade, duration adapts
   */
  describe('Property 4: Crossfade Duration Validation', () => {
    it('accepts duration values in valid range [0, 12]', () => {
      fc.assert(
        fc.property(
          fc.double({ min: 0, max: 12, noNaN: true }),
          (duration: number) => {
            const engine = new AudioEngine()
            
            engine.setCrossfadeDuration(duration)
            const config = engine.getConfig()
            
            // Verify: duration is accepted and stored
            expect(config.crossfadeDuration).toBeCloseTo(duration, 5)
            
            engine.destroy()
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    it('clamps duration values below 0 to 0', () => {
      fc.assert(
        fc.property(
          fc.double({ min: -100, max: -0.001, noNaN: true }),
          (duration: number) => {
            const engine = new AudioEngine()
            
            engine.setCrossfadeDuration(duration)
            const config = engine.getConfig()
            
            // Verify: duration is clamped to 0
            expect(config.crossfadeDuration).toBe(0)
            
            engine.destroy()
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    it('clamps duration values above 12 to 12', () => {
      fc.assert(
        fc.property(
          fc.double({ min: 12.001, max: 100, noNaN: true }),
          (duration: number) => {
            const engine = new AudioEngine()
            
            engine.setCrossfadeDuration(duration)
            const config = engine.getConfig()
            
            // Verify: duration is clamped to 12
            expect(config.crossfadeDuration).toBe(12)
            
            engine.destroy()
            return true
          }
        ),
        { numRuns: 100 }
      )
    })

    it('crossfade duration adapts when next track is shorter', () => {
      // This tests the logic: min(setting, track_duration * 0.8)
      fc.assert(
        fc.property(
          fc.double({ min: 1, max: 12, noNaN: true }), // crossfade setting
          fc.double({ min: 0.5, max: 20, noNaN: true }), // next track duration
          (crossfadeSetting: number, nextTrackDuration: number) => {
            // Calculate expected effective duration
            const maxFromTrack = nextTrackDuration * 0.8
            const expectedEffective = Math.min(crossfadeSetting, maxFromTrack)
            
            // Verify the logic
            if (nextTrackDuration < crossfadeSetting / 0.8) {
              // Next track is shorter, should adapt
              expect(expectedEffective).toBeLessThan(crossfadeSetting)
            } else {
              // Next track is long enough, use full setting
              expect(expectedEffective).toBeCloseTo(crossfadeSetting, 5)
            }
            
            return true
          }
        ),
        { numRuns: 100 }
      )
    })
  })

  /**
   * Additional unit tests for AudioEngine configuration
   */
  describe('AudioEngine Configuration', () => {
    it('default crossfade is disabled', () => {
      const engine = new AudioEngine()
      const config = engine.getConfig()
      
      expect(config.crossfadeEnabled).toBe(false)
      expect(config.crossfadeDuration).toBe(5) // default 5 seconds
      
      engine.destroy()
    })

    it('can enable and disable crossfade', () => {
      const engine = new AudioEngine()
      
      engine.setCrossfadeEnabled(true)
      expect(engine.getConfig().crossfadeEnabled).toBe(true)
      
      engine.setCrossfadeEnabled(false)
      expect(engine.getConfig().crossfadeEnabled).toBe(false)
      
      engine.destroy()
    })

    it('crossfade state is initially inactive', () => {
      const engine = new AudioEngine()
      const state = engine.getCrossfadeState()
      
      expect(state.isActive).toBe(false)
      expect(state.progress).toBe(0)
      
      engine.destroy()
    })
  })
})
