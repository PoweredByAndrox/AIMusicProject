/**
 * Beatmatcher - Time-stretching and beat synchronization
 * Requirements: 29.1, 29.2, 29.3, 29.6
 */

export interface BeatmatchConfig {
  maxBpmAdjustment: number // default 8%
  beatGridResolution: number // ms, default 10
  transitionDuration: number // bars, default 16
}

export interface BeatmatchResult {
  success: boolean
  stretchedBuffer?: AudioBuffer
  mixInPoint?: number
  bpmRatio?: number
  warning?: string
}

export interface DJAnalysis {
  trackId: number
  camelotKey: string
  bpm: number
  energyLevel: number
  introPoint: number
  outroPoint: number
}

export class Beatmatcher {
  private config: BeatmatchConfig
  
  constructor(config?: Partial<BeatmatchConfig>) {
    this.config = {
      maxBpmAdjustment: 8,
      beatGridResolution: 10,
      transitionDuration: 16,
      ...config,
    }
  }
  
  /**
   * Prepare beatmatch between two tracks
   */
  async prepareBeatmatch(
    currentTrack: DJAnalysis,
    nextTrack: DJAnalysis,
    nextAudioBuffer: AudioBuffer
  ): Promise<BeatmatchResult> {
    const bpmRatio = currentTrack.bpm / nextTrack.bpm
    const adjustmentPercent = Math.abs(1 - bpmRatio) * 100
    
    // Check if BPM difference is within acceptable range
    if (adjustmentPercent > this.config.maxBpmAdjustment) {
      return {
        success: false,
        warning: `BPM difference too large: ${adjustmentPercent.toFixed(1)}% (max ${this.config.maxBpmAdjustment}%)`,
      }
    }
    
    try {
      // Time-stretch next track to match current BPM
      const stretchedBuffer = await this.timeStretch(nextAudioBuffer, bpmRatio)
      
      // Find optimal mix point (on beat grid)
      const mixInPoint = this.findOptimalMixPoint(
        currentTrack.outroPoint,
        nextTrack.introPoint,
        currentTrack.bpm
      )
      
      return {
        success: true,
        stretchedBuffer,
        mixInPoint,
        bpmRatio,
      }
    } catch (error) {
      return {
        success: false,
        warning: `Beatmatch failed: ${error}`,
      }
    }
  }
  
  /**
   * Time-stretch audio buffer while preserving pitch
   * Uses simple resampling (for production, use phase vocoder)
   */
  async timeStretch(buffer: AudioBuffer, ratio: number): Promise<AudioBuffer> {
    const newLength = Math.ceil(buffer.length * ratio)
    
    const offlineContext = new OfflineAudioContext(
      buffer.numberOfChannels,
      newLength,
      buffer.sampleRate
    )
    
    // Create buffer source
    const source = offlineContext.createBufferSource()
    source.buffer = buffer
    
    // Adjust playback rate (inverse of ratio for time-stretching)
    source.playbackRate.value = 1 / ratio
    
    source.connect(offlineContext.destination)
    source.start(0)
    
    return offlineContext.startRendering()
  }
  
  /**
   * Find optimal mix point aligned to beat grid
   */
  findOptimalMixPoint(
    outroPoint: number,
    introPoint: number,
    bpm: number
  ): number {
    const beatDuration = 60 / bpm // seconds per beat
    const barDuration = beatDuration * 4 // 4 beats per bar
    
    // Align to nearest bar
    const barsBeforeOutro = Math.floor(outroPoint / barDuration)
    const alignedOutro = barsBeforeOutro * barDuration
    
    // Start transition N bars before outro
    const transitionStart = alignedOutro - (this.config.transitionDuration * barDuration)
    
    return Math.max(0, transitionStart)
  }
  
  /**
   * Calculate beat positions in a track
   */
  calculateBeatGrid(duration: number, bpm: number): number[] {
    const beatDuration = 60 / bpm
    const beats: number[] = []
    
    for (let time = 0; time < duration; time += beatDuration) {
      beats.push(time)
    }
    
    return beats
  }
  
  /**
   * Find nearest beat to a given time
   */
  snapToBeat(time: number, bpm: number): number {
    const beatDuration = 60 / bpm
    return Math.round(time / beatDuration) * beatDuration
  }
  
  /**
   * Check if two tracks are BPM-compatible
   */
  areBpmCompatible(bpm1: number, bpm2: number): boolean {
    const ratio = bpm1 / bpm2
    const adjustmentPercent = Math.abs(1 - ratio) * 100
    return adjustmentPercent <= this.config.maxBpmAdjustment
  }
  
  /**
   * Get configuration
   */
  getConfig(): BeatmatchConfig {
    return { ...this.config }
  }
  
  /**
   * Update configuration
   */
  setConfig(config: Partial<BeatmatchConfig>): void {
    this.config = { ...this.config, ...config }
  }
}

// Singleton instance
let beatmatcherInstance: Beatmatcher | null = null

export function getBeatmatcher(): Beatmatcher {
  if (!beatmatcherInstance) {
    beatmatcherInstance = new Beatmatcher()
  }
  return beatmatcherInstance
}
