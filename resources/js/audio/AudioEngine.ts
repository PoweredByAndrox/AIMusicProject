/**
 * Audio Engine with Crossfade Support
 * 
 * Uses Web Audio API for advanced audio playback with crossfade transitions.
 * 
 * Requirements: 3.1, 3.2, 3.3, 3.5
 */

export interface AudioEngineConfig {
  crossfadeEnabled: boolean
  crossfadeDuration: number // 0-12 seconds
}

export interface CrossfadeState {
  isActive: boolean
  startTime: number
  duration: number
  progress: number
}

export class AudioEngine {
  private audioContext: AudioContext | null = null
  private currentAudio: HTMLAudioElement | null = null
  private nextAudio: HTMLAudioElement | null = null
  private currentGain: GainNode | null = null
  private nextGain: GainNode | null = null
  private masterGain: GainNode | null = null
  private currentSource: MediaElementAudioSourceNode | null = null
  private nextSource: MediaElementAudioSourceNode | null = null
  
  private config: AudioEngineConfig = {
    crossfadeEnabled: false,
    crossfadeDuration: 5,
  }
  
  private crossfadeState: CrossfadeState = {
    isActive: false,
    startTime: 0,
    duration: 0,
    progress: 0,
  }
  
  private crossfadeTimer: number | null = null
  private onTrackEndCallback: (() => void) | null = null
  private onTimeUpdateCallback: ((time: number) => void) | null = null
  private onDurationChangeCallback: ((duration: number) => void) | null = null
  
  constructor() {
    this.initAudioContext()
  }
  
  private initAudioContext(): void {
    if (typeof window !== 'undefined' && window.AudioContext) {
      this.audioContext = new AudioContext()
      this.masterGain = this.audioContext.createGain()
      this.masterGain.connect(this.audioContext.destination)
    }
  }
  
  private ensureAudioContext(): void {
    if (this.audioContext?.state === 'suspended') {
      this.audioContext.resume()
    }
  }
  
  /**
   * Load and prepare a track for playback
   */
  async loadTrack(url: string): Promise<void> {
    this.ensureAudioContext()
    
    // Clean up previous audio
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio.src = ''
      if (this.currentSource) {
        this.currentSource.disconnect()
      }
    }
    
    // Create new audio element
    this.currentAudio = new Audio()
    this.currentAudio.crossOrigin = 'anonymous'
    this.currentAudio.preload = 'auto'
    
    // Set up event listeners
    this.currentAudio.addEventListener('ended', () => {
      if (!this.crossfadeState.isActive) {
        this.onTrackEndCallback?.()
      }
    })
    
    this.currentAudio.addEventListener('timeupdate', () => {
      if (this.currentAudio) {
        this.onTimeUpdateCallback?.(this.currentAudio.currentTime)
        this.checkCrossfadeTrigger()
      }
    })
    
    this.currentAudio.addEventListener('durationchange', () => {
      if (this.currentAudio && this.currentAudio.duration) {
        this.onDurationChangeCallback?.(this.currentAudio.duration)
      }
    })
    
    // Connect to Web Audio API
    if (this.audioContext && this.masterGain) {
      this.currentSource = this.audioContext.createMediaElementSource(this.currentAudio)
      this.currentGain = this.audioContext.createGain()
      this.currentSource.connect(this.currentGain)
      this.currentGain.connect(this.masterGain)
    }
    
    // Load the audio
    this.currentAudio.src = url
    await this.currentAudio.load()
  }
  
  /**
   * Prepare next track for crossfade
   */
  async prepareNextTrack(url: string): Promise<void> {
    if (!this.config.crossfadeEnabled || !this.audioContext || !this.masterGain) {
      return
    }
    
    // Clean up previous next audio
    if (this.nextAudio) {
      this.nextAudio.pause()
      this.nextAudio.src = ''
      if (this.nextSource) {
        this.nextSource.disconnect()
      }
    }
    
    // Create new audio element for next track
    this.nextAudio = new Audio()
    this.nextAudio.crossOrigin = 'anonymous'
    this.nextAudio.preload = 'auto'
    
    // Connect to Web Audio API
    this.nextSource = this.audioContext.createMediaElementSource(this.nextAudio)
    this.nextGain = this.audioContext.createGain()
    this.nextGain.gain.value = 0 // Start silent
    this.nextSource.connect(this.nextGain)
    this.nextGain.connect(this.masterGain)
    
    // Load the audio
    this.nextAudio.src = url
    await this.nextAudio.load()
  }
  
  /**
   * Check if crossfade should be triggered
   */
  private checkCrossfadeTrigger(): void {
    if (!this.config.crossfadeEnabled || !this.currentAudio || this.crossfadeState.isActive) {
      return
    }
    
    const timeRemaining = this.currentAudio.duration - this.currentAudio.currentTime
    const crossfadeDuration = this.getEffectiveCrossfadeDuration()
    
    if (timeRemaining <= crossfadeDuration && timeRemaining > 0) {
      this.startCrossfade()
    }
  }
  
  /**
   * Get effective crossfade duration (adapts if next track is shorter)
   */
  private getEffectiveCrossfadeDuration(): number {
    let duration = this.config.crossfadeDuration
    
    // Adapt if next track is shorter than crossfade duration
    if (this.nextAudio && this.nextAudio.duration) {
      duration = Math.min(duration, this.nextAudio.duration * 0.8) // Max 80% of next track
    }
    
    return duration
  }
  
  /**
   * Start crossfade transition
   */
  private startCrossfade(): void {
    if (!this.currentGain || !this.nextGain || !this.nextAudio || !this.audioContext) {
      return
    }
    
    const duration = this.getEffectiveCrossfadeDuration()
    const startTime = this.audioContext.currentTime
    
    this.crossfadeState = {
      isActive: true,
      startTime,
      duration,
      progress: 0,
    }
    
    // Start next track
    this.nextAudio.play()
    
    // Schedule volume ramps
    // Current track: fade out (1 -> 0)
    this.currentGain.gain.setValueAtTime(1, startTime)
    this.currentGain.gain.linearRampToValueAtTime(0, startTime + duration)
    
    // Next track: fade in (0 -> 1)
    this.nextGain.gain.setValueAtTime(0, startTime)
    this.nextGain.gain.linearRampToValueAtTime(1, startTime + duration)
    
    // Schedule crossfade completion
    this.crossfadeTimer = window.setTimeout(() => {
      this.completeCrossfade()
    }, duration * 1000)
  }
  
  /**
   * Complete crossfade and swap tracks
   */
  private completeCrossfade(): void {
    // Stop and clean up current track
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio.src = ''
    }
    if (this.currentSource) {
      this.currentSource.disconnect()
    }
    
    // Swap next to current
    this.currentAudio = this.nextAudio
    this.currentSource = this.nextSource
    this.currentGain = this.nextGain
    
    // Clear next references
    this.nextAudio = null
    this.nextSource = null
    this.nextGain = null
    
    // Reset crossfade state
    this.crossfadeState = {
      isActive: false,
      startTime: 0,
      duration: 0,
      progress: 0,
    }
    
    // Set up event listeners for new current track
    if (this.currentAudio) {
      this.currentAudio.addEventListener('ended', () => {
        if (!this.crossfadeState.isActive) {
          this.onTrackEndCallback?.()
        }
      })
      
      this.currentAudio.addEventListener('timeupdate', () => {
        if (this.currentAudio) {
          this.onTimeUpdateCallback?.(this.currentAudio.currentTime)
          this.checkCrossfadeTrigger()
        }
      })
    }
    
    // Notify track end (for queue management)
    this.onTrackEndCallback?.()
  }
  
  /**
   * Execute immediate crossfade (for manual track change)
   */
  async executeCrossfade(nextUrl: string, duration?: number): Promise<void> {
    if (!this.audioContext || !this.masterGain) {
      // Fallback: just load and play
      await this.loadTrack(nextUrl)
      this.play()
      return
    }
    
    const crossfadeDuration = duration ?? this.config.crossfadeDuration
    
    // Prepare next track
    await this.prepareNextTrack(nextUrl)
    
    if (!this.nextAudio || !this.nextGain || !this.currentGain) {
      // Fallback
      await this.loadTrack(nextUrl)
      this.play()
      return
    }
    
    const startTime = this.audioContext.currentTime
    
    this.crossfadeState = {
      isActive: true,
      startTime,
      duration: crossfadeDuration,
      progress: 0,
    }
    
    // Start next track
    this.nextAudio.play()
    
    // Schedule volume ramps
    this.currentGain.gain.setValueAtTime(this.currentGain.gain.value, startTime)
    this.currentGain.gain.linearRampToValueAtTime(0, startTime + crossfadeDuration)
    
    this.nextGain.gain.setValueAtTime(0, startTime)
    this.nextGain.gain.linearRampToValueAtTime(1, startTime + crossfadeDuration)
    
    // Schedule completion
    this.crossfadeTimer = window.setTimeout(() => {
      this.completeCrossfade()
    }, crossfadeDuration * 1000)
  }
  
  /**
   * Cancel ongoing crossfade
   */
  cancelCrossfade(): void {
    if (this.crossfadeTimer) {
      clearTimeout(this.crossfadeTimer)
      this.crossfadeTimer = null
    }
    
    if (this.nextAudio) {
      this.nextAudio.pause()
      this.nextAudio.src = ''
    }
    if (this.nextSource) {
      this.nextSource.disconnect()
    }
    
    this.nextAudio = null
    this.nextSource = null
    this.nextGain = null
    
    // Reset current gain to full
    if (this.currentGain && this.audioContext) {
      this.currentGain.gain.cancelScheduledValues(this.audioContext.currentTime)
      this.currentGain.gain.setValueAtTime(1, this.audioContext.currentTime)
    }
    
    this.crossfadeState = {
      isActive: false,
      startTime: 0,
      duration: 0,
      progress: 0,
    }
  }
  
  // Playback controls
  play(): void {
    this.ensureAudioContext()
    this.currentAudio?.play()
  }
  
  pause(): void {
    this.currentAudio?.pause()
  }
  
  stop(): void {
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio.currentTime = 0
    }
    this.cancelCrossfade()
  }
  
  seek(time: number): void {
    if (this.currentAudio) {
      this.currentAudio.currentTime = Math.max(0, Math.min(time, this.currentAudio.duration || 0))
    }
  }
  
  setVolume(volume: number): void {
    const clampedVolume = Math.max(0, Math.min(1, volume))
    if (this.masterGain) {
      this.masterGain.gain.value = clampedVolume
    }
    if (this.currentAudio) {
      this.currentAudio.volume = clampedVolume
    }
  }
  
  // Configuration
  setCrossfadeEnabled(enabled: boolean): void {
    this.config.crossfadeEnabled = enabled
    if (!enabled) {
      this.cancelCrossfade()
    }
  }
  
  setCrossfadeDuration(duration: number): void {
    // Clamp to valid range (0-12 seconds)
    this.config.crossfadeDuration = Math.max(0, Math.min(12, duration))
  }
  
  getConfig(): AudioEngineConfig {
    return { ...this.config }
  }
  
  getCrossfadeState(): CrossfadeState {
    return { ...this.crossfadeState }
  }
  
  // Getters
  getCurrentTime(): number {
    return this.currentAudio?.currentTime ?? 0
  }
  
  getDuration(): number {
    return this.currentAudio?.duration ?? 0
  }
  
  isPlaying(): boolean {
    return this.currentAudio ? !this.currentAudio.paused : false
  }
  
  // Event callbacks
  onTrackEnd(callback: () => void): void {
    this.onTrackEndCallback = callback
  }
  
  onTimeUpdate(callback: (time: number) => void): void {
    this.onTimeUpdateCallback = callback
  }
  
  onDurationChange(callback: (duration: number) => void): void {
    this.onDurationChangeCallback = callback
  }
  
  // Cleanup
  destroy(): void {
    this.cancelCrossfade()
    
    if (this.currentAudio) {
      this.currentAudio.pause()
      this.currentAudio.src = ''
    }
    
    if (this.audioContext) {
      this.audioContext.close()
    }
    
    this.currentAudio = null
    this.nextAudio = null
    this.currentSource = null
    this.nextSource = null
    this.currentGain = null
    this.nextGain = null
    this.masterGain = null
    this.audioContext = null
  }
}

// Singleton instance for global use
let audioEngineInstance: AudioEngine | null = null

export function getAudioEngine(): AudioEngine {
  if (!audioEngineInstance) {
    audioEngineInstance = new AudioEngine()
  }
  return audioEngineInstance
}

export function destroyAudioEngine(): void {
  if (audioEngineInstance) {
    audioEngineInstance.destroy()
    audioEngineInstance = null
  }
}
