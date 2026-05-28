<script setup lang="ts">
/**
 * AudioVisualizer - Real-time audio visualization
 * Requirements: 23.1, 23.2, 23.3, 23.4, 23.5
 */
import { ref, onMounted, onUnmounted, watch } from 'vue'

type VisualizerMode = 'bars' | 'wave' | 'circular'

interface Props {
  audioElement?: HTMLAudioElement | null
  mode?: VisualizerMode
  enabled?: boolean
  color?: string
}

const props = withDefaults(defineProps<Props>(), {
  mode: 'bars',
  enabled: true,
  color: '#1DB954',
})

// Refs
const canvasRef = ref<HTMLCanvasElement | null>(null)
const audioContext = ref<AudioContext | null>(null)
const analyser = ref<AnalyserNode | null>(null)
const dataArray = ref<Uint8Array | null>(null)
const animationId = ref<number | null>(null)
const isConnected = ref(false)

// Connect to audio source
const connectAudio = () => {
  if (!props.audioElement || isConnected.value) return
  
  try {
    if (!audioContext.value) {
      audioContext.value = new AudioContext()
    }
    
    analyser.value = audioContext.value.createAnalyser()
    analyser.value.fftSize = 256
    
    const source = audioContext.value.createMediaElementSource(props.audioElement)
    source.connect(analyser.value)
    analyser.value.connect(audioContext.value.destination)
    
    dataArray.value = new Uint8Array(analyser.value.frequencyBinCount)
    isConnected.value = true
    
    if (props.enabled) {
      startVisualization()
    }
  } catch (e) {
    console.error('Error connecting audio:', e)
  }
}

// Start visualization loop
const startVisualization = () => {
  if (!canvasRef.value || !analyser.value || !dataArray.value) return
  
  const canvas = canvasRef.value
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  
  const draw = () => {
    if (!props.enabled || !analyser.value || !dataArray.value) {
      animationId.value = null
      return
    }
    
    animationId.value = requestAnimationFrame(draw)
    analyser.value.getByteFrequencyData(dataArray.value)
    
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    switch (props.mode) {
      case 'bars':
        drawBars(ctx, canvas, dataArray.value)
        break
      case 'wave':
        drawWave(ctx, canvas, dataArray.value)
        break
      case 'circular':
        drawCircular(ctx, canvas, dataArray.value)
        break
    }
  }
  
  draw()
}

// Stop visualization
const stopVisualization = () => {
  if (animationId.value) {
    cancelAnimationFrame(animationId.value)
    animationId.value = null
  }
}

// Draw bars visualization
const drawBars = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, data: Uint8Array) => {
  const barCount = data.length / 2
  const barWidth = canvas.width / barCount
  const barGap = 2
  
  for (let i = 0; i < barCount; i++) {
    const barHeight = (data[i] / 255) * canvas.height
    const x = i * barWidth
    const y = canvas.height - barHeight
    
    // Create gradient
    const gradient = ctx.createLinearGradient(x, y, x, canvas.height)
    gradient.addColorStop(0, props.color)
    gradient.addColorStop(1, `${props.color}40`)
    
    ctx.fillStyle = gradient
    ctx.fillRect(x + barGap / 2, y, barWidth - barGap, barHeight)
  }
}

// Draw wave visualization
const drawWave = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, data: Uint8Array) => {
  ctx.beginPath()
  ctx.strokeStyle = props.color
  ctx.lineWidth = 2
  
  const sliceWidth = canvas.width / data.length
  let x = 0
  
  for (let i = 0; i < data.length; i++) {
    const v = data[i] / 255
    const y = (v * canvas.height / 2) + canvas.height / 4
    
    if (i === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
    
    x += sliceWidth
  }
  
  ctx.stroke()
  
  // Mirror wave
  ctx.beginPath()
  ctx.strokeStyle = `${props.color}60`
  x = 0
  
  for (let i = 0; i < data.length; i++) {
    const v = data[i] / 255
    const y = canvas.height - ((v * canvas.height / 2) + canvas.height / 4)
    
    if (i === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
    
    x += sliceWidth
  }
  
  ctx.stroke()
}

// Draw circular visualization
const drawCircular = (ctx: CanvasRenderingContext2D, canvas: HTMLCanvasElement, data: Uint8Array) => {
  const centerX = canvas.width / 2
  const centerY = canvas.height / 2
  const radius = Math.min(centerX, centerY) * 0.6
  
  ctx.beginPath()
  ctx.strokeStyle = props.color
  ctx.lineWidth = 2
  
  for (let i = 0; i < data.length; i++) {
    const angle = (i / data.length) * Math.PI * 2
    const amplitude = (data[i] / 255) * radius * 0.5
    const r = radius + amplitude
    
    const x = centerX + Math.cos(angle) * r
    const y = centerY + Math.sin(angle) * r
    
    if (i === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  }
  
  ctx.closePath()
  ctx.stroke()
  
  // Inner circle
  ctx.beginPath()
  ctx.strokeStyle = `${props.color}40`
  ctx.arc(centerX, centerY, radius * 0.3, 0, Math.PI * 2)
  ctx.stroke()
}

// Watchers
watch(() => props.enabled, (enabled) => {
  if (enabled) {
    startVisualization()
  } else {
    stopVisualization()
  }
})

watch(() => props.audioElement, () => {
  if (props.audioElement && !isConnected.value) {
    connectAudio()
  }
})

// Lifecycle
onMounted(() => {
  if (props.audioElement) {
    connectAudio()
  }
})

onUnmounted(() => {
  stopVisualization()
  if (audioContext.value) {
    audioContext.value.close()
  }
})
</script>

<template>
  <div class="audio-visualizer">
    <canvas
      ref="canvasRef"
      width="300"
      height="100"
      class="w-full h-full"
    />
  </div>
</template>
