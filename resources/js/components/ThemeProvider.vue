<script setup lang="ts">
/**
 * ThemeProvider - Theme system with CSS variables
 * Requirements: 21.1, 21.2, 21.5, 21.6
 */
import { ref, computed, watch, onMounted, provide } from 'vue'

type ThemeName = 'dark' | 'light'

interface ThemeColors {
  background: string
  surface: string
  primary: string
  accent: string
  text: string
  textSecondary: string
  success: string
  error: string
}

const themes: Record<ThemeName, ThemeColors> = {
  dark: {
    background: '#121212',
    surface: '#1e1e1e',
    primary: '#1DB954',
    accent: '#1ed760',
    text: '#ffffff',
    textSecondary: '#b3b3b3',
    success: '#1DB954',
    error: '#f44336',
  },
  light: {
    background: '#ffffff',
    surface: '#f5f5f5',
    primary: '#1DB954',
    accent: '#1ed760',
    text: '#121212',
    textSecondary: '#666666',
    success: '#1DB954',
    error: '#d32f2f',
  },
}

// State
const currentTheme = ref<ThemeName>('dark')
const dominantColor = ref<string | null>(null)

// Computed
const themeColors = computed(() => themes[currentTheme.value])

const gradientBackground = computed(() => {
  if (dominantColor.value) {
    return `linear-gradient(180deg, ${dominantColor.value}40 0%, ${themeColors.value.background} 100%)`
  }
  return themeColors.value.background
})

// Methods
const setTheme = (theme: ThemeName) => {
  currentTheme.value = theme
  localStorage.setItem('theme', theme)
  applyTheme()
}

const toggleTheme = () => {
  setTheme(currentTheme.value === 'dark' ? 'light' : 'dark')
}

const setDominantColor = (color: string | null) => {
  dominantColor.value = color
  if (color) {
    document.documentElement.style.setProperty('--dominant-color', color)
  }
}

const extractDominantColor = async (imageUrl: string): Promise<string> => {
  return new Promise((resolve) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const canvas = document.createElement('canvas')
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        resolve('#1DB954')
        return
      }
      
      canvas.width = 50
      canvas.height = 50
      ctx.drawImage(img, 0, 0, 50, 50)
      
      const imageData = ctx.getImageData(0, 0, 50, 50)
      const data = imageData.data
      
      let r = 0, g = 0, b = 0, count = 0
      
      for (let i = 0; i < data.length; i += 4) {
        // Skip very dark or very light pixels
        const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3
        if (brightness > 30 && brightness < 220) {
          r += data[i]
          g += data[i + 1]
          b += data[i + 2]
          count++
        }
      }
      
      if (count > 0) {
        r = Math.round(r / count)
        g = Math.round(g / count)
        b = Math.round(b / count)
      } else {
        r = 29; g = 185; b = 84 // Default green
      }
      
      resolve(`rgb(${r}, ${g}, ${b})`)
    }
    img.onerror = () => resolve('#1DB954')
    img.src = imageUrl
  })
}

const applyTheme = () => {
  const colors = themeColors.value
  const root = document.documentElement
  
  root.style.setProperty('--color-background', colors.background)
  root.style.setProperty('--color-surface', colors.surface)
  root.style.setProperty('--color-primary', colors.primary)
  root.style.setProperty('--color-accent', colors.accent)
  root.style.setProperty('--color-text', colors.text)
  root.style.setProperty('--color-text-secondary', colors.textSecondary)
  root.style.setProperty('--color-success', colors.success)
  root.style.setProperty('--color-error', colors.error)
  
  // Set body background
  document.body.style.backgroundColor = colors.background
  document.body.style.color = colors.text
}

// Provide theme context
provide('theme', {
  currentTheme,
  themeColors,
  gradientBackground,
  setTheme,
  toggleTheme,
  setDominantColor,
  extractDominantColor,
})

// Initialize
onMounted(() => {
  const savedTheme = localStorage.getItem('theme') as ThemeName | null
  if (savedTheme && (savedTheme === 'dark' || savedTheme === 'light')) {
    currentTheme.value = savedTheme
  }
  applyTheme()
})

// Watch for theme changes
watch(currentTheme, () => {
  applyTheme()
})
</script>

<template>
  <div 
    class="theme-provider min-h-screen transition-colors duration-300"
    :style="{ background: gradientBackground }"
  >
    <slot />
  </div>
</template>
