<script setup lang="ts">
/**
 * MusicChatView - AI-powered chat interface for music library.
 * 
 * Uses local LLM (Ollama) for natural language understanding.
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 5.1, 5.2, 5.3, 5.4, 7.1, 7.2, 7.3
 */

import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { usePlayerStore } from '@/stores/playerStore'
import api from '@/api/client'
import { parseQuery, formatTime, isParseError } from '@/utils/queryParser'
import { loadChatHistory, saveChatHistory, clearChatHistory } from '@/utils/chatHistory'
import type { ChatMessage, SearchResult, Track } from '@/types'

const playerStore = usePlayerStore()

// State
const messages = ref<ChatMessage[]>([])
const inputText = ref('')
const isLoading = ref(false)
const messagesContainer = ref<HTMLElement | null>(null)
const inputTextarea = ref<HTMLTextAreaElement | null>(null)
const lastQuery = ref<{ type: 'search' | 'recommendation', query?: string, trackId?: number, trackTitle?: string, startTime?: number, endTime?: number } | null>(null)
const aiEnabled = ref(true) // Use LLM by default
const aiHealthy = ref(false)

// Current track context
const currentTrack = computed(() => playerStore.currentTrack)
const currentTime = computed(() => playerStore.currentTime)

// Generate unique ID
function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
}

// Add message to chat
function addMessage(type: ChatMessage['type'], content: string, extra?: Partial<ChatMessage>) {
  const message: ChatMessage = {
    id: generateId(),
    type,
    content,
    timestamp: new Date(),
    // Save last query for retry functionality
    retryData: type === 'error' && lastQuery.value ? { ...lastQuery.value } : undefined,
    ...extra
  }
  messages.value.push(message)
  saveChatHistory(messages.value)
  scrollToBottom()
}

// Retry failed request
async function retryRequest(message: ChatMessage) {
  if (!message.retryData) return
  
  const retryData = message.retryData
  
  // Remove the error message
  const index = messages.value.findIndex(m => m.id === message.id)
  if (index > -1) {
    messages.value.splice(index, 1)
    saveChatHistory(messages.value)
  }
  
  if (retryData.type === 'recommendation') {
    await handleRecommendation()
  } else if (retryData.type === 'search' && retryData.trackId && retryData.startTime !== undefined && retryData.endTime !== undefined) {
    // Re-run search
    isLoading.value = true
    lastQuery.value = retryData
    
    // Use saved track title from original request, not current track
    const trackTitle = retryData.trackTitle || `Track #${retryData.trackId}`
    
    addMessage('system', `Ищу похожие моменты на ${formatTime(retryData.startTime)} - ${formatTime(retryData.endTime)} в "${trackTitle}"...`)
    
    try {
      const response = await api.searchSimilarSegments({
        track_id: retryData.trackId,
        start_time: retryData.startTime,
        end_time: retryData.endTime,
        limit: 10,
        min_similarity: 0.3
      })
      
      if (response.error) {
        addMessage('error', response.error)
      } else if (response.results.length === 0) {
        addMessage('system', 'Похожих моментов не найдено. Попробуйте другой временной диапазон.')
      } else {
        addMessage('results', `Найдено ${response.results.length} похожих моментов:`, {
          results: response.results,
          query: {
            trackId: retryData.trackId,
            trackTitle: trackTitle,
            startTime: retryData.startTime,
            endTime: retryData.endTime
          }
        })
      }
    } catch (error: any) {
      addMessage('error', error.response?.data?.message || 'Ошибка поиска. Попробуйте позже.')
    } finally {
      isLoading.value = false
    }
  }
}

// Scroll to bottom of messages
async function scrollToBottom() {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

// Check if query is a recommendation request
function isRecommendationQuery(query: string): boolean {
  const lowerQuery = query.toLowerCase()
  const keywords = [
    'рекоменд', 'посоветуй', 'предложи', 'что послушать', 
    'новое', 'новые треки', 'что-нибудь новое', 'подбери',
    'recommend', 'suggest', 'what to listen'
  ]
  return keywords.some(kw => lowerQuery.includes(kw))
}

// Handle recommendation request
async function handleRecommendation() {
  isLoading.value = true
  lastQuery.value = { type: 'recommendation' }
  addMessage('system', 'Анализирую ваши предпочтения и подбираю новые треки...')
  
  try {
    const response = await api.getChatRecommendations(5)
    
    if (!response.has_preferences) {
      addMessage('error', response.message || 'Недостаточно данных о ваших предпочтениях.')
      return
    }
    
    if (response.recommendations.length === 0) {
      addMessage('system', response.message || 'Не нашёл подходящих треков.')
      return
    }
    
    // Build preference summary message
    let summaryText = '📊 На основе анализа ваших предпочтений:\n'
    if (response.preference_summary) {
      const ps = response.preference_summary
      if (ps.top_genres?.length) {
        summaryText += `• Любимые жанры: ${ps.top_genres.join(', ')}\n`
      }
      if (ps.top_moods?.length) {
        summaryText += `• Предпочитаемое настроение: ${ps.top_moods.join(', ')}\n`
      }
      summaryText += `• Средний темп: ${ps.avg_bpm} BPM\n`
      summaryText += `• Проанализировано треков: ${ps.tracks_analyzed}`
    }
    
    addMessage('system', summaryText)
    
    // Add recommendations with explanations
    addMessage('recommendations', `🎵 Рекомендую ${response.recommendations.length} новых треков:`, {
      recommendations: response.recommendations
    })
    
  } catch (error: any) {
    addMessage('error', error.response?.data?.message || 'Ошибка получения рекомендаций.')
  } finally {
    isLoading.value = false
  }
}

// Handle AI Chat with LLM
async function handleAIChat(query: string) {
  isLoading.value = true
  addMessage('system', '🤖 Думаю...')
  
  try {
    const context = currentTrack.value ? { current_track: currentTrack.value } : undefined
    const response = await api.aiChat(query, context)
    
    // Remove "thinking" message
    messages.value = messages.value.filter(m => m.content !== '🤖 Думаю...')
    
    if (response.error) {
      addMessage('error', response.error)
      return
    }
    
    // Add AI response message
    if (response.message) {
      addMessage('system', `🤖 ${response.message}`)
    }
    
    // Handle different response types
    if (response.type === 'search' && response.tracks?.length > 0) {
      addMessage('ai_tracks', `Найдено ${response.tracks.length} треков:`, {
        tracks: response.tracks
      })
    } else if (response.type === 'recommendations' && response.tracks?.length > 0) {
      addMessage('ai_tracks', `Рекомендую ${response.tracks.length} треков:`, {
        tracks: response.tracks
      })
    } else if (response.type === 'similar' && response.tracks?.length > 0) {
      addMessage('ai_tracks', `Похожие треки:`, {
        tracks: response.tracks
      })
    } else if (response.type === 'stats' && response.stats) {
      const stats = response.stats
      addMessage('system', `📊 Статистика библиотеки:\n• Всего треков: ${stats.total}\n• Проанализировано: ${stats.analyzed}\n• С лайками: ${stats.liked}\n• В очереди: ${stats.pending}`)
    } else if (response.type === 'info' && response.tracks?.length > 0) {
      const track = response.tracks[0]
      addMessage('system', `🎵 ${track.title}\n• Исполнитель: ${track.artist || 'Неизвестен'}\n• Жанр: ${track.genre || 'Не определён'}\n• BPM: ${track.bpm || '?'}\n• Настроение: ${track.mood || 'Не определено'}`)
    }
    
  } catch (error: any) {
    // Remove "thinking" message
    messages.value = messages.value.filter(m => m.content !== '🤖 Думаю...')
    addMessage('error', error.response?.data?.message || error.message || 'Ошибка AI чата.')
  } finally {
    isLoading.value = false
  }
}

// Check AI health on mount
async function checkAIHealth() {
  try {
    const health = await api.checkAIHealth()
    aiHealthy.value = health.status === 'ok' && health.model_loaded === true
    if (!aiHealthy.value && health.status === 'ok') {
      console.log('AI model not loaded. Available models:', health.available_models)
    }
  } catch {
    aiHealthy.value = false
  }
}

// Handle send message
async function handleSend() {
  const query = inputText.value.trim()
  if (!query || isLoading.value) return
  
  inputText.value = ''
  addMessage('user', query)
  
  // If AI is enabled and healthy, use LLM for natural language
  if (aiEnabled.value && aiHealthy.value) {
    await handleAIChat(query)
    return
  }
  
  // Fallback to old behavior
  // Check if it's a recommendation request
  if (isRecommendationQuery(query)) {
    await handleRecommendation()
    return
  }
  
  if (!currentTrack.value) {
    addMessage('error', 'Сначала выберите трек для воспроизведения. Затем укажите временной диапазон для поиска похожих моментов.\n\nИли напишите "порекомендуй" для персональных рекомендаций.')
    return
  }
  
  const parseResult = parseQuery(query, currentTrack.value.id, currentTrack.value.duration)
  
  if (isParseError(parseResult)) {
    addMessage('error', parseResult.message)
    return
  }
  
  isLoading.value = true
  lastQuery.value = { 
    type: 'search', 
    query, 
    trackId: currentTrack.value.id,
    trackTitle: currentTrack.value.title,
    startTime: parseResult.startTime, 
    endTime: parseResult.endTime 
  }
  addMessage('system', `Ищу похожие моменты на ${formatTime(parseResult.startTime)} - ${formatTime(parseResult.endTime)} в "${currentTrack.value.title}"...`)
  
  try {
    const response = await api.searchSimilarSegments({
      track_id: currentTrack.value.id,
      start_time: parseResult.startTime,
      end_time: parseResult.endTime,
      limit: 10,
      min_similarity: 0.3
    })
    
    if (response.error) {
      addMessage('error', response.error)
    } else if (response.results.length === 0) {
      addMessage('system', 'Похожих моментов не найдено. Попробуйте другой временной диапазон.')
    } else {
      addMessage('results', `Найдено ${response.results.length} похожих моментов:`, {
        results: response.results,
        query: {
          trackId: currentTrack.value.id,
          trackTitle: currentTrack.value.title,
          startTime: parseResult.startTime,
          endTime: parseResult.endTime
        }
      })
    }
  } catch (error: any) {
    addMessage('error', error.response?.data?.message || 'Ошибка поиска. Попробуйте позже.')
  } finally {
    isLoading.value = false
  }
}

// Play result
async function playResult(result: SearchResult) {
  try {
    const track = await api.getTrack(result.track_id)
    playerStore.playTrack(track)
    setTimeout(() => {
      if (playerStore.currentTrack?.id !== track.id) return
      playerStore.seek(result.start_time)
    }, 500)
  } catch (error) {
    console.error('Failed to play result:', error)
  }
}

// Play recommended track
async function playRecommendedTrack(track: any) {
  try {
    playerStore.playTrack(track)
  } catch (error) {
    console.error('Failed to play track:', error)
  }
}

// Like result
async function likeResult(result: SearchResult) {
  try {
    await api.likeFragment(result.fragment_id)
    addMessage('system', `Добавлено в избранное: ${result.track_title} (${formatTime(result.start_time)} - ${formatTime(result.end_time)})`)
  } catch (error) {
    console.error('Failed to like fragment:', error)
  }
}

// Like recommended track
async function likeRecommendedTrack(track: any) {
  try {
    await api.likeTrack(track.id)
    addMessage('system', `❤️ Добавлено в избранное: ${track.title}`)
  } catch (error) {
    console.error('Failed to like track:', error)
  }
}

// Insert current time range
function insertCurrentTimeRange() {
  if (!currentTrack.value) return
  const start = Math.max(0, currentTime.value - 3)
  const end = Math.min(currentTrack.value.duration, currentTime.value + 3)
  inputText.value = `${formatTime(start)} - ${formatTime(end)}`
}

// Clear history
function handleClearHistory() {
  if (confirm('Очистить историю чата?')) {
    clearChatHistory()
    messages.value = []
    addMessage('system', 'История очищена. Укажи временной диапазон текущего трека для поиска похожих моментов, или напиши "порекомендуй" для персональных рекомендаций.')
  }
}

// Handle Enter key (Enter sends, Shift+Enter adds newline)
function handleKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    handleSend()
  }
}

// Auto-resize textarea
function autoResize() {
  if (inputTextarea.value) {
    inputTextarea.value.style.height = 'auto'
    inputTextarea.value.style.height = Math.min(inputTextarea.value.scrollHeight, 200) + 'px'
  }
}

// Load history and show welcome on mount
onMounted(async () => {
  // Check AI health
  await checkAIHealth()
  
  const history = loadChatHistory()
  if (history.length > 0) {
    messages.value = history
    scrollToBottom()
  } else {
    const welcomeMsg = aiHealthy.value 
      ? 'Привет! Я AI-ассистент для твоей музыкальной библиотеки. 🤖\n\nСпроси меня что угодно:\n• "Найди Drum & Bass треки"\n• "Что-нибудь энергичное на 170 BPM"\n• "Покажи похожее на текущий трек"\n• "Статистика библиотеки"'
      : 'Привет! Я помогу найти похожие музыкальные моменты в твоей библиотеке.\n\n📍 Укажи временной диапазон текущего трека, например: "0:29 - 0:35"\n🎵 Или напиши "порекомендуй" для персональных рекомендаций новых треков\n\n⚠️ AI-режим недоступен. Запустите Ollama для полноценного чата.'
    addMessage('system', welcomeMsg)
  }
})
</script>

<template>
  <div class="flex flex-col h-[calc(100vh-200px)] min-h-[500px] bg-gray-50 rounded-lg overflow-hidden">
    <!-- Header -->
    <div class="bg-white border-b px-4 py-3 flex justify-between items-center">
      <div>
        <h1 class="text-lg font-semibold text-gray-900">
          {{ aiHealthy ? '🤖' : '🎵' }} Music Chat
          <span v-if="aiHealthy" class="text-xs text-green-500 ml-2">AI</span>
        </h1>
        <p class="text-sm text-gray-500">
          {{ aiHealthy ? 'AI-ассистент для музыкальной библиотеки' : 'Поиск похожих музыкальных моментов' }}
        </p>
      </div>
      <button 
        v-if="messages.length > 1"
        @click="handleClearHistory"
        class="text-xs text-gray-400 hover:text-red-500"
        title="Очистить историю"
      >
        🗑️ Очистить
      </button>
    </div>

    <!-- Current Track Context -->
    <div v-if="currentTrack" class="bg-blue-50 border-b border-blue-100 px-4 py-2">
      <div class="flex items-center justify-between">
        <div class="flex items-center space-x-2">
          <span class="text-blue-600">🎧</span>
          <span class="text-sm font-medium text-blue-900 truncate max-w-[200px]">{{ currentTrack.title }}</span>
          <span class="text-sm text-blue-600">{{ formatTime(currentTime) }} / {{ formatTime(currentTrack.duration) }}</span>
        </div>
        <button 
          @click="insertCurrentTimeRange"
          class="text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 px-2 py-1 rounded whitespace-nowrap"
        >
          ⏱️ Текущее время
        </button>
      </div>
    </div>
    <div v-else class="bg-yellow-50 border-b border-yellow-100 px-4 py-2">
      <p class="text-sm text-yellow-700">⚠️ Выберите трек для воспроизведения, чтобы искать похожие моменты</p>
    </div>

    <!-- Messages -->
    <div ref="messagesContainer" class="flex-1 overflow-y-auto p-4 space-y-4">
      <div 
        v-for="message in messages" 
        :key="message.id"
        :class="[
          'rounded-lg p-3',
          message.type === 'user' ? 'ml-auto max-w-[85%] bg-blue-500 text-white' : '',
          message.type === 'system' ? 'max-w-[85%] bg-gray-100 text-gray-700' : '',
          message.type === 'error' ? 'max-w-[85%] bg-red-50 text-red-700 border border-red-200' : '',
          message.type === 'results' ? 'bg-white border shadow-sm w-full' : '',
          message.type === 'recommendations' ? 'bg-white border border-purple-200 shadow-sm w-full' : ''
        ]"
      >
        <!-- Error message with retry button -->
        <div v-if="message.type === 'error'">
          <p class="text-sm whitespace-pre-line">{{ message.content }}</p>
          <button 
            v-if="message.retryData"
            @click="retryRequest(message)"
            :disabled="isLoading"
            class="mt-2 text-xs bg-red-100 hover:bg-red-200 text-red-700 px-3 py-1 rounded-full transition-colors disabled:opacity-50 flex items-center gap-1"
          >
            🔄 Повторить
          </button>
        </div>
        
        <p v-else-if="message.type !== 'results' && message.type !== 'recommendations' && message.type !== 'ai_tracks'" class="text-sm whitespace-pre-line">{{ message.content }}</p>
        
        <!-- AI Tracks display -->
        <div v-else-if="message.type === 'ai_tracks'">
          <p class="text-sm text-gray-600 mb-3">{{ message.content }}</p>
          
          <div class="space-y-2">
            <div 
              v-for="(track, index) in (message as any).tracks" 
              :key="track.id"
              class="flex items-center justify-between p-2 bg-gradient-to-r from-green-50 to-blue-50 rounded-lg border border-green-100 hover:border-green-200 transition-colors"
            >
              <div class="flex items-center space-x-3 flex-1 min-w-0">
                <span class="text-green-600 font-bold w-5">{{ index + 1 }}</span>
                <div class="flex-1 min-w-0">
                  <p class="font-medium text-sm text-gray-900 truncate">{{ track.title }}</p>
                  <p class="text-xs text-gray-500">
                    {{ track.artist || 'Unknown' }} · 
                    {{ track.genre || '?' }} · 
                    {{ track.bpm || '?' }} BPM · 
                    {{ track.mood || '?' }}
                  </p>
                </div>
              </div>
              <div class="flex items-center space-x-1 ml-2">
                <button 
                  @click="playRecommendedTrack(track)"
                  class="p-1.5 hover:bg-green-100 rounded text-green-600"
                  title="Воспроизвести"
                >▶️</button>
                <button 
                  @click="likeRecommendedTrack(track)"
                  class="p-1.5 hover:bg-red-100 rounded text-red-500"
                  title="В избранное"
                >❤️</button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Recommendations display -->
        <div v-else-if="message.type === 'recommendations'">
          <p class="text-sm text-gray-600 mb-3">{{ message.content }}</p>
          
          <div class="space-y-3">
            <div 
              v-for="(rec, index) in (message as any).recommendations" 
              :key="rec.track.id"
              class="p-3 bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg border border-purple-100"
            >
              <div class="flex items-start justify-between">
                <div class="flex-1 min-w-0">
                  <div class="flex items-center space-x-2">
                    <span class="text-purple-600 font-bold">{{ index + 1 }}</span>
                    <p class="font-medium text-gray-900 truncate">{{ rec.track.title }}</p>
                  </div>
                  <p class="text-sm text-gray-500 mt-0.5">{{ rec.track.artist || 'Unknown' }}</p>
                  <p class="text-xs text-gray-400 mt-1">
                    {{ rec.track.genre }} · {{ rec.track.bpm }} BPM · {{ rec.track.mood }}
                  </p>
                  <p class="text-sm text-purple-700 mt-2 italic">💡 {{ rec.explanation }}</p>
                </div>
                <div class="flex items-center space-x-1 ml-3">
                  <button 
                    @click="playRecommendedTrack(rec.track)"
                    class="p-2 hover:bg-purple-100 rounded-full text-purple-600 transition-colors"
                    title="Воспроизвести"
                  >▶️</button>
                  <button 
                    @click="likeRecommendedTrack(rec.track)"
                    class="p-2 hover:bg-red-100 rounded-full text-red-500 transition-colors"
                    title="В избранное"
                  >❤️</button>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Search results display -->
        <div v-else>
          <p class="text-sm text-gray-600 mb-3">{{ message.content }}</p>
          
          <div v-if="message.query" class="text-xs text-gray-500 mb-3 pb-2 border-b">
            Запрос: {{ message.query.trackTitle }} ({{ formatTime(message.query.startTime) }} - {{ formatTime(message.query.endTime) }})
          </div>
          
          <div class="space-y-2">
            <div 
              v-for="(result, index) in message.results" 
              :key="result.fragment_id"
              class="flex items-center justify-between p-2 bg-gray-50 rounded hover:bg-gray-100 transition-colors"
            >
              <div class="flex items-center space-x-3 flex-1 min-w-0">
                <span class="text-gray-400 text-sm w-5">{{ index + 1 }}</span>
                <div class="flex-1 min-w-0">
                  <p class="font-medium text-sm text-gray-900 truncate">{{ result.track_title }}</p>
                  <p class="text-xs text-gray-500">
                    {{ result.track_artist || 'Unknown' }} · 
                    {{ formatTime(result.start_time) }} - {{ formatTime(result.end_time) }} · 
                    {{ Math.round(result.similarity * 100) }}%
                  </p>
                </div>
              </div>
              <div class="flex items-center space-x-1 ml-2">
                <button 
                  @click="playResult(result)"
                  class="p-1.5 hover:bg-blue-100 rounded text-blue-600"
                  title="Воспроизвести"
                >▶️</button>
                <button 
                  @click="likeResult(result)"
                  class="p-1.5 hover:bg-red-100 rounded text-red-500"
                  title="В избранное"
                >❤️</button>
              </div>
            </div>
          </div>
        </div>
        
        <p class="text-xs opacity-60 mt-1">
          {{ message.timestamp.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }) }}
        </p>
      </div>
      
      <div v-if="isLoading" class="flex items-center space-x-2 text-gray-500">
        <div class="animate-spin w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
        <span class="text-sm">Ищу...</span>
      </div>
    </div>

    <!-- Input -->
    <div class="bg-white border-t p-4">
      <div class="flex space-x-2">
        <textarea
          v-model="inputText"
          @keydown="handleKeydown"
          :placeholder="aiHealthy ? 'Спроси что угодно о музыке... (Shift+Enter для новой строки)' : 'Введите время (0:29 - 0:35) или порекомендуй'"
          class="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none min-h-[44px] max-h-[200px]"
          :disabled="isLoading"
          rows="1"
          @input="autoResize"
          ref="inputTextarea"
        ></textarea>
        <button
          @click="handleSend"
          :disabled="isLoading || !inputText.trim()"
          class="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg transition-colors self-end"
        >
          {{ aiHealthy ? 'Спросить' : 'Найти' }}
        </button>
      </div>
      <p class="text-xs text-gray-400 mt-2">
        {{ aiHealthy ? 'Примеры: "Найди House треки", "Импортируй ссылки", "Покажи статистику" | Enter - отправить, Shift+Enter - новая строка' : 'Форматы: "0:29 - 0:35", "29s - 35s" | Команды: "порекомендуй", "посоветуй"' }}
      </p>
    </div>
  </div>
</template>
