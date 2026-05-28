<script setup lang="ts">
/**
 * RatingWidget - 5-star rating component
 * Requirements: 13.1, 13.3, 13.4, 13.5
 */
import { ref, computed, watch } from 'vue'

interface Props {
  modelValue?: number | null
  readonly?: boolean
  size?: 'sm' | 'md' | 'lg'
  showLabel?: boolean
  averageRating?: number | null
  showSentimentActions?: boolean
}

interface Emits {
  (e: 'update:modelValue', value: number | null): void
  (e: 'rate', value: number): void
  (e: 'clear'): void
}

const props = withDefaults(defineProps<Props>(), {
  modelValue: null,
  readonly: false,
  size: 'md',
  showLabel: false,
  averageRating: null,
  showSentimentActions: true,
})

const emit = defineEmits<Emits>()

// State
const hoverRating = ref<number | null>(null)
const isAnimating = ref(false)
const animatedStar = ref<number | null>(null)

// Computed
const displayRating = computed(() => {
  if (hoverRating.value !== null) return hoverRating.value
  return props.modelValue ?? 0
})

const starSize = computed(() => {
  switch (props.size) {
    case 'sm': return 'text-lg'
    case 'lg': return 'text-3xl'
    default: return 'text-2xl'
  }
})

const ratingLabel = computed(() => {
  const rating = props.modelValue
  if (!rating) return 'Not rated'
  const labels = ['', 'Poor', 'Fair', 'Good', 'Great', 'Excellent']
  return labels[rating] || ''
})

const sentimentButtonClass = computed(() => {
  switch (props.size) {
    case 'sm': return 'px-2 py-1 text-[10px]'
    case 'lg': return 'px-3 py-1.5 text-xs'
    default: return 'px-2.5 py-1 text-[11px]'
  }
})

// Methods
const getStarFill = (starIndex: number): 'full' | 'half' | 'empty' => {
  const rating = displayRating.value
  if (rating >= starIndex) return 'full'
  if (rating >= starIndex - 0.5) return 'half'
  return 'empty'
}

const getAverageStarFill = (starIndex: number): 'full' | 'half' | 'empty' => {
  if (!props.averageRating) return 'empty'
  if (props.averageRating >= starIndex) return 'full'
  if (props.averageRating >= starIndex - 0.5) return 'half'
  return 'empty'
}

const onStarHover = (starIndex: number) => {
  if (props.readonly) return
  hoverRating.value = starIndex
}

const onStarLeave = () => {
  hoverRating.value = null
}

const onStarClick = (starIndex: number) => {
  if (props.readonly) return
  
  // If clicking the same rating, clear it
  if (props.modelValue === starIndex) {
    emit('update:modelValue', null)
    emit('clear')
    return
  }
  
  // Animate the clicked star
  animatedStar.value = starIndex
  isAnimating.value = true
  setTimeout(() => {
    isAnimating.value = false
    animatedStar.value = null
  }, 300)
  
  emit('update:modelValue', starIndex)
  emit('rate', starIndex)
}

const onSentimentClick = (sentiment: 'dislike' | 'like') => {
  onStarClick(sentiment === 'dislike' ? 1 : 5)
}

// Watch for external changes
watch(() => props.modelValue, () => {
  // Could trigger animation here if needed
})
</script>

<template>
  <div class="rating-widget">
    <div
      v-if="showSentimentActions && !readonly"
      class="mb-2 flex flex-wrap items-center gap-2"
    >
      <button
        type="button"
        class="rounded-full border font-semibold uppercase tracking-[0.14em] transition"
        :class="[
          sentimentButtonClass,
          modelValue === 1
            ? 'border-rose-300 bg-rose-50 text-rose-700'
            : 'border-slate-200 bg-white text-slate-500 hover:border-rose-200 hover:text-rose-600'
        ]"
        @click="onSentimentClick('dislike')"
      >
        Dislike
      </button>
      <button
        type="button"
        class="rounded-full border font-semibold uppercase tracking-[0.14em] transition"
        :class="[
          sentimentButtonClass,
          modelValue === 5
            ? 'border-emerald-300 bg-emerald-50 text-emerald-700'
            : 'border-slate-200 bg-white text-slate-500 hover:border-emerald-200 hover:text-emerald-600'
        ]"
        @click="onSentimentClick('like')"
      >
        Like
      </button>
    </div>

    <!-- User Rating -->
    <div 
      class="flex items-center space-x-1"
      @mouseleave="onStarLeave"
    >
      <button
        v-for="star in 5"
        :key="star"
        type="button"
        :disabled="readonly"
        @mouseenter="onStarHover(star)"
        @click="onStarClick(star)"
        class="transition-all duration-200 focus:outline-none"
        :class="[
          starSize,
          readonly ? 'cursor-default' : 'cursor-pointer hover:scale-110',
          isAnimating && animatedStar === star ? 'animate-pulse scale-125' : ''
        ]"
      >
        <!-- Full star -->
        <span 
          v-if="getStarFill(star) === 'full'"
          class="text-yellow-400 drop-shadow-glow"
          :class="{ 'animate-glow': isAnimating && animatedStar === star }"
        >
          ★
        </span>
        <!-- Half star -->
        <span 
          v-else-if="getStarFill(star) === 'half'"
          class="relative"
        >
          <span class="text-gray-400">★</span>
          <span class="absolute inset-0 overflow-hidden w-1/2">
            <span class="text-yellow-400">★</span>
          </span>
        </span>
        <!-- Empty star -->
        <span v-else class="text-gray-400 hover:text-gray-300">
          ★
        </span>
      </button>
      
      <!-- Rating label -->
      <span 
        v-if="showLabel && modelValue"
        class="ml-2 text-sm text-gray-400"
      >
        {{ ratingLabel }}
      </span>
    </div>

    <!-- Average Rating (if provided) -->
    <div 
      v-if="averageRating !== null"
      class="flex items-center space-x-1 mt-1"
    >
      <span class="text-xs text-gray-500 mr-1">Avg:</span>
      <span
        v-for="star in 5"
        :key="`avg-${star}`"
        class="text-sm"
      >
        <span 
          v-if="getAverageStarFill(star) === 'full'"
          class="text-yellow-500/70"
        >★</span>
        <span 
          v-else-if="getAverageStarFill(star) === 'half'"
          class="relative"
        >
          <span class="text-gray-500">★</span>
          <span class="absolute inset-0 overflow-hidden w-1/2">
            <span class="text-yellow-500/70">★</span>
          </span>
        </span>
        <span v-else class="text-gray-500">★</span>
      </span>
      <span class="text-xs text-gray-500 ml-1">
        ({{ averageRating.toFixed(1) }})
      </span>
    </div>
  </div>
</template>

<style scoped>
.drop-shadow-glow {
  filter: drop-shadow(0 0 4px rgba(250, 204, 21, 0.5));
}

.animate-glow {
  animation: glow 0.3s ease-out;
}

@keyframes glow {
  0% {
    filter: drop-shadow(0 0 2px rgba(250, 204, 21, 0.3));
    transform: scale(1);
  }
  50% {
    filter: drop-shadow(0 0 12px rgba(250, 204, 21, 0.8));
    transform: scale(1.2);
  }
  100% {
    filter: drop-shadow(0 0 4px rgba(250, 204, 21, 0.5));
    transform: scale(1);
  }
}
</style>
