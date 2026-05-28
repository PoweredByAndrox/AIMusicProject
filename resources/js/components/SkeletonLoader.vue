<script setup lang="ts">
/**
 * SkeletonLoader - Placeholder content during loading
 * Requirements: 24.4
 */
interface Props {
  type?: 'text' | 'avatar' | 'card' | 'track' | 'playlist'
  lines?: number
  width?: string
  height?: string
}

const props = withDefaults(defineProps<Props>(), {
  type: 'text',
  lines: 1,
  width: '100%',
  height: 'auto',
})
</script>

<template>
  <div class="skeleton-loader" :style="{ width, height }">
    <!-- Text skeleton -->
    <template v-if="type === 'text'">
      <div
        v-for="i in lines"
        :key="i"
        class="skeleton-line"
        :class="{ 'w-3/4': i === lines }"
      />
    </template>

    <!-- Avatar skeleton -->
    <template v-else-if="type === 'avatar'">
      <div class="skeleton-avatar" />
    </template>

    <!-- Card skeleton -->
    <template v-else-if="type === 'card'">
      <div class="skeleton-card">
        <div class="skeleton-card-image" />
        <div class="skeleton-card-content">
          <div class="skeleton-line w-3/4" />
          <div class="skeleton-line w-1/2" />
        </div>
      </div>
    </template>

    <!-- Track skeleton -->
    <template v-else-if="type === 'track'">
      <div class="skeleton-track">
        <div class="skeleton-track-number" />
        <div class="skeleton-track-art" />
        <div class="skeleton-track-info">
          <div class="skeleton-line w-48" />
          <div class="skeleton-line w-32" />
        </div>
        <div class="skeleton-track-duration" />
      </div>
    </template>

    <!-- Playlist skeleton -->
    <template v-else-if="type === 'playlist'">
      <div class="skeleton-playlist">
        <div class="skeleton-playlist-cover" />
        <div class="skeleton-line w-3/4 mt-3" />
        <div class="skeleton-line w-1/2 mt-2" />
      </div>
    </template>
  </div>
</template>

<style scoped>
.skeleton-loader {
  @apply animate-pulse;
}

.skeleton-line {
  @apply h-4 bg-gray-700 rounded mb-2;
}

.skeleton-avatar {
  @apply w-12 h-12 bg-gray-700 rounded-full;
}

.skeleton-card {
  @apply bg-gray-800 rounded-lg overflow-hidden;
}

.skeleton-card-image {
  @apply w-full aspect-square bg-gray-700;
}

.skeleton-card-content {
  @apply p-4;
}

.skeleton-track {
  @apply flex items-center px-4 py-3 space-x-4;
}

.skeleton-track-number {
  @apply w-8 h-4 bg-gray-700 rounded;
}

.skeleton-track-art {
  @apply w-10 h-10 bg-gray-700 rounded;
}

.skeleton-track-info {
  @apply flex-1;
}

.skeleton-track-duration {
  @apply w-12 h-4 bg-gray-700 rounded;
}

.skeleton-playlist {
  @apply bg-gray-800 rounded-lg p-4;
}

.skeleton-playlist-cover {
  @apply w-full aspect-square bg-gray-700 rounded-lg;
}
</style>
