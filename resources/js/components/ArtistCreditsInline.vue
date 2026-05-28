<script setup lang="ts">
import { computed } from 'vue'
import type { ArtistCredit } from '@/types'

const props = withDefaults(defineProps<{
    credits?: ArtistCredit[] | null
    fallback?: string | null
    emptyText?: string
    linkClass?: string
}>(), {
    credits: () => [],
    fallback: null,
    emptyText: 'Unknown artist',
    linkClass: '',
})

const normalizedCredits = computed(() => {
    const seen = new Set<string>()

    return (props.credits ?? []).filter((credit) => {
        if (!credit?.slug || !credit?.name || seen.has(credit.slug)) {
            return false
        }

        seen.add(credit.slug)
        return true
    })
})
</script>

<template>
    <span v-if="normalizedCredits.length > 0">
        <template v-for="(credit, index) in normalizedCredits" :key="credit.slug">
            <router-link
                :to="{ name: 'artist', params: { slug: credit.slug } }"
                :class="linkClass"
                @click.stop
            >
                {{ credit.name }}
            </router-link>
            <span v-if="index < normalizedCredits.length - 1">, </span>
        </template>
    </span>
    <span v-else>
        {{ fallback || emptyText }}
    </span>
</template>
