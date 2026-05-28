<script setup lang="ts">
import { computed } from 'vue'

interface Props {
    waveform: number[]
    spectrum: number[]
    sourceLabel: string
}

const props = defineProps<Props>()

function toPolylinePoints(values: number[]): string {
    if (values.length < 2) {
        return '0,50 100,50'
    }

    return values.map((value, index) => {
        const x = (index / (values.length - 1)) * 100
        const y = 50 - (Math.max(-1, Math.min(1, value)) * 42)
        return `${x},${y}`
    }).join(' ')
}

const waveformPoints = computed(() => toPolylinePoints(props.waveform))
const histogramRows = computed(() => props.spectrum.map((value, index) => ({
    id: index,
    value: Math.max(0, Math.min(1, value)),
})))
</script>

<template>
    <section class="surface-card !p-6">
        <div class="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
                <p class="section-kicker">Live Monitor</p>
                <h2 class="mt-3 text-2xl font-semibold tracking-tight text-slate-950">Current source analysis</h2>
                <p class="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
                    This monitor is driven by the current preview player only. It does not fake future container output.
                    Use it to inspect the source you are actually hearing right now.
                </p>
            </div>

            <div class="rounded-[1.2rem] border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm text-slate-600">
                <div class="field-label">Preview source</div>
                <div class="mt-2 text-lg font-semibold text-slate-950">{{ sourceLabel }}</div>
            </div>
        </div>

        <div class="mt-6 grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <div class="rounded-[1.35rem] border border-slate-200 bg-slate-950 p-4">
                <div class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-400">Wave scope</div>

                <div class="mt-4 overflow-hidden rounded-[1rem] border border-white/10 bg-slate-950/80">
                    <svg viewBox="0 0 100 100" preserveAspectRatio="none" class="h-40 w-full">
                        <line x1="0" y1="50" x2="100" y2="50" stroke="rgba(148,163,184,0.18)" stroke-width="0.8" />
                        <polyline
                            :points="waveformPoints"
                            fill="none"
                            stroke="rgba(56,189,248,0.96)"
                            stroke-width="1.4"
                            stroke-linecap="round"
                            stroke-linejoin="round"
                        />
                    </svg>
                </div>
            </div>

            <div class="rounded-[1.35rem] border border-slate-200 bg-white/80 p-4">
                <div class="flex items-center justify-between gap-3">
                    <div class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Spectrum histogram</div>
                    <div class="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
                        Live player FFT
                    </div>
                </div>

                <div class="mt-4 flex h-48 items-end gap-1 overflow-hidden rounded-[1rem] border border-slate-200 bg-slate-950/95 px-3 py-3">
                    <div
                        v-for="row in histogramRows"
                        :key="row.id"
                        class="flex h-full min-w-0 flex-1 items-end"
                    >
                        <div
                            class="w-full rounded-t bg-sky-400/90"
                            :style="{ height: `${Math.max(2, row.value * 100)}%` }"
                        ></div>
                    </div>
                </div>
            </div>
        </div>
    </section>
</template>
