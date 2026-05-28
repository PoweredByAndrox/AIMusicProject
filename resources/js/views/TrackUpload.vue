<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRouter } from 'vue-router'
import { toast } from 'vue3-toastify'
import api from '@/api/client'

const router = useRouter()

const activeTab = ref<'upload' | 'youtube' | 'batch'>('upload')

const isDragging = ref(false)
const uploadProgress = ref(0)
const uploading = ref(false)
const uploadError = ref('')

const youtubeUrl = ref('')
const importing = ref(false)
const importError = ref('')
const importStatus = ref('')

const batchUrls = ref('')
const batchImporting = ref(false)
const batchError = ref('')
const batchStatus = ref('')

const allowedFormats = ['audio/mpeg', 'audio/wav', 'audio/mp3']
const maxFileSize = 100 * 1024 * 1024

const youtubeUrlValid = computed(() => (
    !youtubeUrl.value.trim() || isValidYouTubeUrl(youtubeUrl.value.trim())
))

const batchUrlCount = computed(() => (
    batchUrls.value.split('\n').filter((url) => url.trim()).length
))

const importStats = computed(() => ([
    {
        label: 'File formats',
        value: 'MP3 / WAV',
        detail: 'direct upload with progress tracking',
    },
    {
        label: 'Max size',
        value: '100 MB',
        detail: 'per local file upload',
    },
    {
        label: 'Batch mode',
        value: batchUrlCount.value.toString(),
        detail: 'URLs currently staged',
    },
]))

const tabClass = (tab: 'upload' | 'youtube' | 'batch') => (
    activeTab.value === tab ? 'tab-button tab-button--active' : 'tab-button tab-button--idle'
)

const handleDragOver = (event: DragEvent) => {
    event.preventDefault()
    isDragging.value = true
}

const handleDragLeave = () => {
    isDragging.value = false
}

const handleDrop = (event: DragEvent) => {
    event.preventDefault()
    isDragging.value = false

    const files = event.dataTransfer?.files
    if (files && files.length > 0) {
        void handleFile(files[0])
    }
}

const handleFileSelect = (event: Event) => {
    const input = event.target as HTMLInputElement
    if (input.files && input.files.length > 0) {
        void handleFile(input.files[0])
    }
}

const handleFile = async (file: File) => {
    uploadError.value = ''

    if (!allowedFormats.includes(file.type)) {
        uploadError.value = 'Invalid file format. Please upload MP3 or WAV files.'
        toast.error('Invalid file format')
        return
    }

    if (file.size > maxFileSize) {
        uploadError.value = 'File too large. Maximum size is 100MB.'
        toast.error('File too large (max 100MB)')
        return
    }

    uploading.value = true
    uploadProgress.value = 0
    toast.info('Uploading track...')

    try {
        const track = await api.uploadTrack(file, (progress) => {
            uploadProgress.value = progress
        })
        toast.success('Track uploaded. Analysis started.')
        router.push({ name: 'track', params: { id: track.id } })
    } catch (error: any) {
        uploadError.value = error.response?.data?.message || 'Upload failed. Please try again.'
        toast.error(uploadError.value)
    } finally {
        uploading.value = false
    }
}

const importFromYouTube = async () => {
    if (!youtubeUrl.value.trim()) {
        importError.value = 'Please enter a YouTube URL.'
        toast.warning(importError.value)
        return
    }

    if (!youtubeUrlValid.value) {
        importError.value = 'Enter a valid YouTube video, playlist, or channel URL.'
        toast.warning(importError.value)
        return
    }

    importError.value = ''
    importing.value = true
    importStatus.value = 'Starting import...'
    toast.info('Starting YouTube import...')

    try {
        const response = await api.importYouTube({ url: youtubeUrl.value.trim() })
        importStatus.value = response.message
        toast.success(response.message)
        youtubeUrl.value = ''

        setTimeout(() => {
            router.push({ name: 'library' })
        }, 2000)
    } catch (error: any) {
        importError.value = error.response?.data?.message || 'Import failed. Please check the URL and try again.'
        toast.error(importError.value)
    } finally {
        importing.value = false
    }
}

const importBatch = async () => {
    if (!batchUrls.value.trim()) {
        batchError.value = 'Enter one or more YouTube URLs.'
        toast.warning(batchError.value)
        return
    }

    batchError.value = ''
    batchImporting.value = true
    batchStatus.value = 'Queueing URLs for import...'
    toast.info('Starting batch import...')

    try {
        const response = await api.importYouTubeBatch({ urls: batchUrls.value.trim() })
        batchStatus.value = `Queued: ${response.queued} • Invalid: ${response.invalid}`
        toast.success(`Queued ${response.queued} import task(s).`)

        if (response.errors && response.errors.length > 0) {
            batchError.value = response.errors.join('\n')
        }

        batchUrls.value = ''

        setTimeout(() => {
            router.push({ name: 'library' })
        }, 3000)
    } catch (error: any) {
        batchError.value = error.response?.data?.message || 'Batch import failed.'
        toast.error(batchError.value)
    } finally {
        batchImporting.value = false
    }
}

const isValidYouTubeUrl = (url: string): boolean => {
    const patterns = [
        /^https?:\/\/(www\.)?youtube\.com\/watch\?v=[\w-]+/,
        /^https?:\/\/youtu\.be\/[\w-]+/,
        /^https?:\/\/(www\.)?youtube\.com\/playlist\?list=[\w-]+/,
        /^https?:\/\/(www\.)?youtube\.com\/@[\w-]+/,
        /^https?:\/\/(www\.)?youtube\.com\/channel\/[\w-]+/,
    ]

    return patterns.some((pattern) => pattern.test(url))
}
</script>

<template>
    <div class="space-y-8">
        <section class="surface-card">
            <div class="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p class="section-kicker">Import</p>
                    <h1 class="page-title">Bring new music into the workspace</h1>
                    <p class="page-copy">
                        Upload local audio, pull individual YouTube sources, or queue a whole batch for background processing.
                    </p>
                </div>

                <div class="flex flex-wrap gap-3">
                    <span class="badge-soft">User-scoped likes and training signals stay attached to your account</span>
                </div>
            </div>
        </section>

        <section class="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            <article
                v-for="stat in importStats"
                :key="stat.label"
                class="metric-card"
            >
                <p class="text-xs font-semibold uppercase tracking-[0.22em] text-slate-500">{{ stat.label }}</p>
                <p class="mt-3 text-3xl font-semibold text-slate-950">{{ stat.value }}</p>
                <p class="mt-2 text-sm text-slate-500">{{ stat.detail }}</p>
            </article>
        </section>

        <section class="surface-card space-y-6">
            <div class="flex flex-wrap gap-3">
                <button type="button" :class="tabClass('upload')" @click="activeTab = 'upload'">
                    Local file
                </button>
                <button type="button" :class="tabClass('youtube')" @click="activeTab = 'youtube'">
                    YouTube
                </button>
                <button type="button" :class="tabClass('batch')" @click="activeTab = 'batch'">
                    Batch queue
                </button>
            </div>

            <div v-if="activeTab === 'upload'" class="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
                <div
                    class="rounded-[1.8rem] border-2 border-dashed p-8 text-center transition"
                    :class="isDragging ? 'border-sky-500 bg-sky-50' : 'border-slate-300 bg-slate-50/80'"
                    @dragover="handleDragOver"
                    @dragleave="handleDragLeave"
                    @drop="handleDrop"
                >
                    <div v-if="uploading" class="mx-auto max-w-md space-y-5">
                        <div class="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-sky-100 text-sm font-semibold uppercase tracking-[0.2em] text-sky-700">
                            {{ uploadProgress }}%
                        </div>
                        <div>
                            <p class="text-xl font-semibold text-slate-950">Uploading track...</p>
                            <p class="mt-2 text-sm text-slate-500">The file is on its way to the API and analysis queue.</p>
                        </div>
                        <div class="h-3 rounded-full bg-slate-200">
                            <div class="h-3 rounded-full bg-gradient-to-r from-blue-600 to-sky-500 transition-all" :style="{ width: `${uploadProgress}%` }"></div>
                        </div>
                    </div>

                    <div v-else class="mx-auto max-w-lg">
                        <div class="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-slate-950 text-sm font-semibold uppercase tracking-[0.2em] text-white">
                            WAV
                        </div>
                        <h2 class="mt-6 text-2xl font-semibold text-slate-950">Drop an audio file here</h2>
                        <p class="mt-3 text-sm leading-7 text-slate-500">
                            Drag in a track or open the file picker. Supported formats: MP3 and WAV up to 100MB.
                        </p>
                        <label class="action-primary mt-6 cursor-pointer">
                            Browse files
                            <input
                                type="file"
                                accept=".mp3,.wav,audio/mpeg,audio/wav"
                                class="hidden"
                                @change="handleFileSelect"
                            >
                        </label>
                    </div>
                </div>

                <div class="rounded-[1.8rem] border border-slate-200 bg-white/75 p-6">
                    <p class="section-kicker">What happens next</p>
                    <div class="mt-5 space-y-4">
                        <div class="rounded-2xl bg-slate-50 p-4">
                            <p class="text-sm font-semibold text-slate-900">1. Upload</p>
                            <p class="mt-2 text-sm leading-6 text-slate-600">The file is stored and a track record is created immediately.</p>
                        </div>
                        <div class="rounded-2xl bg-slate-50 p-4">
                            <p class="text-sm font-semibold text-slate-900">2. Analysis</p>
                            <p class="mt-2 text-sm leading-6 text-slate-600">ML workers measure genre, mood, BPM, loudness, and segment structure.</p>
                        </div>
                        <div class="rounded-2xl bg-slate-50 p-4">
                            <p class="text-sm font-semibold text-slate-900">3. Playback</p>
                            <p class="mt-2 text-sm leading-6 text-slate-600">The track becomes available in the unified player path once processing is done.</p>
                        </div>
                    </div>
                </div>

                <div v-if="uploadError" class="xl:col-span-2 rounded-[1.4rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-700">
                    {{ uploadError }}
                </div>
            </div>

            <div v-else-if="activeTab === 'youtube'" class="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
                <div class="rounded-[1.8rem] border border-slate-200 bg-white/80 p-6">
                    <label class="field-label" for="youtube-url">YouTube URL</label>
                    <input
                        id="youtube-url"
                        v-model="youtubeUrl"
                        type="url"
                        placeholder="https://www.youtube.com/watch?v=..."
                        class="field-control"
                        :disabled="importing"
                    >
                    <p class="mt-3 text-sm leading-6" :class="youtubeUrlValid ? 'text-slate-500' : 'text-rose-600'">
                        Supports videos, playlists, channels, and @handle URLs.
                    </p>

                    <button
                        type="button"
                        class="action-primary mt-5 w-full"
                        :disabled="importing || !youtubeUrl.trim() || !youtubeUrlValid"
                        @click="importFromYouTube"
                    >
                        {{ importing ? 'Importing...' : 'Import from YouTube' }}
                    </button>

                    <div v-if="importStatus" class="mt-5 rounded-[1.4rem] border border-sky-200 bg-sky-50 px-5 py-4 text-sm font-medium text-sky-700">
                        {{ importStatus }}
                    </div>

                    <div v-if="importError" class="mt-5 rounded-[1.4rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-700">
                        {{ importError }}
                    </div>
                </div>

                <div class="rounded-[1.8rem] border border-amber-200 bg-amber-50/90 p-6">
                    <p class="section-kicker !text-amber-700">Usage note</p>
                    <h2 class="mt-3 text-2xl font-semibold text-slate-950">Import responsibly</h2>
                    <ul class="mt-5 space-y-3 text-sm leading-6 text-amber-900">
                        <li>Only import content you have the right to download and use.</li>
                        <li>Playlist and channel imports can enqueue many jobs, so they may take time to finish.</li>
                        <li>New items appear in the library once the corresponding track records are created.</li>
                    </ul>
                </div>
            </div>

            <div v-else class="grid gap-6 xl:grid-cols-[1.05fr_0.95fr]">
                <div class="rounded-[1.8rem] border border-slate-200 bg-white/80 p-6">
                    <label class="field-label" for="batch-urls">One URL per line</label>
                    <textarea
                        id="batch-urls"
                        v-model="batchUrls"
                        rows="11"
                        class="field-textarea font-mono text-sm"
                        placeholder="https://www.youtube.com/watch?v=abc123
https://www.youtube.com/watch?v=def456
https://youtu.be/ghi789"
                        :disabled="batchImporting"
                    ></textarea>

                    <div class="mt-3 flex flex-wrap items-center gap-3 text-sm text-slate-500">
                        <span class="badge-soft !normal-case !tracking-normal">{{ batchUrlCount }} URL(s) staged</span>
                        <span>Videos, playlists, and channels are supported.</span>
                    </div>

                    <button
                        type="button"
                        class="action-primary mt-5 w-full"
                        :disabled="batchImporting || !batchUrls.trim()"
                        @click="importBatch"
                    >
                        {{ batchImporting ? 'Queueing import...' : `Queue ${batchUrlCount} URL(s)` }}
                    </button>

                    <div v-if="batchStatus" class="mt-5 rounded-[1.4rem] border border-sky-200 bg-sky-50 px-5 py-4 text-sm font-medium text-sky-700">
                        {{ batchStatus }}
                    </div>

                    <div v-if="batchError" class="mt-5 whitespace-pre-line rounded-[1.4rem] border border-rose-200 bg-rose-50 px-5 py-4 text-sm font-medium text-rose-700">
                        {{ batchError }}
                    </div>
                </div>

                <div class="rounded-[1.8rem] border border-slate-200 bg-slate-50/85 p-6">
                    <p class="section-kicker">Batch workflow</p>
                    <h2 class="mt-3 text-2xl font-semibold text-slate-950">Best for large imports</h2>
                    <div class="mt-5 space-y-4 text-sm leading-6 text-slate-600">
                        <p>Use batch mode when you already have a prepared list of source URLs and want the queue to process them in the background.</p>
                        <p>For large channels, collecting URLs first is still the safest path. A typical local helper command is:</p>
                        <code class="block rounded-2xl bg-slate-950 px-4 py-3 text-xs text-white">
                            yt-dlp --flat-playlist --print url "https://www.youtube.com/@channel"
                        </code>
                        <p>Paste the resulting list here and let the queue process it gradually.</p>
                    </div>
                </div>
            </div>
        </section>
    </div>
</template>
