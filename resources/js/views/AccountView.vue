<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import api from '@/api/client'
import { useAuthStore } from '@/stores/authStore'

const router = useRouter()
const authStore = useAuthStore()
const personalRadioLoading = ref(false)
const personalRadioSaving = ref(false)
const personalRadioError = ref('')
const personalRadioSaved = ref('')
const personalRadioEnabled = ref(false)
const personalRadioAccessMode = ref<'all' | 'selected_users'>('selected_users')
const personalRadioAllowedUserIds = ref<number[]>([])
const personalRadioAvailableUsers = ref<Array<{ id: number; name: string; email: string }>>([])
const personalRadioUrl = ref('')

const user = computed(() => authStore.user)
const isAdmin = computed(() => authStore.isAdmin)

const memberSince = computed(() => {
    if (!user.value?.created_at) return 'Unknown'

    return new Intl.DateTimeFormat('en', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
    }).format(new Date(user.value.created_at))
})

const quickLinks = computed(() => {
    const links = [
        {
            title: 'Library',
            description: 'Browse imported tracks and continue analysis work.',
            to: '/',
        },
        {
            title: 'Recommendations',
            description: 'See the personalized tracks your account model prefers.',
            to: '/recommendations',
        },
        {
            title: 'Playlists',
            description: 'Open AI playlists and the user-scoped playlist system.',
            to: '/playlists',
        },
        {
            title: 'Chat',
            description: 'Use the music assistant with your current account context.',
            to: '/chat',
        },
    ]

    if (isAdmin.value) {
        links.push(
            {
                title: 'Admin',
                description: 'Open training controls, queue repair, and service overview.',
                to: '/admin',
            },
            {
                title: 'Debug',
                description: 'Inspect system-level diagnostics available to admins.',
                to: '/debug',
            },
        )
    }

    return links
})

const logout = async () => {
    await authStore.logout()
    await router.replace('/login')
}

const toAbsoluteUrl = (pathOrUrl: string) => {
    try {
        return new URL(pathOrUrl, window.location.origin).toString()
    } catch {
        return pathOrUrl
    }
}

const loadPersonalRadioSettings = async () => {
    personalRadioLoading.value = true
    personalRadioError.value = ''

    try {
        const settings = await api.getPersonalRadioSettings()
        personalRadioEnabled.value = Boolean(settings.enabled)
        personalRadioAccessMode.value = settings.access_mode === 'all' ? 'all' : 'selected_users'
        personalRadioAllowedUserIds.value = (settings.allowed_user_ids || []).map((userId) => Number(userId)).filter((userId) => Number.isFinite(userId) && userId > 0)
        personalRadioAvailableUsers.value = Array.isArray(settings.available_users) ? settings.available_users : []
        personalRadioUrl.value = toAbsoluteUrl(settings.public_url)
    } catch {
        personalRadioError.value = 'Could not load personal radio settings.'
    } finally {
        personalRadioLoading.value = false
    }
}

const savePersonalRadioSettings = async () => {
    personalRadioSaving.value = true
    personalRadioSaved.value = ''
    personalRadioError.value = ''

    try {
        const settings = await api.updatePersonalRadioSettings({
            enabled: personalRadioEnabled.value,
            access_mode: personalRadioAccessMode.value,
            allowed_user_ids: personalRadioAllowedUserIds.value,
        })

        personalRadioEnabled.value = Boolean(settings.enabled)
        personalRadioAccessMode.value = settings.access_mode === 'all' ? 'all' : 'selected_users'
        personalRadioAllowedUserIds.value = (settings.allowed_user_ids || []).map((userId) => Number(userId)).filter((userId) => Number.isFinite(userId) && userId > 0)
        personalRadioAvailableUsers.value = Array.isArray(settings.available_users) ? settings.available_users : []
        personalRadioUrl.value = toAbsoluteUrl(settings.public_url)
        personalRadioSaved.value = 'Personal radio settings saved.'
    } catch {
        personalRadioError.value = 'Could not save personal radio settings.'
    } finally {
        personalRadioSaving.value = false
    }
}

const togglePersonalRadioUser = (userId: number, checked: boolean) => {
    const nextIds = new Set(personalRadioAllowedUserIds.value)
    if (checked) {
        nextIds.add(userId)
    } else {
        nextIds.delete(userId)
    }

    personalRadioAllowedUserIds.value = Array.from(nextIds.values()).sort((left, right) => left - right)
}

const onPersonalRadioUserToggle = (userId: number, event: Event) => {
    const target = event.target
    if (!(target instanceof HTMLInputElement)) {
        return
    }

    togglePersonalRadioUser(userId, target.checked)
}

onMounted(() => {
    void loadPersonalRadioSettings()
})
</script>

<template>
    <div class="space-y-6">
        <section class="relative overflow-hidden rounded-[2rem] bg-gradient-to-br from-slate-950 via-indigo-900 to-sky-700 p-8 text-white shadow-2xl lg:p-10">
            <div class="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.18),transparent_28%),radial-gradient(circle_at_bottom_left,rgba(59,130,246,0.3),transparent_30%)]"></div>
            <div class="relative flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
                <div>
                    <p class="text-xs font-semibold uppercase tracking-[0.35em] text-sky-100/80">Account Center</p>
                    <h1 class="mt-5 text-4xl font-semibold tracking-tight">{{ user?.name || 'Account' }}</h1>
                    <p class="mt-3 max-w-2xl text-sm leading-6 text-sky-50/85 lg:text-base">
                        This page is the frontend entry point for account-scoped work: identity, role-aware navigation, and quick access to the pages that now rely on user-specific data.
                    </p>
                </div>

                <div class="flex flex-wrap items-center gap-3">
                    <span class="rounded-full border border-white/20 bg-white/10 px-4 py-2 text-sm font-medium backdrop-blur">
                        {{ user?.email }}
                    </span>
                    <span class="rounded-full border border-sky-200/30 bg-sky-100/15 px-4 py-2 text-sm font-semibold uppercase tracking-[0.22em] text-sky-50 backdrop-blur">
                        {{ user?.role }}
                    </span>
                </div>
            </div>
        </section>

        <section class="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
            <div class="rounded-[2rem] border border-slate-200 bg-white p-8 shadow-xl">
                <div class="flex items-center justify-between gap-4">
                    <div>
                        <p class="text-sm font-semibold uppercase tracking-[0.28em] text-indigo-700">Identity</p>
                        <h2 class="mt-3 text-2xl font-semibold text-slate-900">Account details</h2>
                    </div>
                    <RouterLink
                        to="/recommendations"
                        class="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
                    >
                        Open recommendations
                    </RouterLink>
                </div>

                <div class="mt-8 grid gap-4 md:grid-cols-2">
                    <div class="rounded-2xl bg-slate-50 p-5">
                        <div class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Display name</div>
                        <div class="mt-3 text-xl font-semibold text-slate-900">{{ user?.name }}</div>
                    </div>
                    <div class="rounded-2xl bg-slate-50 p-5">
                        <div class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Email</div>
                        <div class="mt-3 text-xl font-semibold text-slate-900 break-all">{{ user?.email }}</div>
                    </div>
                    <div class="rounded-2xl bg-slate-50 p-5">
                        <div class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Role</div>
                        <div class="mt-3 text-xl font-semibold text-slate-900">{{ user?.role }}</div>
                    </div>
                    <div class="rounded-2xl bg-slate-50 p-5">
                        <div class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Member since</div>
                        <div class="mt-3 text-xl font-semibold text-slate-900">{{ memberSince }}</div>
                    </div>
                </div>
            </div>

            <div class="rounded-[2rem] border border-slate-200 bg-white p-8 shadow-xl">
                <p class="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-700">Session</p>
                <h2 class="mt-3 text-2xl font-semibold text-slate-900">Quick actions</h2>

                <div class="mt-6 space-y-3">
                    <RouterLink
                        v-for="link in quickLinks"
                        :key="link.to"
                        :to="link.to"
                        class="block rounded-2xl border border-slate-200 bg-slate-50 px-5 py-4 transition hover:border-slate-300 hover:bg-white hover:shadow-sm"
                    >
                        <div class="flex items-start justify-between gap-4">
                            <div>
                                <div class="text-lg font-semibold text-slate-900">{{ link.title }}</div>
                                <div class="mt-1 text-sm leading-6 text-slate-600">{{ link.description }}</div>
                            </div>
                            <span class="text-sm font-semibold text-sky-700">Open</span>
                        </div>
                    </RouterLink>
                </div>

                <div class="mt-6 rounded-2xl border border-sky-200 bg-sky-50 px-5 py-4 text-sm leading-6 text-sky-900">
                    Profile-edit endpoints are still separate work, but personal radio settings are now live on this page and save through the authenticated API.
                </div>

                <button
                    type="button"
                    class="mt-6 w-full rounded-2xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800"
                    @click="logout"
                >
                    Logout
                </button>
            </div>
        </section>

        <section class="rounded-[2rem] border border-slate-200 bg-white p-8 shadow-xl">
            <div class="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                <div>
                    <p class="text-sm font-semibold uppercase tracking-[0.28em] text-sky-700">Personal Radio</p>
                    <h2 class="mt-3 text-2xl font-semibold text-slate-900">Your private wave</h2>
                    <p class="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
                        Turn on your own recommendation-driven radio station and choose whether it is open to everyone or only to specific users from this system.
                    </p>
                </div>

                <div v-if="personalRadioUrl" class="rounded-2xl border border-slate-200 bg-slate-50 px-5 py-4 text-sm text-slate-600">
                    <div class="text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">Station URL</div>
                    <a class="mt-2 block break-all font-medium text-sky-700 hover:text-sky-900" :href="personalRadioUrl" target="_blank" rel="noopener">
                        {{ personalRadioUrl }}
                    </a>
                    <a
                        class="mt-3 inline-flex rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-700"
                        :href="personalRadioUrl"
                        target="_blank"
                        rel="noopener"
                    >
                        Open my wave
                    </a>
                </div>
            </div>

            <div class="mt-8 grid gap-6 xl:grid-cols-[0.6fr_1fr]">
                <div class="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5">
                    <div class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Availability</div>
                    <label class="mt-4 flex items-center justify-between gap-4 rounded-2xl border border-slate-200 bg-white px-4 py-4">
                        <div>
                            <div class="text-sm font-semibold text-slate-900">Enable personal radio</div>
                            <div class="mt-1 text-sm text-slate-500">
                                When enabled, your station becomes available at your personal slug.
                            </div>
                        </div>
                        <input v-model="personalRadioEnabled" type="checkbox" class="h-5 w-5 rounded border-slate-300 text-sky-600 focus:ring-sky-500">
                    </label>

                    <div v-if="personalRadioLoading" class="mt-4 text-sm text-slate-500">
                        Loading personal radio settings...
                    </div>
                    <div v-else-if="personalRadioSaved" class="mt-4 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                        {{ personalRadioSaved }}
                    </div>
                    <div v-if="personalRadioError" class="mt-4 rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
                        {{ personalRadioError }}
                    </div>
                </div>

                <div class="rounded-[1.5rem] border border-slate-200 bg-slate-50 p-5">
                    <div class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Access</div>
                    <p class="mt-3 text-sm leading-6 text-slate-600">
                        Choose whether everyone can open your wave or only specific users.
                    </p>

                    <label class="mt-4 block text-xs font-semibold uppercase tracking-[0.18em] text-slate-500" for="personal-radio-access-mode">
                        Access mode
                    </label>
                    <select
                        id="personal-radio-access-mode"
                        v-model="personalRadioAccessMode"
                        class="mt-3 w-full rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm font-medium text-slate-700 outline-none transition focus:border-sky-300 focus:ring-2 focus:ring-sky-100"
                    >
                        <option value="all">All authenticated users</option>
                        <option value="selected_users">Specific users</option>
                    </select>

                    <div v-if="personalRadioAccessMode === 'selected_users'" class="mt-5">
                        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Allowed users</div>
                        <p class="mt-3 text-sm leading-6 text-slate-600">
                            Select the users who can open your personal radio station.
                        </p>

                        <div v-if="personalRadioAvailableUsers.length === 0" class="mt-4 rounded-2xl border border-dashed border-slate-200 bg-white px-4 py-4 text-sm text-slate-500">
                            No other users are available yet.
                        </div>

                        <div v-else class="mt-4 grid gap-3">
                            <label
                                v-for="availableUser in personalRadioAvailableUsers"
                                :key="availableUser.id"
                                class="flex items-start justify-between gap-4 rounded-2xl border border-slate-200 bg-white px-4 py-4"
                            >
                                <div class="min-w-0">
                                    <div class="text-sm font-semibold text-slate-900">{{ availableUser.name }}</div>
                                    <div class="mt-1 break-all text-sm text-slate-500">{{ availableUser.email }}</div>
                                </div>
                                <input
                                    :checked="personalRadioAllowedUserIds.includes(availableUser.id)"
                                    type="checkbox"
                                    class="mt-1 h-5 w-5 rounded border-slate-300 text-sky-600 focus:ring-sky-500"
                                    @change="onPersonalRadioUserToggle(availableUser.id, $event)"
                                >
                            </label>
                        </div>
                    </div>

                    <div class="mt-4 flex flex-wrap items-center gap-3">
                        <button
                            type="button"
                            class="rounded-2xl bg-sky-600 px-5 py-3 text-sm font-semibold text-white transition hover:bg-sky-700 disabled:cursor-not-allowed disabled:bg-slate-300"
                            :disabled="personalRadioLoading || personalRadioSaving"
                            @click="savePersonalRadioSettings"
                        >
                            {{ personalRadioSaving ? 'Saving...' : 'Save personal radio' }}
                        </button>

                        <button
                            type="button"
                            class="rounded-2xl border border-slate-200 px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-white"
                            :disabled="personalRadioLoading || personalRadioSaving"
                            @click="loadPersonalRadioSettings"
                        >
                            Reload
                        </button>
                    </div>
                </div>
            </div>
        </section>
    </div>
</template>
