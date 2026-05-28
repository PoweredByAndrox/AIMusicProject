<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { useRouter, useRoute, RouterLink } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const form = reactive({
    email: '',
    password: '',
})

const error = ref('')

const highlights = [
    'Your personal recommendation model and track history stay tied to your account.',
    'Admin users unlock queue diagnostics, training controls, and service health tools.',
    'One login gives you the player, recommendations, playlists, radio, and chat flows.',
]

const redirectLabel = computed(() => {
    return typeof route.query.redirect === 'string' ? route.query.redirect : null
})

const submit = async () => {
    error.value = ''

    try {
        await authStore.login(form)
        const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/account'
        await router.replace(redirect)
    } catch (err: any) {
        error.value = err?.response?.data?.message ?? 'Login failed'
    }
}
</script>

<template>
    <div class="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <section class="relative overflow-hidden rounded-[2rem] bg-gradient-to-br from-slate-950 via-sky-900 to-cyan-600 p-8 text-white shadow-2xl lg:p-12">
            <div class="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(255,255,255,0.28),transparent_32%),radial-gradient(circle_at_bottom_left,rgba(34,211,238,0.28),transparent_28%)]"></div>
            <div class="relative">
                <p class="text-xs font-semibold uppercase tracking-[0.35em] text-cyan-100/80">Music AI Workspace</p>
                <h1 class="mt-6 max-w-xl text-4xl font-semibold tracking-tight lg:text-5xl">Sign in to your music workspace.</h1>
                <p class="mt-5 max-w-2xl text-sm leading-6 text-sky-50/85 lg:text-base">
                    Keep your recommendations, ratings, fragments, playlists, and admin tooling tied to one account instead of one browser session.
                </p>

                <div class="mt-8 grid gap-3">
                    <div
                        v-for="item in highlights"
                        :key="item"
                        class="rounded-2xl border border-white/15 bg-white/10 px-4 py-3 backdrop-blur"
                    >
                        <p class="text-sm leading-6 text-sky-50">{{ item }}</p>
                    </div>
                </div>

                <div v-if="redirectLabel" class="mt-8 rounded-2xl border border-amber-200/30 bg-amber-50/10 px-4 py-3 text-sm text-amber-50 backdrop-blur">
                    After login we will send you back to <span class="font-semibold">{{ redirectLabel }}</span>.
                </div>
            </div>
        </section>

        <section class="rounded-[2rem] border border-slate-200 bg-white p-8 shadow-xl lg:p-10">
            <div class="flex items-center justify-between gap-4">
                <div>
                    <p class="text-sm font-semibold uppercase tracking-[0.28em] text-sky-700">Account Access</p>
                    <h2 class="mt-3 text-3xl font-semibold tracking-tight text-slate-900">Welcome back</h2>
                </div>
                <RouterLink
                    to="/register"
                    class="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
                >
                    Create account
                </RouterLink>
            </div>

            <form class="mt-8 space-y-5" @submit.prevent="submit">
                <label class="block">
                    <span class="text-sm font-medium text-slate-700">Email</span>
                    <input
                        v-model="form.email"
                        type="email"
                        class="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 outline-none transition focus:border-sky-400 focus:bg-white focus:ring-4 focus:ring-sky-100"
                        placeholder="you@example.com"
                        required
                    />
                </label>

                <label class="block">
                    <span class="text-sm font-medium text-slate-700">Password</span>
                    <input
                        v-model="form.password"
                        type="password"
                        class="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 outline-none transition focus:border-sky-400 focus:bg-white focus:ring-4 focus:ring-sky-100"
                        placeholder="Enter your password"
                        required
                    />
                </label>

                <div v-if="error" class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                    {{ error }}
                </div>

                <button
                    type="submit"
                    class="w-full rounded-2xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                    :disabled="authStore.loading"
                >
                    {{ authStore.loading ? 'Signing in...' : 'Sign In' }}
                </button>
            </form>

            <div class="mt-8 rounded-2xl bg-slate-50 px-4 py-4 text-sm text-slate-600">
                New here?
                <RouterLink to="/register" class="font-semibold text-sky-700 transition hover:text-sky-800">Create an account</RouterLink>
                and we will open your personal workspace right away.
            </div>
        </section>
    </div>
</template>
