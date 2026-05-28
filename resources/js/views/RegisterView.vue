<script setup lang="ts">
import { computed, reactive, ref } from 'vue'
import { useRouter, useRoute, RouterLink } from 'vue-router'
import { useAuthStore } from '@/stores/authStore'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const form = reactive({
    name: '',
    email: '',
    password: '',
    password_confirmation: '',
})

const error = ref('')

const benefits = [
    'Each account gets its own ratings, fragments, listening history, and model training path.',
    'The very first account becomes admin automatically and unlocks service controls.',
    'Future radio, playlists, and chat features will stay scoped to the account that created them.',
]

const passwordsMatch = computed(() => {
    return form.password.length === 0 || form.password === form.password_confirmation
})

const submit = async () => {
    error.value = ''

    if (!passwordsMatch.value) {
        error.value = 'Passwords do not match'
        return
    }

    try {
        await authStore.register(form)
        const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/account'
        await router.replace(redirect)
    } catch (err: any) {
        error.value = err?.response?.data?.message ?? 'Registration failed'
    }
}
</script>

<template>
    <div class="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <section class="rounded-[2rem] border border-slate-200 bg-white p-8 shadow-xl lg:p-10">
            <div class="flex items-center justify-between gap-4">
                <div>
                    <p class="text-sm font-semibold uppercase tracking-[0.28em] text-emerald-700">New Workspace</p>
                    <h1 class="mt-3 text-3xl font-semibold tracking-tight text-slate-900">Create your account</h1>
                </div>
                <RouterLink
                    to="/login"
                    class="rounded-full border border-slate-200 px-4 py-2 text-sm font-medium text-slate-600 transition hover:border-slate-300 hover:bg-slate-50"
                >
                    Sign in
                </RouterLink>
            </div>

            <form class="mt-8 space-y-5" @submit.prevent="submit">
                <label class="block">
                    <span class="text-sm font-medium text-slate-700">Name</span>
                    <input
                        v-model="form.name"
                        type="text"
                        class="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 outline-none transition focus:border-emerald-400 focus:bg-white focus:ring-4 focus:ring-emerald-100"
                        placeholder="Your display name"
                        required
                    />
                </label>

                <label class="block">
                    <span class="text-sm font-medium text-slate-700">Email</span>
                    <input
                        v-model="form.email"
                        type="email"
                        class="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 outline-none transition focus:border-emerald-400 focus:bg-white focus:ring-4 focus:ring-emerald-100"
                        placeholder="you@example.com"
                        required
                    />
                </label>

                <div class="grid gap-5 md:grid-cols-2">
                    <label class="block">
                        <span class="text-sm font-medium text-slate-700">Password</span>
                        <input
                            v-model="form.password"
                            type="password"
                            class="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 outline-none transition focus:border-emerald-400 focus:bg-white focus:ring-4 focus:ring-emerald-100"
                            placeholder="Minimum 8 characters"
                            required
                            minlength="8"
                        />
                    </label>

                    <label class="block">
                        <span class="text-sm font-medium text-slate-700">Confirm Password</span>
                        <input
                            v-model="form.password_confirmation"
                            type="password"
                            class="mt-2 w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-slate-900 outline-none transition focus:border-emerald-400 focus:bg-white focus:ring-4 focus:ring-emerald-100"
                            placeholder="Repeat password"
                            required
                            minlength="8"
                        />
                    </label>
                </div>

                <p class="text-sm" :class="passwordsMatch ? 'text-slate-500' : 'text-amber-700'">
                    Passwords must match before the account is created.
                </p>

                <div v-if="error" class="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                    {{ error }}
                </div>

                <button
                    type="submit"
                    class="w-full rounded-2xl bg-slate-950 px-5 py-3 text-sm font-semibold text-white transition hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-60"
                    :disabled="authStore.loading"
                >
                    {{ authStore.loading ? 'Creating account...' : 'Create Account' }}
                </button>
            </form>

            <div class="mt-8 rounded-2xl bg-slate-50 px-4 py-4 text-sm text-slate-600">
                Already have an account?
                <RouterLink to="/login" class="font-semibold text-emerald-700 transition hover:text-emerald-800">Sign in</RouterLink>
                to continue where you left off.
            </div>
        </section>

        <section class="relative overflow-hidden rounded-[2rem] bg-gradient-to-br from-emerald-700 via-teal-700 to-slate-900 p-8 text-white shadow-2xl lg:p-12">
            <div class="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(255,255,255,0.18),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(20,184,166,0.35),transparent_30%)]"></div>
            <div class="relative">
                <p class="text-xs font-semibold uppercase tracking-[0.35em] text-emerald-100/85">Personal Music AI</p>
                <h2 class="mt-6 text-4xl font-semibold tracking-tight lg:text-5xl">Build a user-scoped workspace from day one.</h2>
                <p class="mt-5 text-sm leading-6 text-emerald-50/90 lg:text-base">
                    Registration is no longer just a backend switch. The frontend now has a dedicated entry flow for user-scoped recommendations, playlists, and admin tooling.
                </p>

                <div class="mt-8 grid gap-3">
                    <div
                        v-for="item in benefits"
                        :key="item"
                        class="rounded-2xl border border-white/15 bg-white/10 px-4 py-3 backdrop-blur"
                    >
                        <p class="text-sm leading-6 text-emerald-50">{{ item }}</p>
                    </div>
                </div>
            </div>
        </section>
    </div>
</template>
