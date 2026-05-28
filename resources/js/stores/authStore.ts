import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import api from '@/api/client'
import type { AuthSession, AuthUser } from '@/types'
import { AUTH_TOKEN_STORAGE_KEY, AUTH_USER_STORAGE_KEY } from '@/constants/auth'

function readStoredUser(): AuthUser | null {
    const raw = localStorage.getItem(AUTH_USER_STORAGE_KEY)
    if (!raw) return null

    try {
        return JSON.parse(raw) as AuthUser
    } catch {
        localStorage.removeItem(AUTH_USER_STORAGE_KEY)
        return null
    }
}

export const useAuthStore = defineStore('auth', () => {
    const token = ref<string | null>(localStorage.getItem(AUTH_TOKEN_STORAGE_KEY))
    const user = ref<AuthUser | null>(readStoredUser())
    const loading = ref(false)

    const isAuthenticated = computed(() => !!token.value && !!user.value)
    const isAdmin = computed(() => user.value?.role === 'admin')

    function persistSession(session: AuthSession) {
        token.value = session.token
        user.value = session.user
        localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, session.token)
        localStorage.setItem(AUTH_USER_STORAGE_KEY, JSON.stringify(session.user))
    }

    function clearSession() {
        token.value = null
        user.value = null
        localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY)
        localStorage.removeItem(AUTH_USER_STORAGE_KEY)
    }

    async function initialize() {
        if (!token.value) {
            clearSession()
            return
        }

        try {
            const session = await api.getCurrentUser()
            user.value = session.user
            localStorage.setItem(AUTH_USER_STORAGE_KEY, JSON.stringify(session.user))
        } catch {
            clearSession()
        }
    }

    async function register(payload: {
        name: string
        email: string
        password: string
        password_confirmation: string
    }) {
        loading.value = true
        try {
            const session = await api.register(payload)
            persistSession(session)
            return session
        } finally {
            loading.value = false
        }
    }

    async function login(payload: { email: string; password: string }) {
        loading.value = true
        try {
            const session = await api.login(payload)
            persistSession(session)
            return session
        } finally {
            loading.value = false
        }
    }

    async function logout() {
        try {
            if (token.value) {
                await api.logout()
            }
        } finally {
            clearSession()
        }
    }

    return {
        token,
        user,
        loading,
        isAuthenticated,
        isAdmin,
        initialize,
        register,
        login,
        logout,
        clearSession,
    }
})
