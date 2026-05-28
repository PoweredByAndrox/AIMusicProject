import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import { AUTH_TOKEN_STORAGE_KEY, AUTH_USER_STORAGE_KEY } from '@/constants/auth'
import { useAuthStore } from '@/stores/authStore'

const routes: RouteRecordRaw[] = [
    {
        path: '/radio/:slug',
        name: 'public-radio',
        component: () => import('@/views/PublicRadioView.vue'),
        meta: { hideAppChrome: true },
    },
    {
        path: '/embed/radio/:slug',
        name: 'public-radio-embed',
        component: () => import('@/views/PublicRadioView.vue'),
        meta: { hideAppChrome: true, embed: true },
    },
    {
        path: '/login',
        name: 'login',
        component: () => import('@/views/LoginView.vue'),
        meta: { guestOnly: true },
    },
    {
        path: '/register',
        name: 'register',
        component: () => import('@/views/RegisterView.vue'),
        meta: { guestOnly: true },
    },
    {
        path: '/',
        name: 'library',
        component: () => import('@/views/TrackLibrary.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/account',
        name: 'account',
        component: () => import('@/views/AccountView.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/upload',
        name: 'upload',
        component: () => import('@/views/TrackUpload.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/track/:id',
        name: 'track',
        component: () => import('@/views/TrackDetail.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/track/:id/tuning-lab',
        name: 'track-tuning-lab',
        component: () => import('@/views/StemTuningLabView.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/artist/:slug',
        name: 'artist',
        component: () => import('@/views/ArtistDetailView.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/recommendations',
        name: 'recommendations',
        component: () => import('@/views/RecommendationsView.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/playlists',
        name: 'playlists',
        component: () => import('@/components/PlaylistList.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/playlists/:id',
        name: 'playlist-detail',
        component: () => import('@/components/PlaylistDetail.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/patterns',
        name: 'patterns',
        component: () => import('@/views/PatternList.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/patterns/:id',
        name: 'pattern-detail',
        component: () => import('@/views/PatternDetail.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/track/:id/segments',
        name: 'segment-labeler',
        component: () => import('@/views/SegmentLabeler.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/debug',
        name: 'debug',
        component: () => import('@/views/DebugView.vue'),
        meta: { requiresAuth: true, requiresAdmin: true },
    },
    {
        path: '/admin',
        name: 'admin',
        component: () => import('@/views/AdminView.vue'),
        meta: { requiresAuth: true, requiresAdmin: true },
    },
    {
        path: '/chat',
        name: 'chat',
        component: () => import('@/views/MusicChatView.vue'),
        meta: { requiresAuth: true },
    },
    {
        path: '/friends',
        name: 'friends',
        component: () => import('@/views/FriendsView.vue'),
        meta: { requiresAuth: true },
    },
]

const router = createRouter({
    history: createWebHistory(),
    routes,
})

let authRefreshPromise: Promise<void> | null = null

async function syncSessionFromServer() {
    if (!authRefreshPromise) {
        const authStore = useAuthStore()
        authRefreshPromise = authStore.initialize().finally(() => {
            authRefreshPromise = null
        })
    }

    await authRefreshPromise
}

router.beforeEach(async (to) => {
    let token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
    let rawUser = localStorage.getItem(AUTH_USER_STORAGE_KEY)
    let user: { role?: string } | null = null

    if (rawUser) {
        try {
            user = JSON.parse(rawUser) as { role?: string }
        } catch {
            localStorage.removeItem(AUTH_USER_STORAGE_KEY)
        }
    }

    if (to.meta.requiresAuth && !token) {
        return {
            name: 'login',
            query: { redirect: to.fullPath },
        }
    }

    if (token && (!user || to.meta.requiresAdmin)) {
        await syncSessionFromServer()
        token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
        rawUser = localStorage.getItem(AUTH_USER_STORAGE_KEY)

        if (rawUser) {
            try {
                user = JSON.parse(rawUser) as { role?: string }
            } catch {
                localStorage.removeItem(AUTH_USER_STORAGE_KEY)
                user = null
            }
        } else {
            user = null
        }
    }

    if (to.meta.requiresAuth && !token) {
        return {
            name: 'login',
            query: { redirect: to.fullPath },
        }
    }

    if (to.meta.guestOnly && token) {
        return { name: 'account' }
    }

    if (to.meta.requiresAdmin && user?.role !== 'admin') {
        return { name: 'account' }
    }

    return true
})

export default router
