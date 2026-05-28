<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { RouterLink, RouterView, useRoute, useRouter } from 'vue-router'
import QueueStatus from '@/components/QueueStatus.vue'
import PlayerBar from '@/components/PlayerBar.vue'
import { toastMessage } from '@/stores/playerStore'
import { useAuthStore } from '@/stores/authStore'
import { useFriendsStore } from '@/stores/friendsStore'

const BUILD_VERSION = '2026-03-23-ui-v2'
const NAVBAR_COLLAPSE_SCROLL_Y = 112
const NAVBAR_EXPAND_SCROLL_Y = 16

console.log(`Music AI Frontend v${BUILD_VERSION}`)

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()
const friendsStore = useFriendsStore()

const toast = computed(() => toastMessage.value)
const isAuthenticated = computed(() => authStore.isAuthenticated)
const isAdmin = computed(() => authStore.isAdmin)
const currentUser = computed(() => authStore.user)
const hideAppChrome = computed(() => Boolean(route.meta.hideAppChrome))
const isNavbarCompact = ref(false)
const friendsUnreadTotal = computed(() => (
    isAuthenticated.value ? friendsStore.unreadTotal : 0
))

let scrollTicking = false
let scrollRestoring = false

type NavbarDebugApi = {
    snapshot: () => void
    forceCompact: (value: boolean) => void
    sync: () => void
}

type NavbarDebugWindow = Window & typeof globalThis & {
    __appNavbarDebug?: NavbarDebugApi
}

const primaryLinks = computed(() => {
    if (!isAuthenticated.value) {
        return [
            { to: '/login', label: 'Login' },
            { to: '/register', label: 'Register' },
        ]
    }

    const links = [
        { to: '/account', label: 'Account' },
        { to: '/', label: 'Library' },
        { to: '/upload', label: 'Import' },
        { to: '/recommendations', label: 'Recommendations' },
        { to: '/playlists', label: 'Playlists' },
        { to: '/patterns', label: 'Patterns' },
        { to: '/friends', label: 'Friends' },
        { to: '/chat', label: 'Chat' },
    ]

    if (isAdmin.value) {
        links.push(
            { to: '/admin', label: 'Admin' },
            { to: '/debug', label: 'Debug' },
        )
    }

    return links
})

const roleChipClass = computed(() => (
    isAdmin.value
        ? 'badge-soft !border-indigo-200 !bg-indigo-50 !text-indigo-700'
        : 'badge-soft'
))

const toastClass = computed(() => {
    if (!toast.value) return ''
    const base = 'fixed left-1/2 top-16 z-[100] w-max max-w-[calc(100vw-1.5rem)] -translate-x-1/2 rounded-2xl px-4 py-3 text-sm font-semibold text-white shadow-2xl transition-all sm:top-20 sm:px-5'
    switch (toast.value.type) {
        case 'error':
            return `${base} bg-rose-600`
        case 'warning':
            return `${base} bg-amber-500`
        case 'info':
            return `${base} bg-sky-600`
        default:
            return `${base} bg-slate-900`
    }
})

const logout = async () => {
    friendsStore.disconnect()
    await authStore.logout()
    await router.replace('/login')
}

const linkBadge = (to: string) => (
    to === '/friends' ? friendsUnreadTotal.value : 0
)

const restoreNavbarScrollPosition = (targetScrollY: number) => {
    const normalizedTargetScrollY = Math.max(0, targetScrollY)

    nextTick(() => {
        window.requestAnimationFrame(() => {
            const currentScrollY = window.scrollY || window.pageYOffset || 0

            if (Math.abs(currentScrollY - normalizedTargetScrollY) < 0.5) {
                return
            }

            scrollRestoring = true
            window.scrollTo({ top: normalizedTargetScrollY, behavior: 'auto' })

            window.requestAnimationFrame(() => {
                scrollRestoring = false
                scrollTicking = false
            })
        })
    })
}

const syncNavbarCompact = () => {
    const currentScrollY = window.scrollY || window.pageYOffset || 0

    if (!isNavbarCompact.value && currentScrollY >= NAVBAR_COLLAPSE_SCROLL_Y) {
        isNavbarCompact.value = true
        restoreNavbarScrollPosition(currentScrollY)
    } else if (isNavbarCompact.value && currentScrollY <= NAVBAR_EXPAND_SCROLL_Y) {
        isNavbarCompact.value = false
        restoreNavbarScrollPosition(currentScrollY)
    }

    scrollTicking = false
}

const onWindowScroll = () => {
    if (scrollRestoring) {
        return
    }

    if (scrollTicking) {
        return
    }

    scrollTicking = true
    window.requestAnimationFrame(syncNavbarCompact)
}

onMounted(() => {
    if (isAuthenticated.value) {
        void friendsStore.bootstrap()
    }

    const debugWindow = window as NavbarDebugWindow
    debugWindow.__appNavbarDebug = {
        snapshot: () => {},
        forceCompact: (value: boolean) => {
            isNavbarCompact.value = value
            scrollTicking = false
        },
        sync: () => {
            syncNavbarCompact()
        },
    }

    window.addEventListener('scroll', onWindowScroll, { passive: true })
    syncNavbarCompact()
})

onBeforeUnmount(() => {
    const debugWindow = window as NavbarDebugWindow
    delete debugWindow.__appNavbarDebug
    window.removeEventListener('scroll', onWindowScroll)
})

watch(
    () => route.fullPath,
    () => {
        window.requestAnimationFrame(syncNavbarCompact)
    },
)

watch(isAuthenticated, (authenticated) => {
    if (authenticated) {
        void friendsStore.bootstrap()
        return
    }

    friendsStore.disconnect()
})
</script>

<template>
    <div class="app-shell">
        <template v-if="hideAppChrome">
            <RouterView />
        </template>
        <template v-else>
            <header
                class="shell-panel app-navbar"
                :class="{ 'app-navbar--compact': isNavbarCompact }"
            >
                <div class="app-navbar__inner flex flex-col gap-4 sm:gap-5">
                    <div class="app-navbar__hero flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                        <div class="app-navbar__identity flex items-start gap-3 sm:gap-4">
                            <RouterLink
                                :to="isAuthenticated ? '/account' : '/login'"
                                class="brand-mark shrink-0"
                            >
                                MA
                            </RouterLink>

                            <div class="app-navbar__intro min-w-0">
                                <p class="brand-copy">Music AI Workspace</p>
                                <RouterLink
                                    :to="isAuthenticated ? '/account' : '/login'"
                                    class="brand-title mt-1 inline-block"
                                >
                                    Personalized music tooling with shared admin controls
                                </RouterLink>
                                <p class="app-navbar__summary mt-2 hidden max-w-2xl text-sm leading-6 text-slate-600 sm:block">
                                    One frontend shell for library work, recommendations, imports, chat, and the user-scoped model tooling behind them.
                                </p>
                            </div>
                        </div>

                        <nav
                            class="app-navbar__nav app-navbar__nav--compact hidden flex-1 items-center justify-center gap-2 lg:flex"
                            aria-label="Primary compact navigation"
                        >
                            <RouterLink
                                v-for="link in primaryLinks"
                                :key="`compact-${link.to}`"
                                :to="link.to"
                                class="nav-chip"
                                active-class="nav-chip-active"
                                exact-active-class="nav-chip-active"
                            >
                                <span>{{ link.label }}</span>
                                <span
                                    v-if="linkBadge(link.to) > 0"
                                    class="ml-2 inline-flex min-w-5 items-center justify-center rounded-full bg-sky-600 px-1.5 py-0.5 text-[10px] font-semibold text-white"
                                >
                                    {{ linkBadge(link.to) }}
                                </span>
                            </RouterLink>
                        </nav>

                        <div class="app-navbar__actions flex w-full flex-col items-stretch gap-2 sm:w-auto sm:flex-row sm:flex-wrap sm:items-center sm:gap-3">
                            <span v-if="!isNavbarCompact" class="badge-soft hidden sm:inline-flex">
                                Build {{ BUILD_VERSION }}
                            </span>

                            <template v-if="isAuthenticated && currentUser">
                                <span :class="roleChipClass">
                                    {{ currentUser.role }}
                                </span>
                                <span class="badge-soft !normal-case !tracking-normal">
                                    {{ currentUser.name }}
                                </span>
                                <button
                                    type="button"
                                    class="action-secondary w-full sm:w-auto"
                                    @click="logout"
                                >
                                    Logout
                                </button>
                            </template>
                        </div>
                    </div>

                    <nav class="app-navbar__nav app-navbar__nav--rail flex gap-2 overflow-x-auto pb-1 sm:flex-wrap sm:overflow-visible sm:pb-0">
                        <RouterLink
                            v-for="link in primaryLinks"
                            :key="link.to"
                            :to="link.to"
                            class="nav-chip shrink-0"
                            active-class="nav-chip-active"
                            exact-active-class="nav-chip-active"
                        >
                            <span>{{ link.label }}</span>
                            <span
                                v-if="linkBadge(link.to) > 0"
                                class="ml-2 inline-flex min-w-5 items-center justify-center rounded-full bg-sky-600 px-1.5 py-0.5 text-[10px] font-semibold text-white"
                            >
                                {{ linkBadge(link.to) }}
                            </span>
                        </RouterLink>
                    </nav>
                </div>
            </header>

            <main class="app-content pb-28 sm:pb-24">
                <RouterView />
            </main>

            <QueueStatus v-if="isAdmin" />

            <Transition name="toast">
                <div v-if="toast" :class="toastClass">
                    {{ toast.text }}
                </div>
            </Transition>

            <PlayerBar v-if="isAuthenticated" />
        </template>
    </div>
</template>

<style>
.app-navbar {
    transition:
        box-shadow 0.24s ease,
        background-color 0.24s ease,
        backdrop-filter 0.24s ease;
}

.app-navbar__actions {
    transition: transform 0.24s ease;
}

.app-navbar__intro {
    max-width: 42rem;
    max-height: 12rem;
    opacity: 1;
    overflow: hidden;
    transform: translateY(0);
    transition:
        opacity 0.22s ease,
        transform 0.22s ease;
}

.app-navbar__summary,
.brand-copy,
.brand-title {
    transition:
        opacity 0.22s ease,
        transform 0.22s ease;
}

.app-navbar__nav {
    transition:
        transform 0.24s ease,
        opacity 0.22s ease;
}

.app-navbar__nav--compact {
    max-width: 0;
    opacity: 0;
    overflow: hidden;
    transform: translateY(-4px);
    pointer-events: none;
}

.app-navbar__nav--rail {
    max-height: 6rem;
    opacity: 1;
    overflow: hidden;
}

.app-navbar--compact {
    padding-top: 0.85rem;
    padding-bottom: 0.85rem;
}

.app-navbar--compact .app-navbar__inner {
    gap: 0.7rem;
}

.app-navbar--compact .app-navbar__hero {
    gap: 0.75rem;
}

.app-navbar--compact .app-navbar__identity {
    align-items: center;
}

.app-navbar--compact .app-navbar__actions {
    transform: translateY(-1px);
}

.app-navbar--compact .app-navbar__intro {
    max-width: 0;
    max-height: 0;
    opacity: 0;
    margin: 0;
    transform: translateY(-10px);
    pointer-events: none;
}

.app-navbar--compact .app-navbar__nav {
    transform: translateY(-1px);
}

.app-navbar--compact .app-navbar__nav--compact {
    max-width: 100%;
    opacity: 1;
    pointer-events: auto;
    transform: translateY(0);
}

.app-navbar--compact .app-navbar__nav--rail {
    max-height: 0;
    opacity: 0;
    margin-top: 0;
    pointer-events: none;
    transform: translateY(-8px);
}

@media (max-width: 1023px) {
    .app-navbar--compact .app-navbar__hero {
        gap: 0.9rem;
    }

    .app-navbar--compact .app-navbar__intro {
        max-width: 100%;
    }

    .app-navbar__nav--compact {
        display: none !important;
    }

    .app-navbar__nav--rail {
        max-height: 8rem;
        opacity: 1;
        pointer-events: auto;
        transform: none;
    }
}

@media (max-width: 639px) {
    .app-navbar__nav--rail {
        scrollbar-width: none;
    }

    .app-navbar__nav--rail::-webkit-scrollbar {
        display: none;
    }
}

.toast-enter-active,
.toast-leave-active {
    transition: all 0.3s ease;
}

.toast-enter-from,
.toast-leave-to {
    opacity: 0;
    transform: translate(-50%, -18px);
}
</style>
