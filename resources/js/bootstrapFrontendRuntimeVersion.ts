const FRONTEND_RUNTIME_STORAGE_KEY = 'music-ai-frontend-runtime-version'
const FRONTEND_RUNTIME_RELOAD_KEY = 'music-ai-frontend-runtime-reload'
const FRONTEND_RUNTIME_QUERY_KEY = '__frontend_runtime'

function normalizeUrlForRuntimeVersion(version: string): string {
    const url = new URL(window.location.href)
    url.searchParams.set(FRONTEND_RUNTIME_QUERY_KEY, version)
    return url.toString()
}

function cleanupRuntimeVersionQuery(version: string): void {
    const url = new URL(window.location.href)
    if (url.searchParams.get(FRONTEND_RUNTIME_QUERY_KEY) !== version) {
        return
    }

    url.searchParams.delete(FRONTEND_RUNTIME_QUERY_KEY)
    window.history.replaceState(window.history.state, document.title, url.toString())
}

function ensureFreshFrontendRuntime(): void {
    const runtimeVersion = __FRONTEND_RUNTIME_VERSION__

    try {
        const previousVersion = window.sessionStorage.getItem(FRONTEND_RUNTIME_STORAGE_KEY)
        const lastReloadedVersion = window.sessionStorage.getItem(FRONTEND_RUNTIME_RELOAD_KEY)

        window.sessionStorage.setItem(FRONTEND_RUNTIME_STORAGE_KEY, runtimeVersion)

        if (previousVersion && previousVersion !== runtimeVersion && lastReloadedVersion !== runtimeVersion) {
            window.sessionStorage.setItem(FRONTEND_RUNTIME_RELOAD_KEY, runtimeVersion)
            window.location.replace(normalizeUrlForRuntimeVersion(runtimeVersion))
            return
        }

        if (lastReloadedVersion === runtimeVersion) {
            window.sessionStorage.removeItem(FRONTEND_RUNTIME_RELOAD_KEY)
        }

        cleanupRuntimeVersionQuery(runtimeVersion)
    } catch {
        // Storage can fail in hardened browser contexts. Do nothing in that case.
    }
}

ensureFreshFrontendRuntime()
