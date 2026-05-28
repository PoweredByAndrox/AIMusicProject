import { AUTH_TOKEN_STORAGE_KEY } from '@/constants/auth'

type FrontendLogLevel = 'debug' | 'log' | 'info' | 'warn' | 'error'
type FrontendLogValue = string | number | boolean | null | FrontendLogValue[] | { [key: string]: FrontendLogValue }

type FrontendLogEntry = {
    level: FrontendLogLevel
    message: string
    timestamp: string
    url: string
    route: string
    context: { [key: string]: FrontendLogValue }
}

type FrontendLogDebugApi = {
    getState: () => void
    forceProbe: () => void
    flushNow: () => void
}

type FrontendLogDebugWindow = Window & typeof globalThis & {
    __frontendLogDebug?: FrontendLogDebugApi
}

const FRONTEND_LOG_ENDPOINT = '/api/debug/frontend-logs'
const FRONTEND_LOG_BATCH_SIZE = 20
const FRONTEND_LOG_FLUSH_DELAY_MS = 1500
const FRONTEND_LOG_MAX_QUEUE_SIZE = 200
const FRONTEND_LOG_MAX_STRING_LENGTH = 2000
const FRONTEND_LOG_MAX_DEPTH = 4
const FRONTEND_LOG_MAX_ARRAY_LENGTH = 20
const FRONTEND_LOG_MAX_OBJECT_KEYS = 20
const FRONTEND_LOG_SHIPPING_QUERY_KEY = '__frontendLogShipping'
const FRONTEND_LOG_SHIPPING_STORAGE_KEY = '__frontend_log_shipping'

const originalConsole: Record<FrontendLogLevel, (...args: unknown[]) => void> = {
    debug: console.debug.bind(console),
    log: console.log.bind(console),
    info: console.info.bind(console),
    warn: console.warn.bind(console),
    error: console.error.bind(console),
}

const frontendLogQueue: FrontendLogEntry[] = []
let frontendLogFlushTimer: number | null = null
let frontendLogRequestInFlight = false
let frontendLogSequence = 0

const isFrontendLogShippingEnabled = (): boolean => {
    const params = new URLSearchParams(window.location.search)
    const queryFlag = params.get(FRONTEND_LOG_SHIPPING_QUERY_KEY)
    if (queryFlag === '1') {
        return true
    }

    return localStorage.getItem(FRONTEND_LOG_SHIPPING_STORAGE_KEY) === '1'
}

const truncateFrontendLogString = (value: string, maxLength = FRONTEND_LOG_MAX_STRING_LENGTH): string => (
    value.length <= maxLength ? value : `${value.slice(0, maxLength)}...`
)

const serializeFrontendLogValue = (
    value: unknown,
    depth = 0,
    seen = new WeakSet<object>(),
): FrontendLogValue => {
    if (value === null) {
        return null
    }

    if (typeof value === 'string') {
        return truncateFrontendLogString(value)
    }

    if (typeof value === 'number' || typeof value === 'boolean') {
        return value
    }

    if (typeof value === 'bigint') {
        return value.toString()
    }

    if (typeof value === 'undefined') {
        return '[undefined]'
    }

    if (typeof value === 'symbol') {
        return String(value)
    }

    if (typeof value === 'function') {
        return `[Function ${(value as Function).name || 'anonymous'}]`
    }

    if (value instanceof Error) {
        return {
            name: truncateFrontendLogString(value.name),
            message: truncateFrontendLogString(value.message),
            stack: truncateFrontendLogString(value.stack ?? ''),
        }
    }

    if (value instanceof Event) {
        return {
            type: truncateFrontendLogString(value.type),
            target: truncateFrontendLogString((value.target as Element | null)?.nodeName ?? 'unknown'),
        }
    }

    if (depth >= FRONTEND_LOG_MAX_DEPTH) {
        return `[MaxDepth ${Object.prototype.toString.call(value)}]`
    }

    if (Array.isArray(value)) {
        return value
            .slice(0, FRONTEND_LOG_MAX_ARRAY_LENGTH)
            .map((item) => serializeFrontendLogValue(item, depth + 1, seen))
    }

    if (typeof value === 'object') {
        if (seen.has(value as object)) {
            return '[Circular]'
        }

        seen.add(value as object)

        try {
            const serialized: { [key: string]: FrontendLogValue } = {}
            const entries = Object.entries(value as Record<string, unknown>).slice(0, FRONTEND_LOG_MAX_OBJECT_KEYS)

            for (const [key, entryValue] of entries) {
                serialized[key] = serializeFrontendLogValue(entryValue, depth + 1, seen)
            }

            return serialized
        } catch {
            return truncateFrontendLogString(Object.prototype.toString.call(value))
        }
    }

    return truncateFrontendLogString(String(value))
}

const getFrontendLogRoute = (): string => (
    `${window.location.pathname}${window.location.search}${window.location.hash}`
)

const scheduleFrontendLogFlush = (immediate = false): void => {
    if (frontendLogFlushTimer !== null) {
        window.clearTimeout(frontendLogFlushTimer)
    }

    frontendLogFlushTimer = window.setTimeout(() => {
        frontendLogFlushTimer = null
        void flushFrontendLogQueue()
    }, immediate ? 0 : FRONTEND_LOG_FLUSH_DELAY_MS)
}

const requeueFrontendLogBatch = (batch: FrontendLogEntry[]): void => {
    frontendLogQueue.unshift(...batch)

    if (frontendLogQueue.length > FRONTEND_LOG_MAX_QUEUE_SIZE) {
        frontendLogQueue.splice(FRONTEND_LOG_MAX_QUEUE_SIZE)
    }
}

const flushFrontendLogQueue = async (): Promise<void> => {
    if (frontendLogRequestInFlight || frontendLogQueue.length === 0) {
        return
    }

    const token = localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
    if (!token) {
        scheduleFrontendLogFlush()
        return
    }

    const batch = frontendLogQueue.splice(0, FRONTEND_LOG_BATCH_SIZE)
    frontendLogRequestInFlight = true

    try {
        const response = await fetch(FRONTEND_LOG_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': `Bearer ${token}`,
            },
            body: JSON.stringify({ logs: batch }),
            credentials: 'same-origin',
            keepalive: true,
        })

        if (!response.ok) {
            requeueFrontendLogBatch(batch)
        }
    } catch {
        requeueFrontendLogBatch(batch)
    } finally {
        frontendLogRequestInFlight = false

        if (frontendLogQueue.length > 0) {
            scheduleFrontendLogFlush(false)
        }
    }
}

const enqueueFrontendLog = (
    level: FrontendLogLevel,
    args: unknown[],
    extraContext: { [key: string]: FrontendLogValue } = {},
): void => {
    frontendLogSequence += 1

    const entry: FrontendLogEntry = {
        level,
        message: typeof args[0] === 'string' ? truncateFrontendLogString(args[0]) : `console.${level}`,
        timestamp: new Date().toISOString(),
        url: window.location.href,
        route: getFrontendLogRoute(),
        context: {
            sequence: frontendLogSequence,
            args: args.map((arg) => serializeFrontendLogValue(arg)),
            ...extraContext,
        },
    }

    frontendLogQueue.push(entry)

    if (frontendLogQueue.length > FRONTEND_LOG_MAX_QUEUE_SIZE) {
        frontendLogQueue.splice(0, frontendLogQueue.length - FRONTEND_LOG_MAX_QUEUE_SIZE)
    }

    if (level === 'error' || frontendLogQueue.length >= FRONTEND_LOG_BATCH_SIZE) {
        scheduleFrontendLogFlush(true)
        return
    }

    scheduleFrontendLogFlush()
}

const installFrontendLogCapture = (): void => {
    const debugWindow = window as FrontendLogDebugWindow

    if (!isFrontendLogShippingEnabled()) {
        debugWindow.__frontendLogDebug = {
            getState: () => {},
            forceProbe: () => {},
            flushNow: () => {},
        }
        originalConsole.info(
            '[frontend-logs] shipping disabled; enable with ?__frontendLogShipping=1 or localStorage.__frontend_log_shipping=1',
        )
        return
    }

    const patchedConsole = console as Record<FrontendLogLevel, (...args: unknown[]) => void>
    const consoleMethods: FrontendLogLevel[] = ['debug', 'log', 'info', 'warn', 'error']

    for (const method of consoleMethods) {
        patchedConsole[method] = (...args: unknown[]) => {
            originalConsole[method](...args)
            enqueueFrontendLog(method, args)
        }
    }

    window.addEventListener('error', (event) => {
        enqueueFrontendLog('error', [event.message], {
            source: truncateFrontendLogString(event.filename || ''),
            lineno: event.lineno,
            colno: event.colno,
            error: serializeFrontendLogValue(event.error),
        })
        scheduleFrontendLogFlush(true)
    })

    window.addEventListener('unhandledrejection', (event) => {
        enqueueFrontendLog('error', ['Unhandled promise rejection'], {
            reason: serializeFrontendLogValue(event.reason),
        })
        scheduleFrontendLogFlush(true)
    })

    document.addEventListener('visibilitychange', () => {
        if (document.visibilityState === 'hidden') {
            scheduleFrontendLogFlush(true)
        }
    })

    debugWindow.__frontendLogDebug = {
        getState: () => {},
        forceProbe: () => {
            scheduleFrontendLogFlush(true)
        },
        flushNow: () => {
            scheduleFrontendLogFlush(true)
        },
    }
}

installFrontendLogCapture()
if (isFrontendLogShippingEnabled()) {
    scheduleFrontendLogFlush(true)
}
