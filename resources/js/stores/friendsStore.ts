import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { AUTH_TOKEN_STORAGE_KEY } from '@/constants/auth'
import { showToast } from '@/stores/playerStore'
import type {
  DirectChatMessage,
  FriendRequestSummary,
  FriendSearchResult,
  FriendShareDraft,
  FriendSummary,
  FriendsBootstrapState,
} from '@/types/friends'

export const FRIENDS_SHARE_DRAFT_STORAGE_KEY = 'music-ai-friends-share-draft'

type ConnectionState = 'idle' | 'connecting' | 'connected' | 'unauthorized' | 'error'

type FriendsSocketPayload =
  | { type: 'friends_bootstrap'; state: FriendsBootstrapState }
  | { type: 'friends_search_results'; query: string; users: FriendSearchResult[] }
  | { type: 'conversation_history'; userId: number; messages: DirectChatMessage[] }
  | { type: 'direct_message'; message: DirectChatMessage }
  | { type: 'friends_error'; message?: string }
  | { type: 'friends_unauthorized' }

export const useFriendsStore = defineStore('friends', () => {
  const connectionState = ref<ConnectionState>('idle')
  const viewer = ref<FriendsBootstrapState['viewer'] | null>(null)
  const friends = ref<FriendSummary[]>([])
  const incomingRequests = ref<FriendRequestSummary[]>([])
  const outgoingRequests = ref<FriendRequestSummary[]>([])
  const notifications = ref({ unread_messages: 0, pending_requests: 0, total: 0 })
  const searchResults = ref<FriendSearchResult[]>([])
  const searchQuery = ref('')
  const activeConversationUserId = ref<number | null>(null)
  const messagesByUserId = ref<Record<number, DirectChatMessage[]>>({})
  const draft = ref<FriendShareDraft | null>(readShareDraft())

  let socket: WebSocket | null = null
  let reconnectTimer: number | null = null
  let connectPromise: Promise<void> | null = null
  let hasSeenBootstrapOnce = false

  const activeConversation = computed(() => {
    const userId = activeConversationUserId.value
    if (!userId) return null
    return friends.value.find((friend) => friend.user.id === userId)
      || incomingRequests.value.find((request) => request.user.id === userId)
      || outgoingRequests.value.find((request) => request.user.id === userId)
      || null
  })

  const activeMessages = computed(() => {
    const userId = activeConversationUserId.value
    if (!userId) return []
    return messagesByUserId.value[userId] || []
  })

  const unreadTotal = computed(() => notifications.value.total)

  function readToken() {
    return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY)
  }

  function getSocketUrl() {
    const token = readToken()
    if (!token) {
      return null
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = new URL(`${protocol}//${window.location.host}/ws/friends`)
    url.searchParams.set('accessToken', token)
    return url.toString()
  }

  function scheduleReconnect() {
    if (reconnectTimer !== null) {
      return
    }

    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null
      void connect()
    }, 1500)
  }

  function clearReconnectTimer() {
    if (reconnectTimer !== null) {
      window.clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
  }

  function send(payload: Record<string, unknown>) {
    if (!socket || socket.readyState !== window.WebSocket.OPEN) {
      throw new Error('Friends socket is not connected')
    }

    const token = readToken()
    socket.send(JSON.stringify({
      ...payload,
      accessToken: token,
    }))
  }

  function applyBootstrap(state: FriendsBootstrapState) {
    const previousPending = incomingRequests.value.length
    const previousFriends = friends.value.length

    viewer.value = state.viewer
    friends.value = Array.isArray(state.friends) ? state.friends : []
    incomingRequests.value = Array.isArray(state.incoming_requests) ? state.incoming_requests : []
    outgoingRequests.value = Array.isArray(state.outgoing_requests) ? state.outgoing_requests : []
    notifications.value = state.notifications || { unread_messages: 0, pending_requests: 0, total: 0 }

    if (hasSeenBootstrapOnce) {
      if (incomingRequests.value.length > previousPending) {
        showToast('New friend request received', 'info')
      } else if (friends.value.length > previousFriends) {
        showToast('Friend request accepted', 'info')
      }
    }

    hasSeenBootstrapOnce = true
  }

  function appendMessage(message: DirectChatMessage) {
    const viewerId = Number(viewer.value?.id || 0)
    const otherUserId = Number(message.sender.id) === viewerId
      ? Number(message.recipient.id)
      : Number(message.sender.id)

    if (!otherUserId) {
      return
    }

    const existing = messagesByUserId.value[otherUserId] || []
    if (existing.some((entry) => Number(entry.id) === Number(message.id))) {
      return
    }

    messagesByUserId.value = {
      ...messagesByUserId.value,
      [otherUserId]: [...existing, message],
    }
  }

  function disconnect() {
    clearReconnectTimer()
    connectionState.value = 'idle'
    connectPromise = null
    viewer.value = null
    friends.value = []
    incomingRequests.value = []
    outgoingRequests.value = []
    notifications.value = { unread_messages: 0, pending_requests: 0, total: 0 }
    searchResults.value = []
    messagesByUserId.value = {}
    activeConversationUserId.value = null
    hasSeenBootstrapOnce = false

    if (socket) {
      try {
        socket.close()
      } catch {
        // Ignore transport shutdown errors.
      }
      socket = null
    }
  }

  async function connect(force = false) {
    const url = getSocketUrl()
    if (!url) {
      disconnect()
      return
    }

    if (!force && socket && (socket.readyState === window.WebSocket.OPEN || socket.readyState === window.WebSocket.CONNECTING)) {
      return connectPromise ?? Promise.resolve()
    }

    if (force) {
      disconnect()
    }

    connectionState.value = 'connecting'

    connectPromise = new Promise((resolve) => {
      let settled = false
      const finalize = () => {
        if (settled) {
          return
        }
        settled = true
        resolve()
      }

      const ws = new window.WebSocket(url)
      socket = ws

      ws.addEventListener('open', () => {
        connectionState.value = 'connected'
        send({ type: 'bootstrap' })
        finalize()
      })

      ws.addEventListener('message', (event) => {
        let payload: FriendsSocketPayload | null = null

        try {
          payload = JSON.parse(String(event.data)) as FriendsSocketPayload
        } catch {
          return
        }

        if (payload.type === 'friends_bootstrap') {
          applyBootstrap(payload.state)
          return
        }

        if (payload.type === 'friends_search_results') {
          searchResults.value = Array.isArray(payload.users) ? payload.users : []
          return
        }

        if (payload.type === 'conversation_history') {
          messagesByUserId.value = {
            ...messagesByUserId.value,
            [payload.userId]: Array.isArray(payload.messages) ? payload.messages : [],
          }
          return
        }

        if (payload.type === 'direct_message') {
          appendMessage(payload.message)
          const viewerId = Number(viewer.value?.id || 0)
          const isIncoming = Number(payload.message.sender.id) !== viewerId
          const senderName = payload.message.sender.name || 'Friend'
          const otherUserId = isIncoming ? Number(payload.message.sender.id) : Number(payload.message.recipient.id)

          if (isIncoming && activeConversationUserId.value !== otherUserId) {
            showToast(`New message from ${senderName}`, 'info')
          }

          if (isIncoming && activeConversationUserId.value === otherUserId) {
            void markConversationRead(otherUserId)
          }
          return
        }

        if (payload.type === 'friends_unauthorized') {
          connectionState.value = 'unauthorized'
          return
        }

        if (payload.type === 'friends_error') {
          connectionState.value = 'error'
          if (payload.message) {
            showToast(payload.message, 'error')
          }
        }
      })

      ws.addEventListener('close', () => {
        if (socket === ws) {
          socket = null
        }
        connectPromise = null
        finalize()

        if (readToken()) {
          if (connectionState.value !== 'unauthorized') {
            connectionState.value = 'idle'
            scheduleReconnect()
          }
        } else {
          connectionState.value = 'idle'
        }
      })

      ws.addEventListener('error', () => {
        connectionState.value = 'error'
        finalize()
      })
    })

    return connectPromise
  }

  async function bootstrap() {
    await connect()
    send({ type: 'bootstrap' })
  }

  async function searchUsers(query: string) {
    searchQuery.value = query
    await connect()
    send({ type: 'search_users', query })
  }

  async function sendFriendRequest(userId: number) {
    await connect()
    send({ type: 'send_friend_request', userId })
  }

  async function respondToRequest(friendshipId: number, action: 'accept' | 'decline') {
    await connect()
    send({ type: 'respond_friend_request', friendshipId, action })
  }

  async function openConversation(userId: number) {
    activeConversationUserId.value = userId
    await connect()
    send({ type: 'open_conversation', userId })
  }

  async function markConversationRead(userId: number) {
    await connect()
    send({ type: 'mark_conversation_read', userId })
  }

  async function sendDirectMessage(userId: number, text: string, share: FriendShareDraft['share'] | null = null) {
    await connect()
    send({ type: 'send_direct_message', userId, text, share })
  }

  function queueShareDraft(nextDraft: FriendShareDraft) {
    draft.value = nextDraft
    window.sessionStorage.setItem(FRIENDS_SHARE_DRAFT_STORAGE_KEY, JSON.stringify(nextDraft))
  }

  function clearShareDraft() {
    draft.value = null
    window.sessionStorage.removeItem(FRIENDS_SHARE_DRAFT_STORAGE_KEY)
  }

  function consumeShareDraft() {
    const current = draft.value
    clearShareDraft()
    return current
  }

  function readShareDraft(): FriendShareDraft | null {
    const raw = window.sessionStorage.getItem(FRIENDS_SHARE_DRAFT_STORAGE_KEY)
    if (!raw) {
      return null
    }

    try {
      return JSON.parse(raw) as FriendShareDraft
    } catch {
      window.sessionStorage.removeItem(FRIENDS_SHARE_DRAFT_STORAGE_KEY)
      return null
    }
  }

  return {
    connectionState,
    viewer,
    friends,
    incomingRequests,
    outgoingRequests,
    notifications,
    unreadTotal,
    searchQuery,
    searchResults,
    activeConversationUserId,
    activeConversation,
    activeMessages,
    draft,
    connect,
    disconnect,
    bootstrap,
    searchUsers,
    sendFriendRequest,
    respondToRequest,
    openConversation,
    markConversationRead,
    sendDirectMessage,
    queueShareDraft,
    clearShareDraft,
    consumeShareDraft,
  }
})
