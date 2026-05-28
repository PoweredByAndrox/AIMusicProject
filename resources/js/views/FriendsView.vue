<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink } from 'vue-router'
import { useFriendsStore } from '@/stores/friendsStore'
import type { DirectChatMessage, DirectSegmentSharePayload, DirectTrackSharePayload, FriendSummary } from '@/types/friends'

const friendsStore = useFriendsStore()

const searchTerm = ref('')
const messageText = ref('')
const searchBusy = ref(false)
const sending = ref(false)

const activeFriend = computed(() => {
  const conversation = friendsStore.activeConversation
  if (!conversation || !('user' in conversation)) {
    return null
  }

  return conversation as FriendSummary
})

const pendingDraft = computed(() => friendsStore.draft)

function formatTimestamp(value: string | null | undefined) {
  if (!value) return 'Now'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'Now'
  }

  return new Intl.DateTimeFormat('en', {
    hour: '2-digit',
    minute: '2-digit',
    month: 'short',
    day: 'numeric',
  }).format(date)
}

function formatDuration(seconds: number | null | undefined) {
  const totalSeconds = Math.max(0, Number(seconds || 0))
  const minutes = Math.floor(totalSeconds / 60)
  const remainder = Math.floor(totalSeconds % 60)
  return `${minutes}:${remainder.toString().padStart(2, '0')}`
}

function waveformPath(peaks: number[]) {
  if (!Array.isArray(peaks) || peaks.length === 0) {
    return ''
  }

  const width = 200
  const height = 48
  const points = peaks.map((peak, index) => {
    const x = (index / Math.max(1, peaks.length - 1)) * width
    const normalized = Math.max(0, Math.min(1, Number(peak || 0)))
    const y = height - (normalized * (height - 6)) - 3
    return `${x.toFixed(2)},${y.toFixed(2)}`
  })

  return `M0 ${height / 2} L${points.join(' L')}`
}

async function runSearch() {
  searchBusy.value = true
  try {
    await friendsStore.searchUsers(searchTerm.value)
  } finally {
    searchBusy.value = false
  }
}

async function selectFriend(userId: number) {
  await friendsStore.openConversation(userId)
}

async function sendMessage() {
  const friend = activeFriend.value
  if (!friend) {
    return
  }

  const text = messageText.value.trim()
  const share = pendingDraft.value?.share || null
  if (!text && !share) {
    return
  }

  sending.value = true
  try {
    await friendsStore.sendDirectMessage(friend.user.id, text, share)
    messageText.value = ''
    if (share) {
      friendsStore.clearShareDraft()
    }
  } finally {
    sending.value = false
  }
}

function draftTitle() {
  if (!pendingDraft.value) return ''
  if (pendingDraft.value.type === 'track') {
    return pendingDraft.value.preview.title
  }

  return `${pendingDraft.value.preview.title} · ${pendingDraft.value.preview.label || 'Segment'}`
}

function draftSubtitle() {
  if (!pendingDraft.value) return ''
  if (pendingDraft.value.type === 'track') {
    return pendingDraft.value.preview.subtitle || 'Track share'
  }

  return `${formatDuration(pendingDraft.value.preview.start)} - ${formatDuration(pendingDraft.value.preview.end)}`
}

function isIncoming(message: DirectChatMessage) {
  return Number(message.sender.id) !== Number(friendsStore.viewer?.id || 0)
}

function asTrackShare(payload: DirectChatMessage['payload']) {
  return payload && payload.type === 'track' ? payload as DirectTrackSharePayload : null
}

function asSegmentShare(payload: DirectChatMessage['payload']) {
  return payload && payload.type === 'segment' ? payload as DirectSegmentSharePayload : null
}

watch(
  () => friendsStore.friends,
  (friends) => {
    if (friendsStore.activeConversationUserId || friends.length === 0) {
      return
    }

    if (pendingDraft.value) {
      return
    }

    void selectFriend(friends[0].user.id)
  },
  { deep: true },
)

onMounted(() => {
  void friendsStore.bootstrap()
})
</script>

<template>
  <div class="grid gap-6 xl:grid-cols-[20rem_minmax(0,1fr)_22rem]">
    <section class="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-xl">
      <div class="flex items-center justify-between gap-3">
        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.24em] text-sky-700">Friends</p>
          <h1 class="mt-3 text-2xl font-semibold text-slate-950">Direct chat</h1>
        </div>
        <span class="badge-soft !normal-case !tracking-normal">
          {{ friendsStore.notifications.total }} unread
        </span>
      </div>

      <div class="mt-6 rounded-[1.4rem] border border-slate-200 bg-slate-50 p-4">
        <label class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500" for="friend-search">
          Find users
        </label>
        <div class="mt-3 flex gap-3">
          <input
            id="friend-search"
            v-model="searchTerm"
            type="search"
            placeholder="Search by name or email"
            class="flex-1 rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-sky-300 focus:ring-2 focus:ring-sky-100"
            @keydown.enter.prevent="runSearch"
          >
          <button type="button" class="action-primary !rounded-xl !px-4 !py-3 !text-sm" @click="runSearch">
            {{ searchBusy ? '...' : 'Search' }}
          </button>
        </div>

        <div class="mt-4 space-y-3">
          <div
            v-for="result in friendsStore.searchResults"
            :key="result.user.id"
            class="rounded-[1rem] border border-slate-200 bg-white px-4 py-4"
          >
            <div class="flex items-start justify-between gap-4">
              <div class="min-w-0">
                <div class="text-sm font-semibold text-slate-950">{{ result.user.name }}</div>
                <div class="mt-1 break-all text-sm text-slate-500">{{ result.user.email }}</div>
              </div>
              <button
                v-if="!result.relationship"
                type="button"
                class="action-secondary !rounded-xl !px-4 !py-2 !text-xs"
                @click="friendsStore.sendFriendRequest(result.user.id)"
              >
                Add
              </button>
              <span v-else class="badge-soft !normal-case !tracking-normal">
                {{ result.relationship.status }} · {{ result.relationship.direction }}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div class="mt-6">
        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Friends</div>
        <div class="mt-3 space-y-3">
          <button
            v-for="friend in friendsStore.friends"
            :key="friend.user.id"
            type="button"
            class="flex w-full items-start justify-between gap-4 rounded-[1rem] border px-4 py-4 text-left transition"
            :class="friend.user.id === friendsStore.activeConversationUserId ? 'border-sky-300 bg-sky-50' : 'border-slate-200 bg-slate-50 hover:border-slate-300 hover:bg-white'"
            @click="selectFriend(friend.user.id)"
          >
            <div class="min-w-0">
              <div class="text-sm font-semibold text-slate-950">{{ friend.user.name }}</div>
              <div class="mt-1 truncate text-sm text-slate-500">
                {{ friend.latest_message?.text || friend.latest_message?.payload?.type || 'No messages yet' }}
              </div>
            </div>
            <span v-if="friend.unread_count > 0" class="rounded-full bg-sky-600 px-2 py-1 text-xs font-semibold text-white">
              {{ friend.unread_count }}
            </span>
          </button>
        </div>
      </div>
    </section>

    <section class="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-xl">
      <div class="flex items-center justify-between gap-3">
        <div>
          <p class="text-xs font-semibold uppercase tracking-[0.24em] text-slate-500">Conversation</p>
          <h2 class="mt-3 text-2xl font-semibold text-slate-950">
            {{ activeFriend?.user.name || 'Choose a friend' }}
          </h2>
        </div>
        <span class="badge-soft !normal-case !tracking-normal">
          {{ friendsStore.connectionState }}
        </span>
      </div>

      <div v-if="pendingDraft" class="mt-5 rounded-[1.35rem] border border-sky-200 bg-sky-50 px-4 py-4">
        <div class="text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">Pending share</div>
        <div class="mt-2 text-sm font-semibold text-sky-950">{{ draftTitle() }}</div>
        <div class="mt-1 text-sm text-sky-700">{{ draftSubtitle() }}</div>

        <svg
          v-if="pendingDraft.preview.waveform?.peaks?.length"
          class="mt-3 h-14 w-full rounded-xl border border-sky-200 bg-white p-2"
          viewBox="0 0 200 48"
          preserveAspectRatio="none"
        >
          <path
            :d="waveformPath(pendingDraft.preview.waveform.peaks)"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            class="text-sky-600"
          />
        </svg>

        <button type="button" class="mt-3 text-sm font-semibold text-sky-700 hover:text-sky-900" @click="friendsStore.clearShareDraft()">
          Clear share
        </button>
      </div>

      <div v-if="!activeFriend" class="mt-6 rounded-[1.5rem] border border-dashed border-slate-200 bg-slate-50 px-5 py-10 text-center text-sm text-slate-500">
        Choose a friend from the left column to open a persisted realtime chat.
      </div>

      <div v-else class="mt-6">
        <div class="max-h-[30rem] space-y-4 overflow-y-auto rounded-[1.5rem] border border-slate-200 bg-slate-50 p-4">
          <article
            v-for="message in friendsStore.activeMessages"
            :key="message.id"
            class="rounded-[1.15rem] px-4 py-3"
            :class="isIncoming(message) ? 'mr-10 border border-slate-200 bg-white' : 'ml-10 border border-sky-200 bg-sky-50'"
          >
            <div class="flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">
              <span>{{ message.sender.name }}</span>
              <span>{{ formatTimestamp(message.created_at) }}</span>
            </div>

            <p v-if="message.text" class="mt-3 whitespace-pre-line text-sm leading-6 text-slate-700">
              {{ message.text }}
            </p>

            <div v-if="asTrackShare(message.payload)" class="mt-3 rounded-[1rem] border border-slate-200 bg-slate-950 px-4 py-4 text-white">
              <div class="text-xs font-semibold uppercase tracking-[0.16em] text-sky-200">Track share</div>
              <div class="mt-2 text-lg font-semibold">{{ asTrackShare(message.payload)?.title }}</div>
              <div class="mt-1 text-sm text-slate-300">
                {{ asTrackShare(message.payload)?.artist || 'Unknown artist' }} · {{ formatDuration(asTrackShare(message.payload)?.duration) }}
              </div>
              <RouterLink
                :to="asTrackShare(message.payload)?.route || '/'"
                class="mt-3 inline-flex rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-700"
              >
                Open track
              </RouterLink>
            </div>

            <div v-if="asSegmentShare(message.payload)" class="mt-3 rounded-[1rem] border border-slate-200 bg-slate-950 px-4 py-4 text-white">
              <div class="flex items-center justify-between gap-3">
                <div>
                  <div class="text-xs font-semibold uppercase tracking-[0.16em] text-sky-200">Segment share</div>
                  <div class="mt-2 text-base font-semibold">{{ asSegmentShare(message.payload)?.track_title }}</div>
                  <div class="mt-1 text-sm text-slate-300">
                    {{ asSegmentShare(message.payload)?.label }} · {{ formatDuration(asSegmentShare(message.payload)?.start) }} - {{ formatDuration(asSegmentShare(message.payload)?.end) }}
                  </div>
                </div>
              </div>

              <svg
                v-if="asSegmentShare(message.payload)?.waveform?.peaks?.length"
                class="mt-3 h-14 w-full rounded-xl border border-white/10 bg-slate-900 p-2"
                viewBox="0 0 200 48"
                preserveAspectRatio="none"
              >
                <path
                  :d="waveformPath(asSegmentShare(message.payload)?.waveform.peaks || [])"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  class="text-sky-300"
                />
              </svg>

              <div class="mt-3 flex flex-wrap gap-3">
                <RouterLink
                  :to="asSegmentShare(message.payload)?.track_route || '/'"
                  class="inline-flex rounded-xl bg-sky-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-sky-700"
                >
                  Open track
                </RouterLink>
                <RouterLink
                  :to="asSegmentShare(message.payload)?.segment_route || '/'"
                  class="inline-flex rounded-xl border border-white/15 px-4 py-2 text-sm font-semibold text-sky-100 transition hover:border-sky-300 hover:text-white"
                >
                  Open segments
                </RouterLink>
              </div>
            </div>
          </article>
        </div>

        <div class="mt-4 flex gap-3">
          <textarea
            v-model="messageText"
            rows="3"
            placeholder="Write a message or send the pending share..."
            class="flex-1 rounded-[1.35rem] border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-700 outline-none transition focus:border-sky-300 focus:ring-2 focus:ring-sky-100"
          ></textarea>
          <button
            type="button"
            class="action-primary !rounded-[1.2rem] !px-5 !py-3 !text-sm"
            :disabled="sending"
            @click="sendMessage"
          >
            {{ sending ? 'Sending...' : 'Send' }}
          </button>
        </div>
      </div>
    </section>

    <section class="rounded-[2rem] border border-slate-200 bg-white p-6 shadow-xl">
      <div class="text-xs font-semibold uppercase tracking-[0.24em] text-emerald-700">Requests</div>
      <h2 class="mt-3 text-2xl font-semibold text-slate-950">Friend requests</h2>

      <div class="mt-6">
        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Incoming</div>
        <div class="mt-3 space-y-3">
          <article
            v-for="request in friendsStore.incomingRequests"
            :key="`incoming-${request.friendship_id}`"
            class="rounded-[1rem] border border-slate-200 bg-slate-50 px-4 py-4"
          >
            <div class="text-sm font-semibold text-slate-950">{{ request.user.name }}</div>
            <div class="mt-1 break-all text-sm text-slate-500">{{ request.user.email }}</div>
            <div class="mt-3 flex gap-3">
              <button type="button" class="action-primary !rounded-xl !px-4 !py-2 !text-xs" @click="friendsStore.respondToRequest(request.friendship_id, 'accept')">
                Accept
              </button>
              <button type="button" class="action-secondary !rounded-xl !px-4 !py-2 !text-xs" @click="friendsStore.respondToRequest(request.friendship_id, 'decline')">
                Decline
              </button>
            </div>
          </article>
          <p v-if="friendsStore.incomingRequests.length === 0" class="text-sm text-slate-500">No incoming requests.</p>
        </div>
      </div>

      <div class="mt-6">
        <div class="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Outgoing</div>
        <div class="mt-3 space-y-3">
          <article
            v-for="request in friendsStore.outgoingRequests"
            :key="`outgoing-${request.friendship_id}`"
            class="rounded-[1rem] border border-slate-200 bg-slate-50 px-4 py-4"
          >
            <div class="text-sm font-semibold text-slate-950">{{ request.user.name }}</div>
            <div class="mt-1 break-all text-sm text-slate-500">{{ request.user.email }}</div>
            <div class="mt-3 text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">Pending</div>
          </article>
          <p v-if="friendsStore.outgoingRequests.length === 0" class="text-sm text-slate-500">No outgoing requests.</p>
        </div>
      </div>
    </section>
  </div>
</template>
