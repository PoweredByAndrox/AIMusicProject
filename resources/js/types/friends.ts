export interface FriendUser {
  id: number
  name: string
  email: string
}

export interface FriendRelationshipState {
  friendship_id: number
  status: 'pending' | 'accepted' | 'declined'
  direction: 'incoming' | 'outgoing'
}

export interface FriendSearchResult {
  user: FriendUser
  relationship: FriendRelationshipState | null
}

export interface FriendRequestSummary {
  friendship_id: number
  status: 'pending' | 'accepted' | 'declined'
  direction: 'incoming' | 'outgoing'
  user: FriendUser
  created_at: string | null
}

export interface DirectTrackSharePayload {
  type: 'track'
  track_id: number
  title: string
  artist: string | null
  duration: number
  genre: string | null
  route: string
}

export interface DirectSegmentSharePayload {
  type: 'segment'
  track_id: number
  track_title: string
  artist: string | null
  track_route: string
  segment_id: number
  segment_route: string
  label: string
  start: number
  end: number
  waveform: {
    duration: number
    peaks: number[]
  }
}

export type DirectMessageSharePayload = DirectTrackSharePayload | DirectSegmentSharePayload

export interface DirectChatMessage {
  id: number
  type: 'message' | 'track_share' | 'segment_share'
  text: string | null
  payload: DirectMessageSharePayload | null
  sender: FriendUser
  recipient: FriendUser
  created_at: string | null
  read_at: string | null
}

export interface FriendSummary {
  friendship_id: number
  user: FriendUser
  accepted_at: string | null
  latest_message: DirectChatMessage | null
  last_message_at: string | null
  unread_count: number
}

export interface FriendNotificationSummary {
  unread_messages: number
  pending_requests: number
  total: number
}

export interface FriendsBootstrapState {
  viewer: FriendUser
  friends: FriendSummary[]
  incoming_requests: FriendRequestSummary[]
  outgoing_requests: FriendRequestSummary[]
  notifications: FriendNotificationSummary
}

export interface FriendShareDraft {
  type: 'track' | 'segment'
  share: {
    type: 'track' | 'segment'
    track_id?: number
    segment_id?: number
  }
  preview: {
    title: string
    subtitle?: string | null
    label?: string | null
    start?: number
    end?: number
    waveform?: {
      duration: number
      peaks: number[]
    }
  }
}
