/**
 * Chat History - Manages chat message persistence in localStorage.
 * 
 * Requirements: 7.1, 7.2, 7.3, 7.4
 */

import type { ChatMessage } from '@/types'

const STORAGE_KEY = 'music_chat_history'
const MAX_MESSAGES = 100

/**
 * Serialize ChatMessage for storage.
 * Converts Date to ISO string.
 */
function serializeMessage(message: ChatMessage): any {
  return {
    ...message,
    timestamp: message.timestamp.toISOString()
  }
}

/**
 * Deserialize stored message back to ChatMessage.
 * Converts ISO string back to Date.
 */
function deserializeMessage(stored: any): ChatMessage {
  return {
    ...stored,
    timestamp: new Date(stored.timestamp)
  }
}

/**
 * Save chat history to localStorage.
 * Limits to MAX_MESSAGES most recent messages.
 * 
 * Requirements: 7.1, 7.4
 * 
 * @param messages Array of chat messages to save
 */
export function saveChatHistory(messages: ChatMessage[]): void {
  try {
    // Keep only the most recent MAX_MESSAGES
    const toSave = messages.slice(-MAX_MESSAGES)
    const serialized = toSave.map(serializeMessage)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(serialized))
  } catch (error) {
    console.error('Failed to save chat history:', error)
  }
}

/**
 * Load chat history from localStorage.
 * 
 * Requirements: 7.2
 * 
 * @returns Array of chat messages or empty array if none found
 */
export function loadChatHistory(): ChatMessage[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (!stored) return []
    
    const parsed = JSON.parse(stored)
    if (!Array.isArray(parsed)) return []
    
    return parsed.map(deserializeMessage)
  } catch (error) {
    console.error('Failed to load chat history:', error)
    return []
  }
}

/**
 * Clear all chat history from localStorage.
 * 
 * Requirements: 7.3
 */
export function clearChatHistory(): void {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch (error) {
    console.error('Failed to clear chat history:', error)
  }
}

/**
 * Get the maximum number of messages allowed.
 * 
 * Requirements: 7.4
 */
export function getMaxMessages(): number {
  return MAX_MESSAGES
}

/**
 * Add a message to history and save.
 * Automatically enforces MAX_MESSAGES limit.
 * 
 * @param messages Current messages array
 * @param newMessage Message to add
 * @returns Updated messages array
 */
export function addMessageToHistory(
  messages: ChatMessage[],
  newMessage: ChatMessage
): ChatMessage[] {
  const updated = [...messages, newMessage]
  
  // Enforce limit
  const limited = updated.slice(-MAX_MESSAGES)
  
  // Save to storage
  saveChatHistory(limited)
  
  return limited
}
