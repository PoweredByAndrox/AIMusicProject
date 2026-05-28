/**
 * Property-based tests for Chat History module.
 * 
 * **Property 12: Round-trip персистентности истории чата**
 * **Property 13: Инвариант размера истории чата**
 * **Validates: Requirements 7.1, 7.4**
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import fc from 'fast-check'
import {
  saveChatHistory,
  loadChatHistory,
  clearChatHistory,
  getMaxMessages,
  addMessageToHistory
} from './chatHistory'
import type { ChatMessage } from '@/types'

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => { store[key] = value },
    removeItem: (key: string) => { delete store[key] },
    clear: () => { store = {} }
  }
})()

Object.defineProperty(global, 'localStorage', { value: localStorageMock })

// Arbitrary for ChatMessage
const chatMessageArb = fc.record({
  id: fc.string({ minLength: 1, maxLength: 50 }),
  type: fc.constantFrom('user', 'system', 'results', 'error') as fc.Arbitrary<ChatMessage['type']>,
  content: fc.string({ maxLength: 500 }),
  timestamp: fc.date({ min: new Date('2020-01-01'), max: new Date('2030-01-01') })
})

describe('Chat History', () => {
  beforeEach(() => {
    localStorageMock.clear()
  })

  afterEach(() => {
    localStorageMock.clear()
  })

  /**
   * Property 12: Round-trip персистентности истории чата
   * 
   * For any list of messages, saving and loading returns equivalent messages.
   * 
   * **Validates: Requirements 7.1, 7.2**
   */
  describe('Property 12: Round-trip persistence', () => {
    it('saving and loading returns equivalent messages', () => {
      fc.assert(
        fc.property(
          fc.array(chatMessageArb, { minLength: 0, maxLength: 50 }),
          (messages) => {
            // Save messages
            saveChatHistory(messages)
            
            // Load messages
            const loaded = loadChatHistory()
            
            // Should have same length (up to MAX_MESSAGES)
            const expectedLength = Math.min(messages.length, getMaxMessages())
            expect(loaded.length).toBe(expectedLength)
            
            // Compare content (last N messages)
            const expectedMessages = messages.slice(-getMaxMessages())
            for (let i = 0; i < loaded.length; i++) {
              expect(loaded[i].id).toBe(expectedMessages[i].id)
              expect(loaded[i].type).toBe(expectedMessages[i].type)
              expect(loaded[i].content).toBe(expectedMessages[i].content)
              // Timestamps should be equivalent (within 1 second due to serialization)
              expect(Math.abs(loaded[i].timestamp.getTime() - expectedMessages[i].timestamp.getTime())).toBeLessThan(1000)
            }
          }
        ),
        { numRuns: 50 }
      )
    })

    it('empty array round-trips correctly', () => {
      saveChatHistory([])
      const loaded = loadChatHistory()
      expect(loaded).toEqual([])
    })

    it('single message round-trips correctly', () => {
      fc.assert(
        fc.property(chatMessageArb, (message) => {
          saveChatHistory([message])
          const loaded = loadChatHistory()
          
          expect(loaded.length).toBe(1)
          expect(loaded[0].id).toBe(message.id)
          expect(loaded[0].type).toBe(message.type)
          expect(loaded[0].content).toBe(message.content)
        }),
        { numRuns: 20 }
      )
    })
  })

  /**
   * Property 13: Инвариант размера истории чата
   * 
   * History size never exceeds MAX_MESSAGES.
   * 
   * **Validates: Requirements 7.4**
   */
  describe('Property 13: Size invariant', () => {
    it('loaded history never exceeds MAX_MESSAGES', () => {
      fc.assert(
        fc.property(
          fc.array(chatMessageArb, { minLength: 0, maxLength: 200 }),
          (messages) => {
            saveChatHistory(messages)
            const loaded = loadChatHistory()
            
            expect(loaded.length).toBeLessThanOrEqual(getMaxMessages())
          }
        ),
        { numRuns: 50 }
      )
    })

    it('addMessageToHistory enforces limit', () => {
      fc.assert(
        fc.property(
          fc.array(chatMessageArb, { minLength: 0, maxLength: 150 }),
          chatMessageArb,
          (existingMessages, newMessage) => {
            const result = addMessageToHistory(existingMessages, newMessage)
            
            expect(result.length).toBeLessThanOrEqual(getMaxMessages())
            // New message should be at the end
            expect(result[result.length - 1].id).toBe(newMessage.id)
          }
        ),
        { numRuns: 50 }
      )
    })

    it('when at limit, oldest messages are removed', () => {
      const maxMessages = getMaxMessages()
      
      // Create exactly MAX_MESSAGES messages
      const messages: ChatMessage[] = Array.from({ length: maxMessages }, (_, i) => ({
        id: `msg-${i}`,
        type: 'user' as const,
        content: `Message ${i}`,
        timestamp: new Date()
      }))
      
      // Add one more
      const newMessage: ChatMessage = {
        id: 'new-msg',
        type: 'user',
        content: 'New message',
        timestamp: new Date()
      }
      
      const result = addMessageToHistory(messages, newMessage)
      
      // Should still be at MAX_MESSAGES
      expect(result.length).toBe(maxMessages)
      
      // First message should be msg-1 (msg-0 was removed)
      expect(result[0].id).toBe('msg-1')
      
      // Last message should be the new one
      expect(result[result.length - 1].id).toBe('new-msg')
    })
  })

  describe('clearChatHistory', () => {
    it('clears all messages', () => {
      fc.assert(
        fc.property(
          fc.array(chatMessageArb, { minLength: 1, maxLength: 50 }),
          (messages) => {
            saveChatHistory(messages)
            expect(loadChatHistory().length).toBeGreaterThan(0)
            
            clearChatHistory()
            expect(loadChatHistory()).toEqual([])
          }
        ),
        { numRuns: 20 }
      )
    })
  })

  describe('getMaxMessages', () => {
    it('returns 100', () => {
      expect(getMaxMessages()).toBe(100)
    })
  })
})
