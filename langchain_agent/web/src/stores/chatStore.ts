/**
 * Zustand store for chat state management.
 * Handles messages, streaming state, and conversation metadata.
 */

import { create } from 'zustand'

// Message type for chat display
export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  isStreaming?: boolean
}

// Conversation summary for sidebar
export interface ConversationSummary {
  thread_id: string
  title: string
  created_at: string
  updated_at?: string
}

interface ChatState {
  // Current conversation
  threadId: string | null
  messages: ChatMessage[]
  isProcessing: boolean
  streamingContent: string

  // WebSocket state
  isConnected: boolean
  isConnecting: boolean
  connectionError: string | null

  // Conversation list
  conversations: ConversationSummary[]
  conversationsLoading: boolean

  // Actions
  setThreadId: (threadId: string) => void
  addMessage: (message: ChatMessage) => void
  updateLastMessage: (content: string) => void
  setIsProcessing: (isProcessing: boolean) => void
  setStreamingContent: (content: string) => void
  appendStreamingContent: (chunk: string) => void
  finalizeStreaming: () => void
  clearMessages: () => void
  setConversations: (conversations: ConversationSummary[]) => void
  setConversationsLoading: (loading: boolean) => void
  upsertConversation: (conversation: ConversationSummary) => void
  setMessages: (messages: ChatMessage[]) => void
  loadConversation: (threadId: string) => Promise<void>
  startNewConversation: () => void
  setConnectionState: (connected: boolean, connecting: boolean, error: string | null) => void
}

export const useChatStore = create<ChatState>((set, get) => ({
  // Initial state
  threadId: null,
  messages: [],
  isProcessing: false,
  streamingContent: '',
  isConnected: false,
  isConnecting: false,
  connectionError: null,
  conversations: [],
  conversationsLoading: false,

  // Actions
  setThreadId: (threadId) => set({ threadId }),

  addMessage: (message) => set((state) => ({
    messages: [...state.messages, message]
  })),

  updateLastMessage: (content) => set((state) => {
    const messages = [...state.messages]
    if (messages.length > 0) {
      const lastIndex = messages.length - 1
      messages[lastIndex] = {
        ...messages[lastIndex],
        content,
        isStreaming: false,
      }
    }
    return { messages }
  }),

  setIsProcessing: (isProcessing) => set({ isProcessing }),

  setStreamingContent: (content) => set({ streamingContent: content }),

  appendStreamingContent: (chunk) => set((state) => ({
    streamingContent: state.streamingContent + chunk
  })),

  finalizeStreaming: () => {
    const { streamingContent, messages } = get()
    if (streamingContent) {
      const lastMessage = messages[messages.length - 1]
      if (lastMessage && lastMessage.role === 'assistant' && lastMessage.isStreaming) {
        set((state) => {
          const updatedMessages = [...state.messages]
          const lastIndex = updatedMessages.length - 1
          updatedMessages[lastIndex] = {
            ...updatedMessages[lastIndex],
            content: streamingContent,
            isStreaming: false,
          }
          return {
            messages: updatedMessages,
            streamingContent: '',
            isProcessing: false,
          }
        })
      } else {
        set({
          streamingContent: '',
          isProcessing: false,
        })
      }
    } else {
      set({ isProcessing: false })
    }
  },

  clearMessages: () => set({ messages: [], streamingContent: '' }),

  setConversations: (conversations) => set({ conversations }),

  setConversationsLoading: (loading) => set({ conversationsLoading: loading }),

  upsertConversation: (conversation) => set((state) => {
    const existingIndex = state.conversations.findIndex(
      (c) => c.thread_id === conversation.thread_id
    )
    if (existingIndex >= 0) {
      // Update existing conversation and move to top
      const updated = [...state.conversations]
      updated.splice(existingIndex, 1)
      return { conversations: [conversation, ...updated] }
    } else {
      // Add new conversation at the top
      return { conversations: [conversation, ...state.conversations] }
    }
  }),

  setMessages: (messages) => set({ messages }),

  loadConversation: async (threadId) => {
    try {
      const response = await fetch(`/api/conversations/${threadId}`)
      if (!response.ok) {
        console.error('Failed to load conversation:', response.statusText)
        return
      }

      const data = await response.json()
      const messages: ChatMessage[] = data.messages.map((msg: { type: string; content: string }, index: number) => ({
        id: `msg-${threadId}-${index}`,
        role: msg.type === 'human' ? 'user' : 'assistant',
        content: msg.content,
        timestamp: new Date(data.created_at),
      }))

      set({
        threadId,
        messages,
        streamingContent: '',
        isProcessing: false,
      })
    } catch (error) {
      console.error('Error loading conversation:', error)
    }
  },

  startNewConversation: () => {
    const newThreadId = `conversation_${Math.random().toString(36).slice(2, 10)}`
    set({
      threadId: newThreadId,
      messages: [],
      streamingContent: '',
      isProcessing: false,
    })
  },

  setConnectionState: (connected, connecting, error) => set({
    isConnected: connected,
    isConnecting: connecting,
    connectionError: error,
  }),
}))
