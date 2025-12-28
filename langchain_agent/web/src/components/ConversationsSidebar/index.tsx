/**
 * ConversationsSidebar - List of past conversations with management.
 */

import { useEffect, useCallback } from 'react'
import { Plus, Trash2, MessageSquare, RefreshCw } from 'lucide-react'
import { useChatStore, type ConversationSummary } from '../../stores/chatStore'
import { useObservabilityStore } from '../../stores/observabilityStore'
import { ConversationItem } from './ConversationItem'

export function ConversationsSidebar() {
  const {
    conversations,
    conversationsLoading,
    threadId,
    setConversations,
    setConversationsLoading,
    startNewConversation,
    clearMessages,
  } = useChatStore()

  const { clearState } = useObservabilityStore()

  // Fetch conversations on mount
  useEffect(() => {
    fetchConversations()
  }, [])

  const fetchConversations = useCallback(async () => {
    setConversationsLoading(true)
    try {
      const response = await fetch('/api/conversations?limit=20')
      if (response.ok) {
        const data: ConversationSummary[] = await response.json()
        setConversations(data)
      }
    } catch (error) {
      console.error('Failed to fetch conversations:', error)
    } finally {
      setConversationsLoading(false)
    }
  }, [setConversations, setConversationsLoading])

  const handleNewConversation = useCallback(() => {
    startNewConversation()
    clearMessages()
    clearState()
  }, [startNewConversation, clearMessages, clearState])

  const handleClearAll = useCallback(async () => {
    if (!confirm('Are you sure you want to delete all conversations?')) {
      return
    }

    try {
      const response = await fetch('/api/conversations', {
        method: 'DELETE',
      })
      if (response.ok) {
        setConversations([])
        handleNewConversation()
      }
    } catch (error) {
      console.error('Failed to clear conversations:', error)
    }
  }, [setConversations, handleNewConversation])

  return (
    <div className="flex flex-col h-full bg-gray-950 border-r border-gray-800">
      {/* Header */}
      <div className="px-4 py-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-300">Conversations</h2>
          <button
            onClick={fetchConversations}
            className="p-1.5 text-gray-500 hover:text-gray-300 transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${conversationsLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        <button
          onClick={handleNewConversation}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm text-white transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2">
        {conversationsLoading ? (
          <div className="flex items-center justify-center py-8 text-gray-500">
            <RefreshCw className="w-5 h-5 animate-spin" />
          </div>
        ) : conversations.length === 0 ? (
          <div className="px-4 py-8 text-center text-gray-500 text-sm">
            <MessageSquare className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p>No conversations yet</p>
            <p className="text-xs mt-1">Start chatting to create one</p>
          </div>
        ) : (
          <div className="space-y-1 px-2">
            {conversations.map((conversation) => (
              <ConversationItem
                key={conversation.thread_id}
                conversation={conversation}
                isActive={conversation.thread_id === threadId}
              />
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      {conversations.length > 0 && (
        <div className="px-4 py-3 border-t border-gray-800">
          <button
            onClick={handleClearAll}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-gray-800 hover:bg-red-900/50 rounded-lg text-sm text-gray-400 hover:text-red-400 transition-colors"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </button>
        </div>
      )}
    </div>
  )
}
