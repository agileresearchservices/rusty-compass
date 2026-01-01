/**
 * ConversationsSidebar - List of past conversations with management.
 */

import { useEffect, useCallback, useState } from 'react'
import { Plus, Trash2, MessageSquare, RefreshCw } from 'lucide-react'
import { useChatStore, type ConversationSummary } from '../../stores/chatStore'
import { useObservabilityStore } from '../../stores/observabilityStore'
import { ConversationItem } from './ConversationItem'
import { ErrorNotification } from '../ErrorNotification'
import { SkeletonConversationItem } from '../SkeletonLoader'

interface ConversationsSidebarProps {
  onConversationSelect?: () => void
}

export function ConversationsSidebar({ onConversationSelect }: ConversationsSidebarProps) {
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
  const [error, setError] = useState<string | null>(null)

  // Fetch conversations on mount
  useEffect(() => {
    fetchConversations()
  }, [])

  const fetchConversations = useCallback(async () => {
    setConversationsLoading(true)
    setError(null)
    try {
      const response = await fetch('http://localhost:8000/api/conversations?limit=20')
      if (response.ok) {
        const data: ConversationSummary[] = await response.json()
        setConversations(data)
      } else {
        setError('Failed to load conversations. Please try again.')
      }
    } catch (error) {
      console.error('Failed to fetch conversations:', error)
      setError('Unable to connect to server. Please check your connection and try again.')
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

    setError(null)
    try {
      const response = await fetch('http://localhost:8000/api/conversations', {
        method: 'DELETE',
      })
      if (response.ok) {
        setConversations([])
        handleNewConversation()
      } else {
        setError('Failed to delete conversations. Please try again.')
      }
    } catch (error) {
      console.error('Failed to clear conversations:', error)
      setError('Unable to delete conversations. Please check your connection and try again.')
    }
  }, [setConversations, handleNewConversation])

  return (
    <div className="flex flex-col h-full bg-gray-950 border-r border-gray-800">
      {/* Error notification */}
      {error && (
        <div className="px-4 pt-4">
          <ErrorNotification message={error} onDismiss={() => setError(null)} />
        </div>
      )}

      {/* Header */}
      <div className="px-4 py-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-gray-300">Conversations</h2>
          <button
            onClick={fetchConversations}
            aria-label="Refresh conversations"
            className="p-1.5 text-gray-400 hover:text-gray-200 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 rounded"
            title="Refresh conversations"
          >
            <RefreshCw className={`w-4 h-4 ${conversationsLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        <button
          onClick={handleNewConversation}
          className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm text-white transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-950"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto py-2">
        {conversationsLoading ? (
          <SkeletonConversationItem count={5} />
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
                onSelect={onConversationSelect}
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
            className="w-full flex items-center justify-center gap-2 px-3 py-2 bg-gray-800 hover:bg-red-900/50 rounded-lg text-sm text-gray-300 hover:text-red-400 transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-950"
          >
            <Trash2 className="w-4 h-4" />
            Clear All
          </button>
        </div>
      )}
    </div>
  )
}
