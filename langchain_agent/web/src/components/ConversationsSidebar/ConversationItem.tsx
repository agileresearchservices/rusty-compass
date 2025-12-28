/**
 * ConversationItem - Single conversation in the sidebar list.
 */

import { useCallback } from 'react'
import { MessageSquare, Trash2 } from 'lucide-react'
import { useChatStore, type ConversationSummary } from '../../stores/chatStore'
import { useObservabilityStore } from '../../stores/observabilityStore'
import clsx from 'clsx'

interface ConversationItemProps {
  conversation: ConversationSummary
  isActive: boolean
}

export function ConversationItem({ conversation, isActive }: ConversationItemProps) {
  const { setThreadId, clearMessages, setConversations, conversations } = useChatStore()
  const { clearState } = useObservabilityStore()

  const handleSelect = useCallback(() => {
    if (isActive) return

    setThreadId(conversation.thread_id)
    clearMessages()
    clearState()
    // TODO: Load conversation history from checkpoint
  }, [isActive, conversation.thread_id, setThreadId, clearMessages, clearState])

  const handleDelete = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()

    if (!confirm(`Delete "${conversation.title}"?`)) {
      return
    }

    try {
      const response = await fetch(`/api/conversations/${conversation.thread_id}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        setConversations(conversations.filter((c) => c.thread_id !== conversation.thread_id))
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }, [conversation, setConversations, conversations])

  // Format date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    const now = new Date()
    const diffDays = Math.floor((now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24))

    if (diffDays === 0) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else if (diffDays === 1) {
      return 'Yesterday'
    } else if (diffDays < 7) {
      return date.toLocaleDateString([], { weekday: 'short' })
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
    }
  }

  return (
    <div
      onClick={handleSelect}
      className={clsx(
        'w-full flex items-start gap-2 px-3 py-2 rounded-lg text-left transition-colors group cursor-pointer',
        isActive
          ? 'bg-gray-800 text-white'
          : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200'
      )}
    >
      <MessageSquare className="w-4 h-4 mt-0.5 flex-shrink-0" />

      <div className="flex-1 min-w-0">
        <div className="text-sm truncate">{conversation.title}</div>
        <div className="text-xs text-gray-500 mt-0.5">
          {formatDate(conversation.updated_at || conversation.created_at)}
        </div>
      </div>

      <button
        onClick={handleDelete}
        className={clsx(
          'p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity',
          'text-gray-500 hover:text-red-400 hover:bg-red-500/10'
        )}
        title="Delete conversation"
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}
