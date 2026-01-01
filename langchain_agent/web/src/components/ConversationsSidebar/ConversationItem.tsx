/**
 * ConversationItem - Single conversation in the sidebar list.
 */

import { useCallback, useState } from 'react'
import { MessageSquare, Trash2 } from 'lucide-react'
import { useChatStore, type ConversationSummary } from '../../stores/chatStore'
import { useObservabilityStore } from '../../stores/observabilityStore'
import { ErrorNotification } from '../ErrorNotification'
import clsx from 'clsx'

interface ConversationItemProps {
  conversation: ConversationSummary
  isActive: boolean
  onSelect?: () => void
}

export function ConversationItem({ conversation, isActive, onSelect }: ConversationItemProps) {
  const { loadConversation, setConversations, conversations } = useChatStore()
  const { clearState } = useObservabilityStore()
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [isDeleting, setIsDeleting] = useState(false)

  const handleSelect = useCallback(async () => {
    if (isActive) return

    clearState()
    await loadConversation(conversation.thread_id)
    onSelect?.()
  }, [isActive, conversation.thread_id, loadConversation, clearState, onSelect])

  const handleDelete = useCallback(async (e: React.MouseEvent) => {
    e.stopPropagation()

    if (!confirm(`Delete "${conversation.title}"?`)) {
      return
    }

    setDeleteError(null)
    setIsDeleting(true)
    try {
      const response = await fetch(`http://localhost:8000/api/conversations/${conversation.thread_id}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        setConversations(conversations.filter((c) => c.thread_id !== conversation.thread_id))
      } else {
        setDeleteError('Failed to delete conversation. Please try again.')
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error)
      setDeleteError('Unable to delete conversation. Please check your connection.')
    } finally {
      setIsDeleting(false)
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
    <>
      {deleteError && (
        <div className="px-2 pb-2">
          <ErrorNotification
            message={deleteError}
            onDismiss={() => setDeleteError(null)}
            autoClose={false}
          />
        </div>
      )}
      <div
        className={clsx(
          'w-full flex items-start gap-2 px-3 py-2 rounded-lg text-left transition-colors group',
          isActive
            ? 'bg-gray-800 text-white'
            : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200'
        )}
      >
        <button
          onClick={handleSelect}
          disabled={isDeleting}
          aria-current={isActive ? 'page' : undefined}
          aria-label={`${conversation.title}${isActive ? ', current conversation' : ''}`}
          className="flex-1 flex items-start gap-2 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded disabled:opacity-50"
        >
          <MessageSquare className="w-4 h-4 mt-0.5 flex-shrink-0" aria-hidden="true" />

          <div className="flex-1 min-w-0 text-left">
            <div className="text-sm truncate">{conversation.title}</div>
            <div className="text-xs text-gray-400 mt-0.5">
              {formatDate(conversation.updated_at || conversation.created_at)}
            </div>
          </div>
        </button>

        <button
          onClick={handleDelete}
          disabled={isDeleting}
          aria-label={`Delete conversation "${conversation.title}"`}
          className={clsx(
            'p-1 rounded opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0',
            'text-gray-400 hover:text-red-400 hover:bg-red-500/10 focus:outline-none focus:ring-2 focus:ring-red-500 disabled:opacity-50'
          )}
        >
          <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
        </button>
      </div>
    </>
  )
}
