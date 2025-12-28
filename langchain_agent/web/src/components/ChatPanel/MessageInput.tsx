/**
 * MessageInput - Chat input form with send button.
 */

import { useState, useCallback, KeyboardEvent } from 'react'
import { Send } from 'lucide-react'
import { useWebSocket } from '../../hooks/useWebSocket'
import { useChatStore } from '../../stores/chatStore'
import clsx from 'clsx'

export function MessageInput() {
  const [message, setMessage] = useState('')
  const { sendMessage, isConnected } = useWebSocket()
  const { isProcessing } = useChatStore()

  const handleSubmit = useCallback(() => {
    const trimmed = message.trim()
    if (!trimmed || isProcessing || !isConnected) return

    sendMessage(trimmed)
    setMessage('')
  }, [message, sendMessage, isProcessing, isConnected])

  const handleKeyDown = useCallback(
    (e: KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSubmit()
      }
    },
    [handleSubmit]
  )

  const canSend = message.trim() && !isProcessing && isConnected

  return (
    <div className="p-4">
      <div className="flex items-end gap-2">
        <div className="flex-1 relative">
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isConnected
                ? "Ask about LangChain, LangGraph, or RAG..."
                : "Connecting..."
            }
            disabled={!isConnected || isProcessing}
            rows={1}
            className={clsx(
              'w-full resize-none rounded-lg border bg-gray-800 px-4 py-3 text-sm',
              'text-gray-100 placeholder-gray-500',
              'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              isConnected ? 'border-gray-700' : 'border-yellow-600'
            )}
            style={{
              minHeight: '44px',
              maxHeight: '200px',
            }}
          />
        </div>

        <button
          onClick={handleSubmit}
          disabled={!canSend}
          className={clsx(
            'flex-shrink-0 p-3 rounded-lg transition-colors',
            'focus:outline-none focus:ring-2 focus:ring-blue-500',
            canSend
              ? 'bg-blue-600 hover:bg-blue-700 text-white'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'
          )}
        >
          <Send className="w-5 h-5" />
        </button>
      </div>

      {/* Connection status */}
      {!isConnected && (
        <div className="mt-2 text-xs text-yellow-500">
          Connecting to server...
        </div>
      )}
    </div>
  )
}
