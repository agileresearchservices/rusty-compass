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
  const { sendMessage } = useWebSocket()
  const { isProcessing, isConnected, connectionError } = useChatStore()

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
          <label htmlFor="message-input" className="sr-only">
            Chat message
          </label>
          <textarea
            id="message-input"
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
            aria-label="Chat message"
            aria-invalid={!isConnected ? 'true' : 'false'}
            aria-describedby={!isConnected ? 'connection-status' : undefined}
            className={clsx(
              'w-full resize-none rounded-lg border bg-gray-800 px-4 py-3 text-sm',
              'text-gray-100 placeholder-gray-400',
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
          aria-label="Send message"
          aria-disabled={!canSend}
          className={clsx(
            'flex-shrink-0 p-3 rounded-lg transition-colors',
            'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900',
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
        <div
          id="connection-status"
          className={clsx(
            'mt-2 text-xs font-medium',
            connectionError
              ? 'text-red-400'
              : 'text-yellow-500'
          )}
        >
          {connectionError ? (
            <div className="flex items-center gap-2">
              <span>⚠️ {connectionError}</span>
              <span className="text-xs text-gray-500">(Check that the server is running)</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <span className="inline-block w-1.5 h-1.5 bg-yellow-500 rounded-full animate-pulse" />
              Connecting to server...
            </div>
          )}
        </div>
      )}
    </div>
  )
}
