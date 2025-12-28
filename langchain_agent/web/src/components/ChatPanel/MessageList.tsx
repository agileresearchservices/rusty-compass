/**
 * MessageList - Displays chat messages with auto-scroll.
 */

import { useEffect, useRef } from 'react'
import { useChatStore } from '../../stores/chatStore'
import { Message } from './Message'

export function MessageList() {
  const { messages, streamingContent, isProcessing } = useChatStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])

  // Show streaming content in the last message if it's an assistant message
  const displayMessages = messages.map((msg, index) => {
    if (
      index === messages.length - 1 &&
      msg.role === 'assistant' &&
      msg.isStreaming &&
      streamingContent
    ) {
      return { ...msg, content: streamingContent }
    }
    return msg
  })

  if (messages.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 px-4">
        <div className="text-center max-w-md">
          <h3 className="text-lg font-medium text-gray-300 mb-2">
            Welcome to LangChain Agent
          </h3>
          <p className="text-sm">
            Ask questions about LangChain, LangGraph, or RAG concepts.
            Watch the observability panel to see how the agent processes your query.
          </p>
          <div className="mt-4 text-xs text-gray-600">
            Try: "What is LangGraph?" or "How does RAG work?"
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full overflow-y-auto px-4 py-4 space-y-4">
      {displayMessages.map((message) => (
        <Message key={message.id} message={message} />
      ))}

      {/* Show typing indicator when processing but no streaming content yet */}
      {isProcessing && !streamingContent && messages[messages.length - 1]?.role !== 'assistant' && (
        <div className="flex items-center gap-2 text-gray-500">
          <div className="flex gap-1">
            <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
            <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
            <span className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
          </div>
          <span className="text-sm">Agent is thinking...</span>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  )
}
