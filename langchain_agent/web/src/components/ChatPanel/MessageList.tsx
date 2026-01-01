/**
 * MessageList - Displays chat messages with auto-scroll.
 */

import { useEffect, useRef } from 'react'
import { useChatStore } from '../../stores/chatStore'
import { useObservabilityStore } from '../../stores/observabilityStore'
import { Message } from './Message'

export function MessageList() {
  const { messages, streamingContent, isProcessing } = useChatStore()
  const { currentNode, steps } = useObservabilityStore()
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Map node IDs to user-friendly display names
  const getNodeDisplayName = (node: string): string => {
    const names: Record<string, string> = {
      query_evaluator: 'Evaluating query',
      agent: 'Planning response',
      tools: 'Searching documents',
      document_grader: 'Grading documents',
      response_grader: 'Checking response quality',
      query_transformer: 'Refining search',
      response_improver: 'Improving response',
    }
    return names[node] || 'Processing'
  }

  // Get a brief summary of the current step
  const getCurrentStepSummary = (): string | null => {
    if (!currentNode || !steps.length) return null
    const currentStep = steps.find(s => s.node === currentNode)
    if (!currentStep || !currentStep.events.length) return null

    // Get the most recent event for this step
    const latestEvent = currentStep.events[currentStep.events.length - 1]

    // Extract summary based on event type
    if (latestEvent.type === 'hybrid_search_result') {
      return `Found ${latestEvent.candidate_count} candidates`
    }
    if (latestEvent.type === 'document_grading_summary') {
      return `${latestEvent.relevant_count}/${latestEvent.total_count} relevant`
    }
    if (latestEvent.type === 'response_grading') {
      return `Score: ${(latestEvent.score * 100).toFixed(0)}%`
    }

    return null
  }

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
    <div className="h-full overflow-y-auto px-4 py-4 space-y-4" role="log" aria-live="polite" aria-label="Chat messages">
      {displayMessages.map((message) => (
        <Message key={message.id} message={message} />
      ))}

      {/* Show typing indicator when processing but no streaming content yet */}
      {isProcessing && !streamingContent && messages[messages.length - 1]?.role !== 'assistant' && (
        <div className="flex items-center gap-2 text-gray-500" aria-live="polite" aria-label="Agent processing">
          <div className="flex gap-1">
            <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" aria-hidden="true" />
          </div>
          <div className="text-sm">
            {currentNode ? (
              <span className="flex items-center gap-2">
                <span className="font-medium text-blue-400">
                  {getNodeDisplayName(currentNode)}
                </span>
                {getCurrentStepSummary() && (
                  <span className="text-gray-400">• {getCurrentStepSummary()}</span>
                )}
              </span>
            ) : (
              <span>Agent is thinking...</span>
            )}
          </div>
        </div>
      )}

      <div ref={messagesEndRef} />
    </div>
  )
}
