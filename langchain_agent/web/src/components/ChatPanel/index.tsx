/**
 * ChatPanel - Main chat interface container.
 * Displays message history and input form.
 */

import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'
import { useChatStore } from '../../stores/chatStore'
import { useObservabilityStore } from '../../stores/observabilityStore'

export function ChatPanel() {
  const { isProcessing } = useChatStore()
  const { isExecuting } = useObservabilityStore()

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <h1 className="text-lg font-semibold text-gray-100">Chat</h1>
          {(isProcessing || isExecuting) && (
            <span className="node-badge node-badge-running" aria-live="polite" aria-label="Processing response">
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-1.5" aria-hidden="true" />
              Processing
            </span>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-hidden">
        <MessageList />
      </div>

      {/* Input */}
      <div className="border-t border-gray-700">
        <MessageInput />
      </div>
    </div>
  )
}
