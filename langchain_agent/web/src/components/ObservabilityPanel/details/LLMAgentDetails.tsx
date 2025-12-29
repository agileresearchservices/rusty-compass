/**
 * LLMAgentDetails - Display LLM agent execution details including reasoning, tool calls, and responses
 */

import { useObservabilityStore } from '../../../stores/observabilityStore'
import type {
  LLMReasoningChunkEvent,
  LLMResponseChunkEvent,
  ToolCallEvent,
} from '../../../types/events'

export function LLMAgentDetails() {
  const { steps, currentNode } = useObservabilityStore()

  // Find the current agent step
  const agentStep = steps.find(
    (step) => step.node === 'agent' && (currentNode === 'agent' || step.status === 'complete')
  )

  if (!agentStep) {
    return (
      <div className="text-sm text-gray-500">
        No agent execution data available
      </div>
    )
  }

  // Extract different event types from the step's events
  const reasoningChunks = agentStep.events.filter(
    (e) => e.type === 'llm_reasoning_chunk'
  ) as LLMReasoningChunkEvent[]

  const responseChunks = agentStep.events.filter(
    (e) => e.type === 'llm_response_chunk'
  ) as LLMResponseChunkEvent[]

  const toolCalls = agentStep.events.filter(
    (e) => e.type === 'tool_call'
  ) as ToolCallEvent[]

  // Combine response chunks into full response
  const fullResponse = responseChunks.map((c) => c.content).join('')
  const fullReasoning = reasoningChunks.map((c) => c.content).join('')

  return (
    <div className="space-y-4 text-sm">
      {/* Reasoning Section */}
      {fullReasoning && (
        <div>
          <div className="text-xs font-medium text-gray-400 mb-2">LLM Reasoning</div>
          <div className="bg-gray-900/50 rounded border border-gray-700/30 p-3 text-xs text-gray-300 max-h-48 overflow-y-auto leading-relaxed whitespace-pre-wrap break-words">
            {fullReasoning}
          </div>
        </div>
      )}

      {/* Tool Calls Section */}
      {toolCalls.length > 0 && (
        <div>
          <div className="text-xs font-medium text-gray-400 mb-2">
            Tool Calls ({toolCalls.length})
          </div>
          <div className="space-y-2">
            {toolCalls.map((toolCall, idx) => (
              <div
                key={idx}
                className="bg-purple-500/5 border border-purple-500/20 rounded p-3"
              >
                <div className="text-xs font-medium text-purple-400 mb-2">
                  {toolCall.tool_name}
                </div>
                <div className="bg-black/30 rounded p-2 font-mono text-xs text-gray-300 max-h-24 overflow-y-auto break-words">
                  {JSON.stringify(toolCall.tool_args, null, 2)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Response Section */}
      {fullResponse && (
        <div>
          <div className="text-xs font-medium text-gray-400 mb-2">
            Response ({fullResponse.length} chars)
          </div>
          <div className="bg-gray-900/50 rounded border border-gray-700/30 p-3 text-xs text-gray-300 max-h-48 overflow-y-auto leading-relaxed whitespace-pre-wrap break-words">
            {fullResponse}
          </div>
        </div>
      )}

      {/* Status Indicator */}
      <div className="text-xs text-gray-500 pt-2 border-t border-gray-700/30">
        {responseChunks.length > 0 && (
          <>
            <div>
              Response complete:{' '}
              {responseChunks[responseChunks.length - 1]?.is_complete ? '✓' : 'Streaming...'}
            </div>
          </>
        )}
        {reasoningChunks.length > 0 && (
          <div>
            Reasoning complete:{' '}
            {reasoningChunks[reasoningChunks.length - 1]?.is_complete ? '✓' : 'Processing...'}
          </div>
        )}
      </div>
    </div>
  )
}
