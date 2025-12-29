/**
 * RawEventInspector - Debug view for raw JSON event data
 * Displays pretty-printed JSON with navigation and copy-to-clipboard functionality
 */

import { useState } from 'react'
import { Copy, ChevronLeft, ChevronRight } from 'lucide-react'
import type { AgentEvent } from '../../types/events'

interface RawEventInspectorProps {
  events: AgentEvent[]
}

export function RawEventInspector({ events }: RawEventInspectorProps) {
  const [currentEventIndex, setCurrentEventIndex] = useState(0)
  const [copyFeedback, setCopyFeedback] = useState(false)

  // Handle empty events
  if (!events || events.length === 0) {
    return (
      <div className="rounded border bg-gray-500/5 border-gray-500/20 px-4 py-3">
        <div className="text-xs text-gray-500">No events to display</div>
      </div>
    )
  }

  // Clamp index to valid range
  const validIndex = Math.min(Math.max(currentEventIndex, 0), events.length - 1)
  const currentEvent = events[validIndex]

  // Pretty print JSON
  const jsonString = JSON.stringify(currentEvent, null, 2)

  // Handle copy to clipboard
  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(jsonString)
    setCopyFeedback(true)
    setTimeout(() => setCopyFeedback(false), 2000)
  }

  // Handle navigation
  const handlePrevious = () => {
    if (validIndex > 0) {
      setCurrentEventIndex(validIndex - 1)
    }
  }

  const handleNext = () => {
    if (validIndex < events.length - 1) {
      setCurrentEventIndex(validIndex + 1)
    }
  }

  return (
    <div className="space-y-2">
      {/* Header with navigation and copy button */}
      <div className="flex items-center justify-between gap-2">
        {/* Event counter and navigation */}
        <div className="flex items-center gap-2">
          <button
            onClick={handlePrevious}
            disabled={validIndex === 0}
            className="p-1 rounded border border-gray-600/30 bg-gray-500/5 text-gray-400 hover:bg-gray-500/10 hover:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Previous event"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <span className="text-xs text-gray-400 px-2 py-1 bg-gray-500/5 border border-gray-600/30 rounded whitespace-nowrap">
            Event {validIndex + 1} of {events.length}
          </span>

          <button
            onClick={handleNext}
            disabled={validIndex === events.length - 1}
            className="p-1 rounded border border-gray-600/30 bg-gray-500/5 text-gray-400 hover:bg-gray-500/10 hover:text-gray-300 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Next event"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>

        {/* Copy button */}
        <button
          onClick={handleCopyToClipboard}
          className="p-1.5 rounded border border-gray-600/30 bg-gray-500/5 text-gray-400 hover:bg-gray-500/10 hover:text-gray-300 transition-colors flex items-center gap-1 text-xs"
          title="Copy JSON to clipboard"
        >
          <Copy className="w-4 h-4" />
          {copyFeedback ? (
            <span className="text-green-400">Copied!</span>
          ) : (
            <span>Copy</span>
          )}
        </button>
      </div>

      {/* Event type indicator */}
      <div className="text-xs text-gray-500">
        <span className="text-gray-600">Type:</span> {currentEvent.type}
      </div>

      {/* JSON code block */}
      <div className="rounded border border-gray-600/30 bg-black/40 overflow-hidden">
        <pre className="p-3 text-xs font-mono text-gray-300 overflow-x-auto max-h-96 overflow-y-auto whitespace-pre-wrap break-words">
          {jsonString}
        </pre>
      </div>
    </div>
  )
}
