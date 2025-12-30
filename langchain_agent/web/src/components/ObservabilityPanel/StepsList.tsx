/**
 * StepsList - List of agent execution steps with expandable details.
 */

import { useRef, useState, useEffect } from 'react'
import { useObservabilityStore } from '../../stores/observabilityStore'
import { StepCard } from './StepCard'

export function StepsList() {
  const { steps } = useObservabilityStore()
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const [isAutoScrollEnabled] = useState(true)
  const [isNearBottom, setIsNearBottom] = useState(true)

  // Auto-scroll to bottom when new steps arrive
  useEffect(() => {
    if (isAutoScrollEnabled && isNearBottom && scrollContainerRef.current) {
      scrollContainerRef.current.scrollTo({
        top: scrollContainerRef.current.scrollHeight,
        behavior: 'smooth'
      })
    }
  }, [steps.length, isAutoScrollEnabled, isNearBottom])

  // Detect if user has scrolled away from bottom
  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const element = e.currentTarget
    const distanceFromBottom = element.scrollHeight - element.scrollTop - element.clientHeight
    setIsNearBottom(distanceFromBottom < 50) // Within 50px of bottom
  }

  if (steps.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 px-4">
        <div className="text-center max-w-sm">
          <p className="text-sm">
            Send a message to see the agent's execution steps in real-time.
          </p>
          <p className="text-xs mt-2 text-gray-600">
            Each step shows what the agent is doing and why.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="relative h-full">
      <div
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="h-full overflow-y-auto px-4 py-4 space-y-3"
      >
        {steps.map((step, index) => (
          <StepCard key={step.id} step={step} index={index} />
        ))}
      </div>

      {/* Scroll to bottom button */}
      {!isNearBottom && (
        <button
          onClick={() => {
            scrollContainerRef.current?.scrollTo({
              top: scrollContainerRef.current.scrollHeight,
              behavior: 'smooth'
            })
          }}
          className="absolute bottom-4 right-4 bg-blue-500 text-white rounded-full p-2 shadow-lg hover:bg-blue-600 transition-colors"
          aria-label="Scroll to latest"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </button>
      )}
    </div>
  )
}
