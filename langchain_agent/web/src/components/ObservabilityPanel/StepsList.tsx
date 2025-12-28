/**
 * StepsList - List of agent execution steps with expandable details.
 */

import { useObservabilityStore } from '../../stores/observabilityStore'
import { StepCard } from './StepCard'

export function StepsList() {
  const { steps } = useObservabilityStore()

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
    <div className="h-full overflow-y-auto px-4 py-4 space-y-3">
      {steps.map((step, index) => (
        <StepCard key={step.id} step={step} index={index} />
      ))}
    </div>
  )
}
