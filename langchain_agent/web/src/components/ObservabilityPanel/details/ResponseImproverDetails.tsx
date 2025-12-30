/**
 * ResponseImproverDetails - Shows response improvement attempts with feedback.
 */

import { useObservabilityStore } from '../../../stores/observabilityStore'
import { RefreshCw, AlertCircle } from 'lucide-react'
import clsx from 'clsx'
import type { ObservabilityStep, ResponseImprovementEvent, ResponseGradingEvent } from '../../../types/events'

interface ResponseImproverDetailsProps {
  step: ObservabilityStep
}

export function ResponseImproverDetails({ step }: ResponseImproverDetailsProps) {
  const { steps } = useObservabilityStore()

  // Find all response_improvement events in this specific step (not global state)
  const improvementEvents = step.events.filter(
    (e) => e.type === 'response_improvement'
  ) as ResponseImprovementEvent[]

  // Find the response_grading event that TRIGGERED this improvement
  // Look for the most recent grading event that FAILED (grade !== 'pass')
  // This is the one that caused this response_improver to fire
  let previousGradingEvent: ResponseGradingEvent | null = null

  // Find the index of this response_improver step
  const currentStepIndex = steps.findIndex(s => s === step)

  // Look backwards from this step to find the grading event that triggered it
  for (let i = currentStepIndex - 1; i >= 0; i--) {
    const gradingEvent = steps[i].events.find(
      (e) => e.type === 'response_grading'
    ) as ResponseGradingEvent | undefined
    if (gradingEvent) {
      previousGradingEvent = gradingEvent
      break
    }
  }

  if (improvementEvents.length === 0) {
    return (
      <div className="text-sm text-gray-500">
        Waiting for response improvement...
      </div>
    )
  }

  const latestImprovement = improvementEvents[improvementEvents.length - 1]
  const { feedback, retry_count } = latestImprovement

  // Fallback: if feedback is empty, construct it from the previous grading event's reasoning
  const displayFeedback = feedback || (previousGradingEvent
    ? `[Response needs improvement] ${previousGradingEvent.reasoning}. Please provide a complete, well-structured response.`
    : 'Response did not meet quality criteria.')

  return (
    <div className="space-y-4">
      {/* Retry count indicator */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-500/20">
          <RefreshCw className={clsx('w-5 h-5 text-blue-400', step.status === 'running' && 'animate-spin')} />
          <span className="font-medium text-blue-400">
            Retry Attempt {retry_count}
          </span>
        </div>

        {improvementEvents.length > 1 && (
          <span className="text-xs text-gray-500">
            {improvementEvents.length} improvement{improvementEvents.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Feedback from response grader */}
      <div className="space-y-2">
        <span className="text-xs text-gray-500">Improvement Feedback:</span>
        <div className="bg-gray-800/50 rounded p-3 border border-blue-500/30">
          <p className="text-sm text-gray-300 leading-relaxed max-h-32 overflow-y-auto">
            {displayFeedback}
          </p>
        </div>
      </div>

      {/* Previous grading information if available */}
      {previousGradingEvent && (
        <div className="space-y-3 border-t border-gray-700 pt-3">
          <div className="space-y-2">
            <span className="text-xs text-gray-500">Previous Grade:</span>
            <div className="flex items-center gap-3">
              <div
                className={clsx(
                  'flex items-center gap-2 px-3 py-1.5 rounded-lg',
                  previousGradingEvent.grade === 'fail' ? 'bg-red-500/20' : 'bg-emerald-500/20'
                )}
              >
                <AlertCircle
                  className={clsx(
                    'w-4 h-4',
                    previousGradingEvent.grade === 'fail' ? 'text-red-400' : 'text-emerald-400'
                  )}
                />
                <span
                  className={clsx(
                    'font-medium text-sm',
                    previousGradingEvent.grade === 'fail' ? 'text-red-400' : 'text-emerald-400'
                  )}
                >
                  {previousGradingEvent.grade.toUpperCase()}
                </span>
              </div>

              {previousGradingEvent.retry_count > 0 && (
                <span className="text-xs text-amber-400">
                  Previous attempt #{previousGradingEvent.retry_count}
                </span>
              )}
            </div>
          </div>

          {/* Previous score */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-gray-500">Previous Score</span>
              <span className={previousGradingEvent.grade === 'fail' ? 'text-pink-400' : 'text-emerald-400'}>
                {(previousGradingEvent.score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="score-bar">
              <div
                className={clsx(
                  'score-bar-fill',
                  previousGradingEvent.grade === 'fail'
                    ? 'bg-gradient-to-r from-pink-600 to-pink-400'
                    : 'bg-gradient-to-r from-emerald-600 to-emerald-400'
                )}
                style={{ width: `${previousGradingEvent.score * 100}%` }}
              />
            </div>
          </div>

          {/* Previous reasoning */}
          <div className="space-y-1">
            <span className="text-xs text-gray-500">Previous Evaluation:</span>
            <p className="text-sm text-gray-300 bg-gray-800/50 rounded p-2 max-h-24 overflow-y-auto">
              {previousGradingEvent.reasoning}
            </p>
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className="text-xs text-gray-500 border-t border-gray-700 pt-3">
        <p>
          <strong>Response improvement</strong> uses feedback from the response grader to
          regenerate the answer when the initial response fails quality checks. The agent
          iterates on the response based on specific evaluation criteria.
        </p>
      </div>
    </div>
  )
}
