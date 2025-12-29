/**
 * DocumentGraderDetails - Shows document grading results.
 */

import { CheckCircle, XCircle } from 'lucide-react'
import clsx from 'clsx'
import type { ObservabilityStep, DocumentGradingSummaryEvent, DocumentGradeEvent } from '../../../types/events'

interface DocumentGraderDetailsProps {
  step: ObservabilityStep
}

export function DocumentGraderDetails({ step }: DocumentGraderDetailsProps) {
  // Extract document grades from the specific step's events (not global state)
  const gradeEvents = step.events.filter((e) => e.type === 'document_grade') as DocumentGradeEvent[]
  const summaryEvent = step.events.find((e) => e.type === 'document_grading_summary') as DocumentGradingSummaryEvent | undefined

  if (!summaryEvent) {
    return (
      <div className="text-sm text-gray-500">
        Waiting for document grading...
      </div>
    )
  }

  const { grade, relevant_count, total_count, average_score, reasoning } = summaryEvent
  const passed = grade === 'pass'

  return (
    <div className="space-y-4">
      {/* Overall grade badge */}
      <div className="flex items-center gap-3">
        <div
          className={clsx(
            'flex items-center gap-2 px-3 py-1.5 rounded-lg',
            passed ? 'bg-emerald-500/20' : 'bg-red-500/20'
          )}
        >
          {passed ? (
            <CheckCircle className="w-5 h-5 text-emerald-400" />
          ) : (
            <XCircle className="w-5 h-5 text-red-400" />
          )}
          <span
            className={clsx(
              'font-medium',
              passed ? 'text-emerald-400' : 'text-red-400'
            )}
          >
            {passed ? 'PASS' : 'FAIL'}
          </span>
        </div>

        <span className="text-sm text-gray-400">
          {relevant_count} of {total_count} documents relevant
        </span>
      </div>

      {/* Average score */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Average Relevance</span>
          <span className="text-emerald-400">{(average_score * 100).toFixed(1)}%</span>
        </div>
        <div className="score-bar">
          <div
            className={clsx(
              'score-bar-fill',
              passed
                ? 'bg-gradient-to-r from-emerald-600 to-emerald-400'
                : 'bg-gradient-to-r from-red-600 to-red-400'
            )}
            style={{ width: `${average_score * 100}%` }}
          />
        </div>
      </div>

      {/* Individual document grades */}
      {gradeEvents.length > 0 && (
        <div className="space-y-2">
          <span className="text-xs text-gray-500">Document Grades:</span>
          <div className="space-y-1">
            {gradeEvents.map((event, index) => (
              <div
                key={index}
                className="flex items-center gap-2 text-xs bg-gray-800/50 rounded p-2"
              >
                {event.relevant ? (
                  <CheckCircle className="w-3 h-3 text-emerald-400 flex-shrink-0" />
                ) : (
                  <XCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
                )}
                <span className="text-gray-300 truncate flex-1">
                  {event.source}
                </span>
                <span className="text-gray-500">
                  {(event.score * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Reasoning */}
      <div className="space-y-1">
        <span className="text-xs text-gray-500">Summary:</span>
        <p className="text-sm text-gray-300 bg-gray-800/50 rounded p-2">
          {reasoning}
        </p>
      </div>

      {/* Explanation */}
      <div className="text-xs text-gray-500 border-t border-gray-700 pt-3">
        <p>
          <strong>Document grading</strong> uses the LLM to evaluate if each
          retrieved document is relevant to the user's query. If too few
          documents are relevant, the query may be transformed and retrieval retried.
        </p>
      </div>
    </div>
  )
}
