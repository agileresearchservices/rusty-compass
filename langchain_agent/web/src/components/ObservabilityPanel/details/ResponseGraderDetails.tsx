/**
 * ResponseGraderDetails - Shows response quality grading.
 */

import { useObservabilityStore } from '../../../stores/observabilityStore'
import { CheckCircle, XCircle, RefreshCw } from 'lucide-react'
import clsx from 'clsx'

export function ResponseGraderDetails() {
  const { responseGrading } = useObservabilityStore()

  if (!responseGrading) {
    return (
      <div className="text-sm text-gray-500">
        Waiting for response grading...
      </div>
    )
  }

  const { grade, score, reasoning, retry_count, max_retries } = responseGrading
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

        {/* Retry info */}
        {retry_count > 0 && (
          <div className="flex items-center gap-1.5 text-xs text-amber-400">
            <RefreshCw className="w-3 h-3" />
            Retry {retry_count}/{max_retries}
          </div>
        )}
      </div>

      {/* Quality score */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs">
          <span className="text-gray-500">Quality Score</span>
          <span className={passed ? 'text-emerald-400' : 'text-pink-400'}>
            {(score * 100).toFixed(1)}%
          </span>
        </div>
        <div className="score-bar">
          <div
            className={clsx(
              'score-bar-fill',
              passed
                ? 'bg-gradient-to-r from-emerald-600 to-emerald-400'
                : 'bg-gradient-to-r from-pink-600 to-pink-400'
            )}
            style={{ width: `${score * 100}%` }}
          />
        </div>
      </div>

      {/* Criteria */}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <div className="bg-gray-800/50 rounded p-2 text-center">
          <div className="text-gray-500 mb-1">Relevance</div>
          <CheckCircle
            className={clsx(
              'w-4 h-4 mx-auto',
              passed ? 'text-emerald-400' : 'text-gray-500'
            )}
          />
        </div>
        <div className="bg-gray-800/50 rounded p-2 text-center">
          <div className="text-gray-500 mb-1">Completeness</div>
          <CheckCircle
            className={clsx(
              'w-4 h-4 mx-auto',
              passed ? 'text-emerald-400' : 'text-gray-500'
            )}
          />
        </div>
        <div className="bg-gray-800/50 rounded p-2 text-center">
          <div className="text-gray-500 mb-1">Clarity</div>
          <CheckCircle
            className={clsx(
              'w-4 h-4 mx-auto',
              passed ? 'text-emerald-400' : 'text-gray-500'
            )}
          />
        </div>
      </div>

      {/* Reasoning */}
      <div className="space-y-1">
        <span className="text-xs text-gray-500">Evaluation:</span>
        <p className="text-sm text-gray-300 bg-gray-800/50 rounded p-2">
          {reasoning}
        </p>
      </div>

      {/* Explanation */}
      <div className="text-xs text-gray-500 border-t border-gray-700 pt-3">
        <p>
          <strong>Response grading</strong> evaluates the final answer for
          relevance, completeness, and clarity. If the response fails, the agent
          can retry with feedback to improve.
        </p>
      </div>
    </div>
  )
}
