/**
 * QueryTransformerDetails - Shows query transformation events and iterations.
 */

import type { ObservabilityStep, QueryTransformationEvent } from '../../../types/events'

interface QueryTransformerDetailsProps {
  step: ObservabilityStep
}

export function QueryTransformerDetails({ step }: QueryTransformerDetailsProps) {
  // Extract query transformation events from this specific step (not global state)
  const transformationEvents = step.events.filter(
    (e): e is QueryTransformationEvent => e.type === 'query_transformation'
  )

  if (transformationEvents.length === 0) {
    return (
      <div className="text-sm text-gray-500">
        Waiting for query transformation...
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {transformationEvents.map((event, index) => {
        const { original_query, transformed_query, iteration, max_iterations, reasons } = event

        return (
          <div key={index} className="space-y-4 pb-4 border-b border-gray-700 last:border-b-0 last:pb-0">
            {/* Iteration counter */}
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">Iteration:</span>
              <span className="px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-400 text-xs font-medium">
                {iteration}/{max_iterations}
              </span>
            </div>

            {/* Original query */}
            <div className="space-y-1">
              <span className="text-xs text-gray-500">Original Query:</span>
              <div className="max-h-24 overflow-y-auto bg-gray-900/50 rounded p-3 border border-gray-700/50">
                <code className="text-xs text-gray-300 whitespace-pre-wrap break-words">
                  {original_query}
                </code>
              </div>
            </div>

            {/* Transformed query */}
            <div className="space-y-1">
              <span className="text-xs text-gray-500">Transformed Query:</span>
              <div className="max-h-24 overflow-y-auto bg-violet-900/20 rounded p-3 border border-violet-700/30">
                <code className="text-xs text-violet-300 whitespace-pre-wrap break-words">
                  {transformed_query}
                </code>
              </div>
            </div>

            {/* Reasons */}
            {reasons && reasons.length > 0 && (
              <div className="space-y-2">
                <span className="text-xs text-gray-500">Transformation Reasons:</span>
                <ul className="space-y-1">
                  {reasons.map((reason, reasonIndex) => (
                    <li
                      key={reasonIndex}
                      className="flex gap-2 text-xs text-gray-300 bg-gray-800/50 rounded p-2"
                    >
                      <span className="text-gray-500 flex-shrink-0">•</span>
                      <span>{reason}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )
      })}

      {/* Explanation */}
      <div className="text-xs text-gray-500 border-t border-gray-700 pt-3">
        <p>
          <strong>Query transformation</strong> refines the search query based on
          document grading results. If retrieved documents are not relevant, the
          query is iteratively transformed to improve retrieval performance.
        </p>
      </div>
    </div>
  )
}
