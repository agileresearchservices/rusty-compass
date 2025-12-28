/**
 * QueryEvaluatorDetails - Shows query analysis and lambda selection.
 */

import { useObservabilityStore } from '../../../stores/observabilityStore'
import clsx from 'clsx'

export function QueryEvaluatorDetails() {
  const { queryEvaluation } = useObservabilityStore()

  if (!queryEvaluation) {
    return (
      <div className="text-sm text-gray-500">
        Waiting for query evaluation...
      </div>
    )
  }

  const { lambda_mult, query_analysis, search_strategy } = queryEvaluation

  // Calculate position on the scale (0 = lexical, 1 = semantic)
  const scalePosition = lambda_mult * 100

  return (
    <div className="space-y-4">
      {/* Search strategy badge */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500">Strategy:</span>
        <span
          className={clsx(
            'px-2 py-0.5 rounded-full text-xs font-medium',
            search_strategy === 'lexical-heavy' && 'bg-amber-500/20 text-amber-400',
            search_strategy === 'balanced' && 'bg-violet-500/20 text-violet-400',
            search_strategy === 'semantic-heavy' && 'bg-blue-500/20 text-blue-400'
          )}
        >
          {search_strategy}
        </span>
      </div>

      {/* Lambda scale visualization */}
      <div className="space-y-2">
        <div className="flex justify-between text-xs text-gray-500">
          <span>Lexical (BM25)</span>
          <span>Semantic (Vector)</span>
        </div>

        <div className="relative h-3 bg-gray-700 rounded-full overflow-hidden">
          {/* Gradient background */}
          <div
            className="absolute inset-0"
            style={{
              background: 'linear-gradient(to right, #f59e0b, #8b5cf6, #3b82f6)',
              opacity: 0.3,
            }}
          />

          {/* Position indicator */}
          <div
            className="absolute top-0 bottom-0 w-3 -ml-1.5 bg-white rounded-full shadow-lg"
            style={{ left: `${scalePosition}%` }}
          />
        </div>

        <div className="text-center text-sm">
          <span className="text-white font-medium">
            {(lambda_mult * 100).toFixed(0)}%
          </span>
          <span className="text-gray-500 ml-1">semantic weight</span>
        </div>
      </div>

      {/* Analysis reasoning */}
      <div className="space-y-1">
        <span className="text-xs text-gray-500">Analysis:</span>
        <p className="text-sm text-gray-300 bg-gray-800/50 rounded p-2">
          {query_analysis}
        </p>
      </div>

      {/* Explanation */}
      <div className="text-xs text-gray-500 border-t border-gray-700 pt-3">
        <p>
          <strong>Lambda</strong> controls the hybrid search balance.
          Lower values favor exact keyword matching (BM25), while higher values
          favor semantic similarity (embeddings).
        </p>
      </div>
    </div>
  )
}
