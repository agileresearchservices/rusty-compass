/**
 * MetricsView - Performance metrics and timing visualization.
 */

import { useObservabilityStore } from '../../stores/observabilityStore'
import { Clock, Zap, FileSearch, Brain, CheckSquare } from 'lucide-react'
import clsx from 'clsx'

export function MetricsView() {
  const { metrics, steps, isExecuting } = useObservabilityStore()

  if (!metrics && steps.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-gray-500 px-4">
        <div className="text-center max-w-sm">
          <p className="text-sm">
            Send a message to see performance metrics.
          </p>
        </div>
      </div>
    )
  }

  // Calculate metrics from steps if not yet received
  const stepMetrics = steps.reduce(
    (acc, step) => {
      if (step.durationMs) {
        acc[step.node] = (acc[step.node] || 0) + step.durationMs
        acc.total += step.durationMs
      }
      return acc
    },
    { total: 0 } as Record<string, number>
  )

  const displayMetrics = metrics || {
    query_evaluation_ms: stepMetrics['query_evaluator'],
    retrieval_ms: stepMetrics['tools'],
    document_grading_ms: stepMetrics['document_grader'],
    llm_generation_ms: stepMetrics['agent'],
    response_grading_ms: stepMetrics['response_grader'],
    total_ms: stepMetrics.total,
  }

  const metricItems = [
    {
      label: 'Query Evaluation',
      value: displayMetrics.query_evaluation_ms,
      icon: Zap,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
    },
    {
      label: 'Retrieval',
      value: displayMetrics.retrieval_ms,
      icon: FileSearch,
      color: 'text-violet-400',
      bgColor: 'bg-violet-500/10',
    },
    {
      label: 'Document Grading',
      value: displayMetrics.document_grading_ms,
      icon: CheckSquare,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-500/10',
    },
    {
      label: 'LLM Generation',
      value: displayMetrics.llm_generation_ms,
      icon: Brain,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/10',
    },
    {
      label: 'Response Grading',
      value: displayMetrics.response_grading_ms,
      icon: CheckSquare,
      color: 'text-pink-400',
      bgColor: 'bg-pink-500/10',
    },
  ].filter((item) => item.value !== undefined)

  const totalMs = displayMetrics.total_ms || 0

  return (
    <div className="h-full overflow-y-auto px-4 py-4 space-y-6">
      {/* Total time */}
      <div className="obs-card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-gray-400" />
            <span className="font-medium text-gray-200">Total Duration</span>
          </div>
          {isExecuting && (
            <span className="node-badge node-badge-running">Running</span>
          )}
        </div>

        <div className="text-3xl font-bold text-white mb-2">
          {totalMs < 1000
            ? `${Math.round(totalMs)}ms`
            : `${(totalMs / 1000).toFixed(2)}s`}
        </div>

        {/* Timeline bar */}
        {metricItems.length > 0 && (
          <div className="h-4 bg-gray-700 rounded-full overflow-hidden flex">
            {metricItems.map((item, index) => {
              const width = totalMs > 0 ? ((item.value || 0) / totalMs) * 100 : 0
              return (
                <div
                  key={index}
                  className={clsx('h-full transition-all', item.bgColor)}
                  style={{
                    width: `${width}%`,
                    backgroundColor: item.color.replace('text-', '').replace('-400', ''),
                  }}
                  title={`${item.label}: ${Math.round(item.value || 0)}ms`}
                />
              )
            })}
          </div>
        )}
      </div>

      {/* Individual metrics */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-gray-400">Breakdown</h3>

        {metricItems.map((item, index) => {
          const Icon = item.icon
          const percentage = totalMs > 0 ? ((item.value || 0) / totalMs) * 100 : 0

          return (
            <div key={index} className={clsx('obs-card', item.bgColor)}>
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Icon className={clsx('w-4 h-4', item.color)} />
                  <span className="text-sm text-gray-300">{item.label}</span>
                </div>
                <div className="text-right">
                  <span className={clsx('font-medium', item.color)}>
                    {(item.value || 0) < 1000
                      ? `${Math.round(item.value || 0)}ms`
                      : `${((item.value || 0) / 1000).toFixed(2)}s`}
                  </span>
                  <span className="text-xs text-gray-500 ml-2">
                    ({percentage.toFixed(1)}%)
                  </span>
                </div>
              </div>

              <div className="score-bar">
                <div
                  className={clsx('score-bar-fill', item.bgColor)}
                  style={{
                    width: `${percentage}%`,
                    backgroundColor: item.color.includes('blue')
                      ? '#3b82f6'
                      : item.color.includes('violet')
                      ? '#8b5cf6'
                      : item.color.includes('emerald')
                      ? '#10b981'
                      : item.color.includes('cyan')
                      ? '#06b6d4'
                      : '#ec4899',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* Step count */}
      <div className="obs-card">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Steps Executed</span>
          <span className="text-lg font-medium text-white">{steps.length}</span>
        </div>
      </div>
    </div>
  )
}
