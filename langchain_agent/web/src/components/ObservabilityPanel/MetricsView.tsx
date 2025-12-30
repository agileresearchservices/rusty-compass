/**
 * MetricsView - Performance metrics and timing visualization.
 */

import { useObservabilityStore } from '../../stores/observabilityStore'
import { Clock, Zap, FileSearch, Brain, CheckSquare, Layers, Radio, RotateCw } from 'lucide-react'
import clsx from 'clsx'

export function MetricsView() {
  const { metrics, steps, isExecuting, queryEvaluation, documentGradingSummary } = useObservabilityStore()

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

  // Extract additional metrics from events
  const getDocumentGradingBatchInfo = () => {
    if (!documentGradingSummary) return null
    const { total_count, relevant_count } = documentGradingSummary
    // Consider it batched if there are multiple documents
    const isBatched = total_count > 1
    return {
      isBatched,
      totalCount: total_count,
      relevantCount: relevant_count,
    }
  }

  const getReflectionIterationBreakdown = () => {
    // Count document retries (query_transformer iterations)
    const documentRetries = steps.filter(s => s.node === 'query_transformer').length
    // Count response retries from response_grading events
    let responseRetries = 0
    steps.forEach(step => {
      step.events.forEach(event => {
        if (event.type === 'response_grading' && 'retry_count' in event) {
          responseRetries = Math.max(responseRetries, event.retry_count)
        }
      })
    })
    return {
      documentRetries,
      responseRetries,
      totalRetries: documentRetries + responseRetries,
    }
  }

  const getCacheHitRate = () => {
    // Check if query_evaluator has search_strategy info
    if (!queryEvaluation) return null
    // If it executed, we assume it was a cache miss (not using cached results)
    // In a real scenario, this would be explicitly provided in the event
    return {
      hasInfo: true,
      strategy: queryEvaluation.search_strategy,
    }
  }

  const getStreamingIndicator = () => {
    // Check if LLM response events indicate streaming by looking for chunk events
    const llmChunks = steps.reduce((acc, step) => {
      if (step.node === 'agent') {
        const chunks = step.events.filter(e =>
          e.type === 'llm_response_chunk' || e.type === 'llm_reasoning_chunk'
        ).length
        acc += chunks
      }
      return acc
    }, 0)
    return {
      isStreaming: llmChunks > 1,
      chunkCount: llmChunks,
    }
  }

  const batchInfo = getDocumentGradingBatchInfo()
  const iterationBreakdown = getReflectionIterationBreakdown()
  const cacheMetrics = getCacheHitRate()
  const streamingMetrics = getStreamingIndicator()

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

      {/* Additional Metrics Section */}
      <div className="space-y-3">
        <h3 className="text-sm font-medium text-gray-400">Additional Metrics</h3>

        {/* Search Strategy / Cache Info */}
        {cacheMetrics && (
          <div className="obs-card bg-blue-500/10">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Radio className="w-4 h-4 text-blue-400" />
                <span className="text-sm text-gray-300">Search Strategy</span>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={clsx(
                  'px-2 py-1 rounded text-xs font-medium',
                  cacheMetrics.strategy === 'semantic-heavy'
                    ? 'bg-blue-500/30 text-blue-300'
                    : cacheMetrics.strategy === 'lexical-heavy'
                    ? 'bg-amber-500/30 text-amber-300'
                    : 'bg-purple-500/30 text-purple-300'
                )}
              >
                {cacheMetrics.strategy === 'semantic-heavy'
                  ? 'Semantic-Heavy'
                  : cacheMetrics.strategy === 'lexical-heavy'
                  ? 'Lexical-Heavy'
                  : 'Balanced'}
              </span>
              <span className="text-xs text-gray-500">
                {cacheMetrics.strategy === 'semantic-heavy'
                  ? 'Vector-based search emphasized'
                  : cacheMetrics.strategy === 'lexical-heavy'
                  ? 'Keyword search emphasized'
                  : 'Both methods equally weighted'}
              </span>
            </div>
          </div>
        )}

        {/* Streaming Indicator */}
        <div className={clsx('obs-card', streamingMetrics.isStreaming ? 'bg-cyan-500/10' : 'bg-gray-700/30')}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4" style={{ color: streamingMetrics.isStreaming ? '#06b6d4' : '#9ca3af' }} />
              <span className="text-sm text-gray-300">Response Delivery</span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={clsx(
                'px-2 py-1 rounded text-xs font-medium',
                streamingMetrics.isStreaming
                  ? 'bg-cyan-500/30 text-cyan-300'
                  : 'bg-gray-700 text-gray-400'
              )}
            >
              {streamingMetrics.isStreaming ? 'Streaming' : 'Non-Streaming'}
            </span>
            {streamingMetrics.isStreaming && (
              <span className="text-xs text-gray-500">
                {streamingMetrics.chunkCount} chunks
              </span>
            )}
          </div>
        </div>

        {/* Document Grading Batch Info */}
        {batchInfo && (
          <div className={clsx('obs-card', batchInfo.isBatched ? 'bg-emerald-500/10' : 'bg-gray-700/30')}>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4" style={{ color: batchInfo.isBatched ? '#10b981' : '#9ca3af' }} />
                <span className="text-sm text-gray-300">Document Grading</span>
              </div>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <span
                  className={clsx(
                    'px-2 py-1 rounded text-xs font-medium',
                    batchInfo.isBatched
                      ? 'bg-emerald-500/30 text-emerald-300'
                      : 'bg-gray-700 text-gray-400'
                  )}
                >
                  {batchInfo.isBatched ? 'Batched' : 'Single'}
                </span>
                <span className="text-xs text-gray-500">
                  {batchInfo.relevantCount}/{batchInfo.totalCount} relevant
                </span>
              </div>
              <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-600 to-emerald-400"
                  style={{
                    width: `${batchInfo.totalCount > 0 ? (batchInfo.relevantCount / batchInfo.totalCount) * 100 : 0}%`,
                  }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Reflection Iteration Breakdown */}
        {iterationBreakdown.totalRetries > 0 && (
          <div className="obs-card bg-amber-500/10">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <RotateCw className="w-4 h-4 text-amber-400" />
                <span className="text-sm text-gray-300">Reflection Iterations</span>
              </div>
              <span className="text-sm font-medium text-amber-300">
                {iterationBreakdown.totalRetries} total
              </span>
            </div>
            <div className="space-y-2">
              {iterationBreakdown.documentRetries > 0 && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Document Retries (query refinement)</span>
                  <span className="text-amber-300 font-medium">{iterationBreakdown.documentRetries}</span>
                </div>
              )}
              {iterationBreakdown.responseRetries > 0 && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-500">Response Retries (improvement)</span>
                  <span className="text-amber-300 font-medium">{iterationBreakdown.responseRetries}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* No additional metrics info */}
        {!cacheMetrics && iterationBreakdown.totalRetries === 0 && !batchInfo && (
          <div className="text-xs text-gray-500 text-center py-4">
            No additional metrics available
          </div>
        )}
      </div>
    </div>
  )
}
