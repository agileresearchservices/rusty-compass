/**
 * StepCard - Expandable card showing details of a single execution step.
 */

import { ChevronDown, ChevronRight, Clock } from 'lucide-react'
import { useObservabilityStore } from '../../stores/observabilityStore'
import type { ObservabilityStep } from '../../types/events'
import { QueryEvaluatorDetails } from './details/QueryEvaluatorDetails'
import { SearchDetails } from './details/SearchDetails'
import { DocumentGraderDetails } from './details/DocumentGraderDetails'
import { ResponseGraderDetails } from './details/ResponseGraderDetails'
import { LLMAgentDetails } from './details/LLMAgentDetails'
import { QueryTransformerDetails } from './details/QueryTransformerDetails'
import { ResponseImproverDetails } from './details/ResponseImproverDetails'
import { EventCard } from './EventCard'
import clsx from 'clsx'

interface StepCardProps {
  step: ObservabilityStep
  index: number
}

// Node display configuration
const nodeConfig: Record<string, { label: string; color: string; bgColor: string }> = {
  query_evaluator: {
    label: 'Query Evaluator',
    color: 'text-blue-400',
    bgColor: 'bg-blue-500/10 border-blue-500/30',
  },
  agent: {
    label: 'LLM Agent',
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-500/10 border-cyan-500/30',
  },
  tools: {
    label: 'Knowledge Search',
    color: 'text-violet-400',
    bgColor: 'bg-violet-500/10 border-violet-500/30',
  },
  document_grader: {
    label: 'Document Grader',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-500/10 border-emerald-500/30',
  },
  query_transformer: {
    label: 'Query Transformer',
    color: 'text-amber-400',
    bgColor: 'bg-amber-500/10 border-amber-500/30',
  },
  response_grader: {
    label: 'Response Grader',
    color: 'text-pink-400',
    bgColor: 'bg-pink-500/10 border-pink-500/30',
  },
  response_improver: {
    label: 'Response Improver',
    color: 'text-orange-400',
    bgColor: 'bg-orange-500/10 border-orange-500/30',
  },
}

export function StepCard({ step, index }: StepCardProps) {
  const { expandedSteps, toggleStepExpanded } = useObservabilityStore()
  const isExpanded = expandedSteps.has(step.id)

  const config = nodeConfig[step.node] || {
    label: step.node,
    color: 'text-gray-400',
    bgColor: 'bg-gray-500/10 border-gray-500/30',
  }

  const statusColors = {
    idle: 'bg-gray-500',
    running: 'bg-blue-500 animate-pulse',
    complete: 'bg-emerald-500',
    error: 'bg-red-500',
  }

  return (
    <div
      className={clsx(
        'rounded-lg border transition-all',
        config.bgColor,
        step.status === 'running' && 'ring-2 ring-blue-500/50'
      )}
    >
      {/* Header - always visible */}
      <button
        onClick={() => toggleStepExpanded(step.id)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left"
      >
        {/* Expand icon */}
        <div className="flex-shrink-0 text-gray-500">
          {isExpanded ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
        </div>

        {/* Step number */}
        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-gray-700 flex items-center justify-center text-xs text-gray-300">
          {index + 1}
        </div>

        {/* Status indicator */}
        <div className={clsx('w-2 h-2 rounded-full', statusColors[step.status])} />

        {/* Node name */}
        <div className="flex-1 min-w-0">
          <span className={clsx('font-medium text-sm', config.color)}>
            {config.label}
          </span>
          {step.summary && (
            <span className="ml-2 text-xs text-gray-500 truncate">
              {step.summary}
            </span>
          )}
        </div>

        {/* Duration */}
        {step.durationMs !== undefined && (
          <div className="flex-shrink-0 flex items-center gap-1 text-xs text-gray-500">
            <Clock className="w-3 h-3" />
            {step.durationMs < 1000
              ? `${Math.round(step.durationMs)}ms`
              : `${(step.durationMs / 1000).toFixed(1)}s`}
          </div>
        )}
      </button>

      {/* Expanded content */}
      {isExpanded && (
        <div className="px-4 pb-4 pt-1 border-t border-gray-700/50">
          <StepDetails step={step} />
        </div>
      )}
    </div>
  )
}

function StepDetails({ step }: { step: ObservabilityStep }) {
  switch (step.node) {
    case 'query_evaluator':
      return <QueryEvaluatorDetails />

    case 'tools':
      return <SearchDetails />

    case 'document_grader':
      return <DocumentGraderDetails />

    case 'response_grader':
      return <ResponseGraderDetails />

    case 'agent':
      return <LLMAgentDetails />

    case 'query_transformer':
      return <QueryTransformerDetails />

    case 'response_improver':
      return <ResponseImproverDetails />

    default:
      return (
        <div className="text-sm text-gray-500">
          No details available for this step.
        </div>
      )
  }
}
