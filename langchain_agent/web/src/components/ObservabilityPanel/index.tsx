/**
 * ObservabilityPanel - Real-time visualization of agent execution.
 * Shows steps, graph, and metrics with full observability.
 */

import { LayoutList, GitBranch, BarChart3 } from 'lucide-react'
import { useObservabilityStore } from '../../stores/observabilityStore'
import { StepsList } from './StepsList'
import { AgentGraph } from './AgentGraph'
import { MetricsView } from './MetricsView'
import clsx from 'clsx'

const tabs = [
  { id: 'steps' as const, label: 'Steps', icon: LayoutList },
  { id: 'graph' as const, label: 'Graph', icon: GitBranch },
  { id: 'metrics' as const, label: 'Metrics', icon: BarChart3 },
]

export function ObservabilityPanel() {
  const { activeTab, setActiveTab, isExecuting, steps } = useObservabilityStore()

  return (
    <div className="flex flex-col h-full bg-gray-900/50">
      {/* Header with tabs */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700">
        <div className="flex items-center gap-2">
          <h2 className="text-lg font-semibold text-gray-100">Observability</h2>
          {isExecuting && (
            <span className="node-badge node-badge-running">
              <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse mr-1.5" />
              Active
            </span>
          )}
        </div>

        {/* Tab buttons */}
        <div className="flex gap-1 bg-gray-800 rounded-lg p-1" role="tablist" aria-label="Observability panel tabs">
          {tabs.map((tab) => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                role="tab"
                aria-selected={isActive}
                aria-controls={`${tab.id}-panel`}
                className={clsx(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500',
                  isActive
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:text-gray-200'
                )}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-hidden relative">
        {activeTab === 'steps' && (
          <div id="steps-panel" role="tabpanel" className="h-full overflow-hidden">
            <StepsList />
          </div>
        )}
        {activeTab === 'graph' && (
          <div id="graph-panel" role="tabpanel" className="absolute inset-0">
            <AgentGraph />
          </div>
        )}
        {activeTab === 'metrics' && (
          <div id="metrics-panel" role="tabpanel" className="h-full overflow-hidden">
            <MetricsView />
          </div>
        )}
      </div>

      {/* Footer with step count */}
      <div className="px-4 py-2 border-t border-gray-700 text-xs text-gray-400">
        {steps.length} step{steps.length !== 1 ? 's' : ''} recorded
      </div>
    </div>
  )
}
