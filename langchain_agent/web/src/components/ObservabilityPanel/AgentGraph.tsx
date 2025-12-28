/**
 * AgentGraph - Visual graph of agent workflow using React Flow.
 */

import { useCallback, useMemo } from 'react'
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  Node,
  Edge,
  MarkerType,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useObservabilityStore } from '../../stores/observabilityStore'
import type { NodeName } from '../../types/events'

// Node definitions
const nodeDefinitions: Record<
  NodeName,
  { label: string; color: string; x: number; y: number }
> = {
  query_evaluator: { label: 'Query Evaluator', color: '#3b82f6', x: 250, y: 0 },
  agent: { label: 'LLM Agent', color: '#06b6d4', x: 250, y: 100 },
  tools: { label: 'Knowledge Search', color: '#8b5cf6', x: 450, y: 100 },
  document_grader: { label: 'Doc Grader', color: '#10b981', x: 450, y: 200 },
  query_transformer: { label: 'Query Transform', color: '#f59e0b', x: 250, y: 200 },
  response_grader: { label: 'Response Grader', color: '#ec4899', x: 50, y: 100 },
  response_improver: { label: 'Response Improver', color: '#f97316', x: 50, y: 200 },
}

// Edge definitions (flow connections)
const edgeDefinitions: { source: NodeName; target: NodeName; label?: string }[] = [
  { source: 'query_evaluator', target: 'agent' },
  { source: 'agent', target: 'tools', label: 'tool call' },
  { source: 'agent', target: 'response_grader', label: 'response' },
  { source: 'tools', target: 'document_grader' },
  { source: 'document_grader', target: 'agent', label: 'pass' },
  { source: 'document_grader', target: 'query_transformer', label: 'fail' },
  { source: 'query_transformer', target: 'query_evaluator' },
  { source: 'response_grader', target: 'response_improver', label: 'fail' },
  { source: 'response_improver', target: 'agent' },
]

export function AgentGraph() {
  const { steps, currentNode, isExecuting } = useObservabilityStore()

  // Track which nodes have been visited
  const visitedNodes = useMemo(() => {
    const visited = new Set<string>()
    steps.forEach((step) => visited.add(step.node))
    return visited
  }, [steps])

  // Build nodes with status styling
  const nodes: Node[] = useMemo(() => {
    return Object.entries(nodeDefinitions).map(([id, def]) => {
      const isVisited = visitedNodes.has(id)
      const isCurrent = currentNode === id

      return {
        id,
        position: { x: def.x, y: def.y },
        data: { label: def.label },
        style: {
          background: isVisited ? def.color : '#374151',
          color: '#fff',
          border: isCurrent ? '3px solid #fff' : '2px solid #4b5563',
          borderRadius: '8px',
          padding: '10px 16px',
          fontSize: '12px',
          fontWeight: 500,
          opacity: isVisited || isCurrent ? 1 : 0.5,
          boxShadow: isCurrent ? `0 0 20px ${def.color}` : 'none',
          transition: 'all 0.3s ease',
        },
      }
    })
  }, [visitedNodes, currentNode])

  // Build edges with visited styling
  const edges: Edge[] = useMemo(() => {
    return edgeDefinitions.map((def, index) => {
      const sourceVisited = visitedNodes.has(def.source)
      const targetVisited = visitedNodes.has(def.target)
      const isActive = sourceVisited && targetVisited

      return {
        id: `e${index}`,
        source: def.source,
        target: def.target,
        label: def.label,
        type: 'smoothstep',
        animated: currentNode === def.source && isExecuting,
        style: {
          stroke: isActive ? '#60a5fa' : '#4b5563',
          strokeWidth: isActive ? 2 : 1,
          opacity: isActive ? 1 : 0.4,
        },
        labelStyle: {
          fill: '#9ca3af',
          fontSize: 10,
        },
        labelBgStyle: {
          fill: '#1f2937',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: isActive ? '#60a5fa' : '#4b5563',
        },
      }
    })
  }, [visitedNodes, currentNode, isExecuting])

  const onInit = useCallback(() => {
    // Graph initialized
  }, [])

  return (
    <div className="h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onInit={onInit}
        fitView
        attributionPosition="bottom-left"
        proOptions={{ hideAttribution: true }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        panOnDrag={true}
        zoomOnScroll={true}
        minZoom={0.5}
        maxZoom={1.5}
      >
        <Controls
          showInteractive={false}
          className="!bg-gray-800 !border-gray-700 !rounded-lg"
        />
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#374151"
        />
      </ReactFlow>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-gray-800/90 rounded-lg p-3 text-xs">
        <div className="flex items-center gap-2 mb-2">
          <div className="w-3 h-3 rounded bg-blue-500" />
          <span className="text-gray-400">Visited</span>
        </div>
        <div className="flex items-center gap-2 mb-2">
          <div className="w-3 h-3 rounded bg-gray-600 border-2 border-white" />
          <span className="text-gray-400">Current</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-gray-600 opacity-50" />
          <span className="text-gray-400">Pending</span>
        </div>
      </div>
    </div>
  )
}
