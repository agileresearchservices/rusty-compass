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

// Node definitions - organized in vertical hierarchy with separate retry section
const nodeDefinitions: Record<
  NodeName,
  { label: string; color: string; x: number; y: number; isRetry?: boolean }
> = {
  // === MAIN FLOW (top to bottom) ===
  query_evaluator: { label: 'Query Evaluator', color: '#3b82f6', x: 250, y: 0 },       // Entry point
  agent: { label: 'LLM Agent', color: '#06b6d4', x: 250, y: 100 },                     // Central processor
  response_grader: { label: 'Response Grader', color: '#ec4899', x: 80, y: 220 },     // Left branch
  tools: { label: 'Knowledge Search', color: '#8b5cf6', x: 420, y: 220 },              // Right branch
  document_grader: { label: 'Doc Grader', color: '#10b981', x: 420, y: 340 },          // Below tools
  // === RETRY SECTION (below separator) ===
  response_improver: { label: 'Response Improver', color: '#6b7280', x: 120, y: 480, isRetry: true },
  query_transformer: { label: 'Query Transform', color: '#6b7280', x: 380, y: 480, isRetry: true },
}

// Edge definitions - organized by flow type
const edgeDefinitions: { source: NodeName; target: NodeName; label?: string; isRetryEdge?: boolean }[] = [
  // === MAIN FLOW ===
  { source: 'query_evaluator', target: 'agent' },
  { source: 'agent', target: 'tools', label: 'search' },
  { source: 'agent', target: 'response_grader', label: 'respond' },
  { source: 'tools', target: 'document_grader' },
  { source: 'document_grader', target: 'agent', label: 'pass' },
  // === RETRY EDGES (from graders to retry nodes) ===
  { source: 'document_grader', target: 'query_transformer', label: 'fail', isRetryEdge: true },
  { source: 'response_grader', target: 'response_improver', label: 'fail', isRetryEdge: true },
  // === FEEDBACK EDGES (from retry nodes back to main flow) ===
  { source: 'query_transformer', target: 'query_evaluator', label: 'retry', isRetryEdge: true },
  { source: 'response_improver', target: 'agent', label: 'retry', isRetryEdge: true },
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
    const nodeList: Node[] = Object.entries(nodeDefinitions).map(([id, def]) => {
      const isVisited = visitedNodes.has(id)
      const isCurrent = currentNode === id
      const isRetryNode = def.isRetry ?? false

      // Determine node state for accessibility
      let state = 'pending'
      if (isCurrent) state = 'current'
      else if (isVisited) state = 'visited'

      // Retry nodes have muted colors and dashed borders
      const bgColor = isVisited
        ? (isRetryNode ? '#4b5563' : def.color)  // Gray when visited for retry nodes
        : '#374151'

      return {
        id,
        position: { x: def.x, y: def.y },
        data: {
          label: def.label,
          // Accessibility label for screen readers
          ariaLabel: `${def.label}, ${state}${isRetryNode ? ', retry node' : ''}`
        },
        style: {
          background: bgColor,
          color: '#fff',
          border: isCurrent
            ? '3px solid #fff'
            : isRetryNode
              ? '2px dashed #6b7280'  // Dashed border for retry nodes
              : '2px solid #4b5563',
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

    // Add separator line as a special node
    nodeList.push({
      id: 'separator',
      position: { x: 0, y: 410 },
      data: { label: '── Retry Loops ──' },
      style: {
        background: 'transparent',
        color: '#6b7280',
        border: 'none',
        borderRadius: '0',
        padding: '4px 16px',
        fontSize: '11px',
        fontWeight: 400,
        width: 520,
        textAlign: 'center' as const,
        borderTop: '1px dashed #4b5563',
      },
      draggable: false,
      selectable: false,
    })

    return nodeList
  }, [visitedNodes, currentNode])

  // Build edges with visited styling
  const edges: Edge[] = useMemo(() => {
    return edgeDefinitions.map((def, index) => {
      const sourceVisited = visitedNodes.has(def.source)
      const targetVisited = visitedNodes.has(def.target)
      const isActive = sourceVisited && targetVisited
      const isRetryEdge = def.isRetryEdge ?? false

      // Retry edges use dashed lines and muted colors
      const strokeColor = isActive
        ? (isRetryEdge ? '#f59e0b' : '#60a5fa')  // Orange for retry, blue for main
        : '#4b5563'

      return {
        id: `e${index}`,
        source: def.source,
        target: def.target,
        label: def.label,
        type: 'smoothstep',
        animated: currentNode === def.source && isExecuting,
        style: {
          stroke: strokeColor,
          strokeWidth: isActive ? 2 : 1,
          strokeDasharray: isRetryEdge ? '5,5' : 'none',  // Dashed for retry edges
          opacity: isActive ? 1 : 0.4,
        },
        labelStyle: {
          fill: isRetryEdge ? '#f59e0b' : '#9ca3af',
          fontSize: 10,
        },
        labelBgStyle: {
          fill: '#1f2937',
        },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: strokeColor,
        },
      }
    })
  }, [visitedNodes, currentNode, isExecuting])

  const onInit = useCallback(() => {
    // Graph initialized
  }, [])

  return (
    <div className="h-full w-full relative">
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
          aria-label="Graph controls: use keyboard or mouse to pan and zoom"
        />
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="#374151"
        />
      </ReactFlow>

      {/* Legend */}
      <div
        className="absolute bottom-4 right-4 bg-gray-800/90 rounded-lg p-3 text-xs"
        aria-label="Agent workflow legend"
      >
        <h3 className="font-semibold text-gray-300 mb-2 text-xs">Workflow Status</h3>
        <dl className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-blue-500" aria-hidden="true" />
            <dt className="text-gray-400">Visited:</dt>
            <dd className="sr-only">Node has been executed</dd>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gray-600 border-2 border-white" aria-hidden="true" />
            <dt className="text-gray-400">Current:</dt>
            <dd className="sr-only">Node is currently executing</dd>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded bg-gray-600 opacity-50" aria-hidden="true" />
            <dt className="text-gray-400">Pending:</dt>
            <dd className="sr-only">Node has not been executed yet</dd>
          </div>
        </dl>
      </div>

      {/* Accessibility announcement */}
      <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
        {currentNode ? `Currently executing: ${nodeDefinitions[currentNode as NodeName]?.label || currentNode}` : 'Waiting for query...'}
      </div>
    </div>
  )
}
