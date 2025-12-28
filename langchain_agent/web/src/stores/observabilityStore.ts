/**
 * Zustand store for observability state management.
 * Tracks agent execution steps, metrics, and events.
 */

import { create } from 'zustand'
import type {
  AgentEvent,
  NodeName,
  NodeStatus,
  ObservabilityStep,
  QueryEvaluationEvent,
  DocumentGradingSummaryEvent,
  ResponseGradingEvent,
  MetricsEvent,
  SearchCandidate,
  RerankedDocument,
} from '../types/events'

interface ObservabilityState {
  // Current execution state
  isExecuting: boolean
  currentNode: NodeName | null
  steps: ObservabilityStep[]

  // Key event data for display
  queryEvaluation: QueryEvaluationEvent | null
  searchCandidates: SearchCandidate[]
  rerankedDocuments: RerankedDocument[]
  documentGradingSummary: DocumentGradingSummaryEvent | null
  responseGrading: ResponseGradingEvent | null
  metrics: MetricsEvent | null

  // UI state
  activeTab: 'steps' | 'graph' | 'metrics'
  expandedSteps: Set<string>

  // Actions
  startExecution: () => void
  endExecution: () => void
  addEvent: (event: AgentEvent) => void
  startNode: (node: NodeName, summary?: string) => void
  endNode: (node: NodeName, durationMs: number, summary?: string) => void
  setActiveTab: (tab: 'steps' | 'graph' | 'metrics') => void
  toggleStepExpanded: (stepId: string) => void
  clearState: () => void
}

export const useObservabilityStore = create<ObservabilityState>((set, get) => ({
  // Initial state
  isExecuting: false,
  currentNode: null,
  steps: [],
  queryEvaluation: null,
  searchCandidates: [],
  rerankedDocuments: [],
  documentGradingSummary: null,
  responseGrading: null,
  metrics: null,
  activeTab: 'steps',
  expandedSteps: new Set(),

  // Actions
  startExecution: () => set({
    isExecuting: true,
    currentNode: null,
    steps: [],
    queryEvaluation: null,
    searchCandidates: [],
    rerankedDocuments: [],
    documentGradingSummary: null,
    responseGrading: null,
    metrics: null,
  }),

  endExecution: () => set({
    isExecuting: false,
    currentNode: null,
  }),

  addEvent: (event) => {
    const state = get()

    // Update specific event data based on type
    switch (event.type) {
      case 'query_evaluation':
        set({ queryEvaluation: event as QueryEvaluationEvent })
        break

      case 'hybrid_search_result':
        set({ searchCandidates: (event as { candidates: SearchCandidate[] }).candidates })
        break

      case 'reranker_result':
        set({ rerankedDocuments: (event as { results: RerankedDocument[] }).results })
        break

      case 'document_grading_summary':
        set({ documentGradingSummary: event as DocumentGradingSummaryEvent })
        break

      case 'response_grading':
        set({ responseGrading: event as ResponseGradingEvent })
        break

      case 'metrics':
        set({ metrics: event as MetricsEvent })
        break
    }

    // Add event to current step if there is one
    if (state.currentNode && state.steps.length > 0) {
      set((s) => {
        const steps = [...s.steps]
        const currentStepIndex = steps.findIndex(
          (step) => step.node === s.currentNode && step.status === 'running'
        )
        if (currentStepIndex >= 0) {
          steps[currentStepIndex] = {
            ...steps[currentStepIndex],
            events: [...steps[currentStepIndex].events, event],
          }
        }
        return { steps }
      })
    }
  },

  startNode: (node, summary) => {
    const stepId = `${node}-${Date.now()}`

    set((state) => ({
      currentNode: node,
      steps: [
        ...state.steps,
        {
          id: stepId,
          node,
          status: 'running' as NodeStatus,
          startTime: new Date(),
          events: [],
          summary,
        },
      ],
      expandedSteps: new Set([...state.expandedSteps, stepId]),
    }))
  },

  endNode: (node, durationMs, summary) => {
    set((state) => {
      const steps = [...state.steps]
      const stepIndex = steps.findIndex(
        (step) => step.node === node && step.status === 'running'
      )

      if (stepIndex >= 0) {
        steps[stepIndex] = {
          ...steps[stepIndex],
          status: 'complete',
          endTime: new Date(),
          durationMs,
          summary: summary || steps[stepIndex].summary,
        }
      }

      return {
        steps,
        currentNode: null,
      }
    })
  },

  setActiveTab: (tab) => set({ activeTab: tab }),

  toggleStepExpanded: (stepId) => set((state) => {
    const expandedSteps = new Set(state.expandedSteps)
    if (expandedSteps.has(stepId)) {
      expandedSteps.delete(stepId)
    } else {
      expandedSteps.add(stepId)
    }
    return { expandedSteps }
  }),

  clearState: () => set({
    isExecuting: false,
    currentNode: null,
    steps: [],
    queryEvaluation: null,
    searchCandidates: [],
    rerankedDocuments: [],
    documentGradingSummary: null,
    responseGrading: null,
    metrics: null,
    expandedSteps: new Set(),
  }),
}))
