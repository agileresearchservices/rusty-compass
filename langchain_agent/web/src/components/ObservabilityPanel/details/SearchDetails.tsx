/**
 * SearchDetails - Shows search candidates and reranking results.
 */

import { useState } from 'react'
import { useObservabilityStore } from '../../../stores/observabilityStore'
import { FileText, ArrowUp, ArrowDown, Minus, ChevronDown, ChevronUp } from 'lucide-react'
import clsx from 'clsx'

export function SearchDetails() {
  const { searchCandidates, rerankedDocuments } = useObservabilityStore()
  const [expandedDocs, setExpandedDocs] = useState<Set<number>>(new Set())

  // Toggle document expansion
  const toggleDocExpansion = (index: number) => {
    setExpandedDocs((prev) => {
      const newSet = new Set(prev)
      if (newSet.has(index)) {
        newSet.delete(index)
      } else {
        newSet.add(index)
      }
      return newSet
    })
  }

  // If we have reranked documents, show those; otherwise show candidates
  const documents = rerankedDocuments.length > 0 ? rerankedDocuments : null
  const candidates = searchCandidates

  if (!documents && candidates.length === 0) {
    return (
      <div className="text-sm text-gray-500">
        Waiting for search results...
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Show reranked documents if available */}
      {documents && documents.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">Reranked Results</span>
            <span className="text-xs text-purple-400">
              {documents.length} documents
            </span>
          </div>

          <div className="space-y-2">
            {documents.map((doc, index) => {
              const isExpanded = expandedDocs.has(index)
              const hasVectorScore = doc.vector_score !== undefined
              const hasTextScore = doc.text_score !== undefined
              const hasRrfScore = doc.rrf_score !== undefined

              return (
                <div
                  key={index}
                  className="bg-gray-800/50 rounded-lg p-3 space-y-2"
                >
                  {/* Header with rank and score */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="w-6 h-6 rounded bg-purple-500/20 text-purple-400 flex items-center justify-center text-xs font-medium">
                        {doc.rank}
                      </span>
                      <FileText className="w-4 h-4 text-gray-500" />
                      <span className="text-sm text-gray-300 truncate max-w-[200px]">
                        {doc.source}
                      </span>
                    </div>

                    {/* Rank change indicator */}
                    <div className="flex items-center gap-2">
                      {doc.rank_change !== 0 && (
                        <span
                          className={clsx(
                            'flex items-center text-xs',
                            doc.rank_change > 0 ? 'text-green-400' : 'text-red-400'
                          )}
                        >
                          {doc.rank_change > 0 ? (
                            <>
                              <ArrowUp className="w-3 h-3" />
                              {doc.rank_change}
                            </>
                          ) : (
                            <>
                              <ArrowDown className="w-3 h-3" />
                              {Math.abs(doc.rank_change)}
                            </>
                          )}
                        </span>
                      )}
                      {doc.rank_change === 0 && (
                        <span className="flex items-center text-xs text-gray-500">
                          <Minus className="w-3 h-3" />
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Score bar - Relevance Score */}
                  <div className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-500">Relevance Score</span>
                      <span className="text-purple-400">{(doc.score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="score-bar">
                      <div
                        className="score-bar-fill bg-gradient-to-r from-purple-600 to-purple-400"
                        style={{ width: `${doc.score * 100}%` }}
                      />
                    </div>
                  </div>

                  {/* Score badges - show component scores when expanded */}
                  {isExpanded && (
                    <div className="flex flex-wrap gap-2 pt-1">
                      {hasVectorScore && (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-blue-500/20 text-blue-400 text-xs">
                          <span className="font-medium">Vector:</span>
                          <span>{(doc.vector_score * 100).toFixed(1)}%</span>
                        </span>
                      )}
                      {hasTextScore && (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-green-500/20 text-green-400 text-xs">
                          <span className="font-medium">Text:</span>
                          <span>{(doc.text_score * 100).toFixed(1)}%</span>
                        </span>
                      )}
                      {hasRrfScore && (
                        <span className="inline-flex items-center gap-1 px-2 py-1 rounded bg-orange-500/20 text-orange-400 text-xs">
                          <span className="font-medium">RRF:</span>
                          <span>{(doc.rrf_score * 100).toFixed(1)}%</span>
                        </span>
                      )}
                    </div>
                  )}

                  {/* Snippet - collapsed view */}
                  {!isExpanded && (
                    <p className="text-xs text-gray-400 line-clamp-2">
                      {doc.snippet}
                    </p>
                  )}

                  {/* Full content - expanded view */}
                  {isExpanded && (
                    <div className="max-h-48 overflow-y-auto rounded bg-gray-900/50 p-3 border border-gray-700/50">
                      <div className="text-xs text-gray-300 space-y-2">
                        {doc.page_content && (
                          <p className="whitespace-pre-wrap">{doc.page_content}</p>
                        )}
                        {!doc.page_content && doc.snippet && (
                          <p className="whitespace-pre-wrap">{doc.snippet}</p>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Toggle button */}
                  <button
                    onClick={() => toggleDocExpansion(index)}
                    className="w-full flex items-center justify-center gap-1 text-xs text-purple-400 hover:text-purple-300 hover:bg-purple-500/10 rounded py-1 transition-colors"
                  >
                    {isExpanded ? (
                      <>
                        <ChevronUp className="w-3 h-3" />
                        View less
                      </>
                    ) : (
                      <>
                        <ChevronDown className="w-3 h-3" />
                        View more
                      </>
                    )}
                  </button>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Show raw candidates if no reranked results */}
      {!documents && candidates.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-500">Search Candidates</span>
            <span className="text-xs text-violet-400">
              {candidates.length} found
            </span>
          </div>

          <div className="space-y-2 max-h-60 overflow-y-auto">
            {candidates.slice(0, 6).map((candidate, index) => (
              <div
                key={index}
                className="bg-gray-800/50 rounded-lg p-2 text-xs"
              >
                <div className="flex items-center gap-2 mb-1">
                  <FileText className="w-3 h-3 text-gray-500" />
                  <span className="text-gray-300 truncate">
                    {candidate.source}
                  </span>
                </div>
                <p className="text-gray-500 line-clamp-2">
                  {candidate.snippet}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Explanation */}
      <div className="text-xs text-gray-500 border-t border-gray-700 pt-3">
        <p>
          <strong>Hybrid search</strong> combines BM25 (keyword) and vector similarity.
          The <strong>reranker</strong> (BGE) then scores each document for
          relevance using a cross-encoder model.
        </p>
      </div>
    </div>
  )
}
