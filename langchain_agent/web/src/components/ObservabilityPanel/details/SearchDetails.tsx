/**
 * SearchDetails - Shows search candidates and reranking results.
 */

import { useObservabilityStore } from '../../../stores/observabilityStore'
import { FileText, ArrowUp, ArrowDown, Minus } from 'lucide-react'
import clsx from 'clsx'

export function SearchDetails() {
  const { searchCandidates, rerankedDocuments } = useObservabilityStore()

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
            {documents.map((doc, index) => (
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

                {/* Score bar */}
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

                {/* Snippet */}
                <p className="text-xs text-gray-400 line-clamp-2">
                  {doc.snippet}
                </p>
              </div>
            ))}
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
          The <strong>reranker</strong> (Qwen3) then scores each document for
          relevance using a cross-encoder model.
        </p>
      </div>
    </div>
  )
}
