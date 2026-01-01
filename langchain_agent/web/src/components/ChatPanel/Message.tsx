/**
 * Message - Single chat message display with markdown support.
 */

import ReactMarkdown from 'react-markdown'
import { User, Bot } from 'lucide-react'
import type { ChatMessage } from '../../stores/chatStore'
import clsx from 'clsx'

interface MessageProps {
  message: ChatMessage
}

export function Message({ message }: MessageProps) {
  const isUser = message.role === 'user'

  return (
    <div
      className={clsx(
        'flex gap-3 animate-slide-in',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser ? 'bg-blue-600' : 'bg-gray-700'
        )}
        aria-label={isUser ? 'You' : 'Agent'}
      >
        {isUser ? (
          <User className="w-4 h-4 text-white" aria-hidden="true" />
        ) : (
          <Bot className="w-4 h-4 text-gray-300" aria-hidden="true" />
        )}
      </div>

      {/* Message bubble */}
      <div
        className={clsx(
          'chat-message',
          isUser ? 'chat-message-user' : 'chat-message-assistant'
        )}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="markdown-content text-sm">
            <ReactMarkdown
              components={{
                // Headings
                h1: ({ children }) => (
                  <h1 className="text-lg font-bold text-white mt-4 mb-2">
                    {children}
                  </h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-base font-bold text-white mt-3 mb-2">
                    {children}
                  </h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-sm font-semibold text-gray-100 mt-2 mb-1">
                    {children}
                  </h3>
                ),
                h4: ({ children }) => (
                  <h4 className="text-sm font-semibold text-gray-200 mt-2 mb-1">
                    {children}
                  </h4>
                ),
                h5: ({ children }) => (
                  <h5 className="text-xs font-semibold text-gray-300 mt-2 mb-1">
                    {children}
                  </h5>
                ),
                h6: ({ children }) => (
                  <h6 className="text-xs font-semibold text-gray-400 mt-2 mb-1">
                    {children}
                  </h6>
                ),
                // Paragraphs
                p: ({ children }) => (
                  <p className="mb-3 leading-relaxed">
                    {children}
                  </p>
                ),
                // Code blocks
                pre: ({ children }) => (
                  <pre className="bg-gray-900 rounded-lg p-3 overflow-x-auto text-xs my-3 border border-gray-800">
                    {children}
                  </pre>
                ),
                code: ({ className, children, ...props }) => {
                  const isInline = !className
                  return isInline ? (
                    <code className="bg-gray-700 px-1.5 py-0.5 rounded text-xs text-gray-100" {...props}>
                      {children}
                    </code>
                  ) : (
                    <code className={className} {...props}>
                      {children}
                    </code>
                  )
                },
                // Links
                a: ({ children, href }) => (
                  <a
                    href={href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-400 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-800 rounded"
                  >
                    {children}
                  </a>
                ),
                // Lists
                ul: ({ children }) => (
                  <ul className="list-disc list-inside my-2 space-y-1 ml-2">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-inside my-2 space-y-1 ml-2">
                    {children}
                  </ol>
                ),
                li: ({ children }) => (
                  <li className="text-gray-100">
                    {children}
                  </li>
                ),
                // Tables
                table: ({ children }) => (
                  <div className="overflow-x-auto my-3">
                    <table className="w-full border-collapse text-xs">
                      {children}
                    </table>
                  </div>
                ),
                thead: ({ children }) => (
                  <thead className="bg-gray-800 border border-gray-700">
                    {children}
                  </thead>
                ),
                tbody: ({ children }) => (
                  <tbody className="border border-gray-700">
                    {children}
                  </tbody>
                ),
                tr: ({ children }) => (
                  <tr className="border-b border-gray-700 hover:bg-gray-800/50 transition-colors">
                    {children}
                  </tr>
                ),
                th: ({ children }) => (
                  <th className="px-3 py-2 text-left font-semibold text-gray-200 border-r border-gray-700 last:border-r-0 bg-gray-800">
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td className="px-3 py-2 text-gray-300 border-r border-gray-700 last:border-r-0">
                    {children}
                  </td>
                ),
                // Blockquotes
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-gray-600 pl-4 italic text-gray-300 my-3">
                    {children}
                  </blockquote>
                ),
                // Horizontal rule
                hr: () => (
                  <hr className="my-4 border-t border-gray-700" />
                ),
              }}
            >
              {message.content || '...'}
            </ReactMarkdown>
          </div>
        )}

        {/* Streaming indicator */}
        {message.isStreaming && (
          <span className="inline-block w-2 h-4 bg-gray-400 animate-pulse ml-0.5" aria-label="Generating response" aria-hidden="false" />
        )}
      </div>
    </div>
  )
}
