/**
 * Layout - Main application layout with three-panel design.
 *
 * Desktop:
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  Conversations  │         Chat          │    Observability      │
 * │    Sidebar      │        Panel          │       Panel           │
 * │   (250px)       │       (50%)           │       (50%)           │
 * └─────────────────┴───────────────────────┴───────────────────────┘
 *
 * Mobile: Sidebar in drawer, Chat full width, Observability hidden
 */

import { useState } from 'react'
import { Menu, X } from 'lucide-react'
import { ConversationsSidebar } from './ConversationsSidebar'
import { ChatPanel } from './ChatPanel'
import { ObservabilityPanel } from './ObservabilityPanel'

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const closeSidebar = () => setSidebarOpen(false)

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Mobile menu button */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label={sidebarOpen ? 'Close conversations menu' : 'Open conversations menu'}
        aria-expanded={sidebarOpen}
        className="md:hidden fixed top-4 left-4 z-40 p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
      >
        {sidebarOpen ? (
          <X className="w-6 h-6" />
        ) : (
          <Menu className="w-6 h-6" />
        )}
      </button>

      {/* Mobile overlay backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={closeSidebar}
          aria-hidden="true"
        />
      )}

      {/* Sidebar - desktop visible, mobile in drawer */}
      <div
        className={`${
          sidebarOpen
            ? 'fixed inset-y-0 left-0 z-40 w-64'
            : 'hidden md:block md:w-64 md:flex-shrink-0'
        }`}
      >
        <ConversationsSidebar onConversationSelect={closeSidebar} />
      </div>

      {/* Main content area */}
      <div className="flex-1 flex min-w-0">
        {/* Chat panel */}
        <div className="flex-1 min-w-0 border-r border-gray-800">
          <ChatPanel />
        </div>

        {/* Observability panel */}
        <div className="w-1/2 min-w-[400px] hidden lg:block">
          <ObservabilityPanel />
        </div>
      </div>
    </div>
  )
}
