/**
 * Layout - Main application layout with three-panel design.
 *
 * ┌─────────────────────────────────────────────────────────────────┐
 * │  Conversations  │         Chat          │    Observability      │
 * │    Sidebar      │        Panel          │       Panel           │
 * │   (250px)       │       (50%)           │       (50%)           │
 * └─────────────────┴───────────────────────┴───────────────────────┘
 */

import { ConversationsSidebar } from './ConversationsSidebar'
import { ChatPanel } from './ChatPanel'
import { ObservabilityPanel } from './ObservabilityPanel'

export function Layout() {
  return (
    <div className="flex h-screen bg-gray-900 text-gray-100">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 hidden md:block">
        <ConversationsSidebar />
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
