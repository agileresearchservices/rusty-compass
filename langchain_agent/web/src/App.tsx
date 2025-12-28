import { useEffect } from 'react'
import { Layout } from './components/Layout'
import { useChatStore } from './stores/chatStore'
import { useWebSocket } from './hooks/useWebSocket'

function App() {
  const { threadId, setThreadId } = useChatStore()
  const { connect, disconnect, isConnected } = useWebSocket()

  // Generate initial thread ID if needed
  useEffect(() => {
    if (!threadId) {
      const newThreadId = `conversation_${Math.random().toString(36).slice(2, 10)}`
      setThreadId(newThreadId)
    }
  }, [threadId, setThreadId])

  // Connect WebSocket when thread ID is available
  useEffect(() => {
    if (threadId && !isConnected) {
      connect(threadId)
    }

    return () => {
      disconnect()
    }
  }, [threadId, connect, disconnect, isConnected])

  return <Layout />
}

export default App
