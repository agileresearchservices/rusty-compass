/**
 * WebSocket hook for real-time communication with the agent API.
 * Handles connection, message sending, and event processing.
 */

import { useCallback, useRef } from 'react'
import { useChatStore, type ChatMessage } from '../stores/chatStore'
import { useObservabilityStore } from '../stores/observabilityStore'
import type { AgentEvent, NodeName } from '../types/events'

// Singleton WebSocket instance
let wsInstance: WebSocket | null = null
let currentThreadId: string | null = null

interface UseWebSocketReturn {
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  connect: (threadId: string) => void
  disconnect: () => void
  sendMessage: (message: string) => void
}

export function useWebSocket(): UseWebSocketReturn {
  const { isConnected, isConnecting, connectionError, setConnectionState } = useChatStore()

  const threadIdRef = useRef<string | null>(null)

  const handleMessage = useCallback((data: AgentEvent) => {
    const chatStore = useChatStore.getState()
    const obsStore = useObservabilityStore.getState()

    // Add event to observability store
    obsStore.addEvent(data)

    switch (data.type) {
      case 'connection_established':
        console.log('WebSocket connected:', data)
        break

      case 'node_start':
        obsStore.startNode(data.node as NodeName, data.input_summary)
        break

      case 'node_end':
        obsStore.endNode(data.node as NodeName, data.duration_ms, data.output_summary)
        break

      case 'llm_response_start':
        // Start streaming - add placeholder assistant message
        chatStore.addMessage({
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        })
        chatStore.setStreamingContent('')
        break

      case 'llm_response_chunk':
        if (data.content) {
          chatStore.appendStreamingContent(data.content)
        }
        break

      case 'agent_complete':
        // Finalize the streaming message with the complete response
        if (data.final_response) {
          chatStore.setStreamingContent(data.final_response)
        }
        chatStore.finalizeStreaming()
        obsStore.endExecution()
        break

      case 'agent_error':
        console.error('Agent error:', data.error)
        chatStore.setConnectionState(false, false, data.error)
        chatStore.finalizeStreaming()
        obsStore.endExecution()
        break

      default:
        // Other events are handled by addEvent above
        break
    }
  }, [])

  const connect = useCallback((threadId: string) => {
    // If already connected to this thread, do nothing
    if (wsInstance?.readyState === WebSocket.OPEN && currentThreadId === threadId) {
      return
    }

    // Close existing connection if any
    if (wsInstance) {
      wsInstance.close()
      wsInstance = null
    }

    setConnectionState(false, true, null)
    threadIdRef.current = threadId
    currentThreadId = threadId

    // Build WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}/ws/chat?thread_id=${threadId}`

    console.log('Connecting to WebSocket:', url)
    const ws = new WebSocket(url)

    ws.onopen = () => {
      setConnectionState(true, false, null)
      console.log('WebSocket connected')
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as AgentEvent
        handleMessage(data)
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e)
      }
    }

    ws.onerror = (event) => {
      console.error('WebSocket error:', event)
      setConnectionState(false, false, 'WebSocket connection error')
    }

    ws.onclose = () => {
      setConnectionState(false, false, null)
      wsInstance = null
      console.log('WebSocket disconnected')
    }

    wsInstance = ws
  }, [handleMessage, setConnectionState])

  const disconnect = useCallback(() => {
    if (wsInstance) {
      wsInstance.close()
      wsInstance = null
    }
    threadIdRef.current = null
    currentThreadId = null
    setConnectionState(false, false, null)
  }, [setConnectionState])

  const sendMessage = useCallback((message: string) => {
    if (!wsInstance || wsInstance.readyState !== WebSocket.OPEN) {
      setConnectionState(false, false, 'WebSocket not connected')
      return
    }

    // Add user message to chat
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: message,
      timestamp: new Date(),
    }
    useChatStore.getState().addMessage(userMessage)

    // Set processing state
    useChatStore.getState().setIsProcessing(true)

    // Clear previous observability state and start new execution
    useObservabilityStore.getState().clearState()
    useObservabilityStore.getState().startExecution()

    // Send message over WebSocket
    const payload = {
      type: 'chat_message',
      message,
      thread_id: currentThreadId,
    }

    wsInstance.send(JSON.stringify(payload))
  }, [setConnectionState])

  return {
    isConnected,
    isConnecting,
    error: connectionError,
    connect,
    disconnect,
    sendMessage,
  }
}
