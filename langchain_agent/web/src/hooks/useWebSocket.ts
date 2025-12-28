/**
 * WebSocket hook for real-time communication with the agent API.
 * Handles connection, message sending, and event processing.
 */

import { useCallback, useRef, useState } from 'react'
import { useChatStore, type ChatMessage } from '../stores/chatStore'
import { useObservabilityStore } from '../stores/observabilityStore'
import type { AgentEvent, NodeName } from '../types/events'

interface UseWebSocketReturn {
  isConnected: boolean
  isConnecting: boolean
  error: string | null
  connect: (threadId: string) => void
  disconnect: () => void
  sendMessage: (message: string) => void
}

export function useWebSocket(): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const threadIdRef = useRef<string | null>(null)

  // Chat store actions
  const {
    addMessage,
    setIsProcessing,
    setStreamingContent,
    appendStreamingContent,
    finalizeStreaming,
  } = useChatStore()

  // Observability store actions
  const {
    startExecution,
    endExecution,
    addEvent,
    startNode,
    endNode,
    clearState,
  } = useObservabilityStore()

  const handleMessage = useCallback((data: AgentEvent) => {
    // Add event to observability store
    addEvent(data)

    switch (data.type) {
      case 'connection_established':
        console.log('WebSocket connected:', data)
        break

      case 'node_start':
        startNode(data.node as NodeName, data.input_summary)
        break

      case 'node_end':
        endNode(data.node as NodeName, data.duration_ms, data.output_summary)
        break

      case 'llm_response_start':
        // Start streaming - add placeholder assistant message
        addMessage({
          id: `msg-${Date.now()}`,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
        })
        setStreamingContent('')
        break

      case 'llm_response_chunk':
        if (data.content) {
          appendStreamingContent(data.content)
        }
        break

      case 'agent_complete':
        // Finalize the streaming message with the complete response
        if (data.final_response) {
          setStreamingContent(data.final_response)
        }
        finalizeStreaming()
        endExecution()
        break

      case 'agent_error':
        console.error('Agent error:', data.error)
        setError(data.error)
        finalizeStreaming()
        endExecution()
        break

      default:
        // Other events are handled by addEvent above
        break
    }
  }, [
    addEvent,
    startNode,
    endNode,
    addMessage,
    setStreamingContent,
    appendStreamingContent,
    finalizeStreaming,
    endExecution,
    setError,
  ])

  const connect = useCallback((threadId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    setIsConnecting(true)
    setError(null)
    threadIdRef.current = threadId

    // Build WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}/ws/chat?thread_id=${threadId}`

    const ws = new WebSocket(url)

    ws.onopen = () => {
      setIsConnected(true)
      setIsConnecting(false)
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
      setError('WebSocket connection error')
      setIsConnecting(false)
    }

    ws.onclose = () => {
      setIsConnected(false)
      setIsConnecting(false)
      console.log('WebSocket disconnected')
    }

    wsRef.current = ws
  }, [handleMessage])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    threadIdRef.current = null
    setIsConnected(false)
  }, [])

  const sendMessage = useCallback((message: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError('WebSocket not connected')
      return
    }

    // Add user message to chat
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: message,
      timestamp: new Date(),
    }
    addMessage(userMessage)

    // Set processing state
    setIsProcessing(true)

    // Clear previous observability state and start new execution
    clearState()
    startExecution()

    // Send message over WebSocket
    const payload = {
      type: 'chat_message',
      message,
      thread_id: threadIdRef.current,
    }

    wsRef.current.send(JSON.stringify(payload))
  }, [addMessage, setIsProcessing, clearState, startExecution])

  return {
    isConnected,
    isConnecting,
    error,
    connect,
    disconnect,
    sendMessage,
  }
}
