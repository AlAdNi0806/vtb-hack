'use client'

import { useState, useRef, useCallback } from 'react'

interface UseWebSocketOptions {
  url: string
  onConnect?: () => void
  onDisconnect?: () => void
  onMessage?: (data: any) => void
  onError?: (error: Event) => void
}

export function useWebSocket({
  url,
  onConnect,
  onDisconnect,
  onMessage,
  onError
}: UseWebSocketOptions) {
  const [isConnected, setIsConnected] = useState(false)
  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttempts = useRef(0)
  const maxReconnectAttempts = 5

  const connect = useCallback(async () => {
    try {
      // Close existing connection if any
      if (socketRef.current) {
        socketRef.current.close()
      }

      const socket = new WebSocket(url)
      socketRef.current = socket

      socket.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        reconnectAttempts.current = 0
        onConnect?.()
      }

      socket.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setIsConnected(false)
        socketRef.current = null
        onDisconnect?.()

        // Auto-reconnect logic
        if (!event.wasClean && reconnectAttempts.current < maxReconnectAttempts) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000)
          console.log(`Attempting to reconnect in ${delay}ms...`)
          
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectAttempts.current++
            connect()
          }, delay)
        }
      }

      socket.onerror = (error) => {
        console.error('WebSocket error:', error)
        onError?.(error)
      }

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      return socket
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
      throw error
    }
  }, [url, onConnect, onDisconnect, onMessage, onError])

  const disconnect = useCallback(() => {
    // Clear reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    // Close socket
    if (socketRef.current) {
      socketRef.current.close(1000, 'User disconnected')
      socketRef.current = null
    }

    setIsConnected(false)
    reconnectAttempts.current = 0
  }, [])

  const sendMessage = useCallback((message: any) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      try {
        socketRef.current.send(JSON.stringify(message))
        return true
      } catch (error) {
        console.error('Error sending WebSocket message:', error)
        return false
      }
    } else {
      console.warn('WebSocket is not connected')
      return false
    }
  }, [])

  return {
    socket: socketRef.current,
    isConnected,
    connect,
    disconnect,
    sendMessage
  }
}
