'use client'

import { useState, useEffect } from 'react'
import ConversationInterface from '@/components/ConversationInterface'
import Header from '@/components/Header'
import StatusIndicator from '@/components/StatusIndicator'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useAudioRecorder } from '@/hooks/useAudioRecorder'

export default function Home() {
  const [isConnected, setIsConnected] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')
  
  const {
    socket,
    isConnected: wsConnected,
    connect,
    disconnect,
    sendMessage
  } = useWebSocket({
    url: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
    onConnect: () => {
      setIsConnected(true)
      setConnectionStatus('connected')
    },
    onDisconnect: () => {
      setIsConnected(false)
      setConnectionStatus('disconnected')
    },
    onError: (error) => {
      console.error('WebSocket error:', error)
      setConnectionStatus('disconnected')
    }
  })

  const {
    isRecording,
    startRecording,
    stopRecording,
    audioLevel
  } = useAudioRecorder({
    onAudioData: (audioData) => {
      if (wsConnected && audioData) {
        sendMessage({
          type: 'audio_chunk',
          audio_data: audioData
        })
      }
    }
  })

  const handleConnect = async () => {
    setConnectionStatus('connecting')
    try {
      await connect()
    } catch (error) {
      console.error('Failed to connect:', error)
      setConnectionStatus('disconnected')
    }
  }

  const handleDisconnect = () => {
    disconnect()
    if (isRecording) {
      stopRecording()
    }
  }

  const handleStartConversation = async () => {
    if (!wsConnected) {
      await handleConnect()
    }
    
    // Send start conversation message
    sendMessage({
      type: 'start_conversation'
    })
    
    // Start recording
    await startRecording()
  }

  const handleEndConversation = () => {
    // Send end conversation message
    sendMessage({
      type: 'end_conversation'
    })
    
    // Stop recording
    stopRecording()
  }

  return (
    <main className="min-h-screen flex flex-col">
      <Header />
      
      <div className="flex-1 container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto">
          {/* Status Section */}
          <div className="mb-8">
            <StatusIndicator 
              connectionStatus={connectionStatus}
              isRecording={isRecording}
              audioLevel={audioLevel}
            />
          </div>

          {/* Main Conversation Interface */}
          <div className="card p-6">
            <ConversationInterface
              socket={socket}
              isConnected={wsConnected}
              isRecording={isRecording}
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              onStartConversation={handleStartConversation}
              onEndConversation={handleEndConversation}
            />
          </div>

          {/* Instructions */}
          <div className="mt-8 card p-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">How to Use</h3>
            <div className="space-y-2 text-gray-600">
              <p>1. Click "Connect" to establish connection with the AI backend</p>
              <p>2. Click "Start Conversation" to begin voice interaction</p>
              <p>3. Speak naturally - the AI will detect when you finish talking</p>
              <p>4. Listen to the AI's response and continue the conversation</p>
              <p>5. Click "End Conversation" when you're done</p>
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
