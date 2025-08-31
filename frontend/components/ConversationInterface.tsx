'use client'

import { useState, useEffect, useRef } from 'react'
import { Play, Square, Mic, MicOff, Phone, PhoneOff } from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'ai'
  text: string
  timestamp: Date
  audioData?: string
}

interface ConversationInterfaceProps {
  socket: WebSocket | null
  isConnected: boolean
  isRecording: boolean
  onConnect: () => void
  onDisconnect: () => void
  onStartConversation: () => void
  onEndConversation: () => void
}

export default function ConversationInterface({
  socket,
  isConnected,
  isRecording,
  onConnect,
  onDisconnect,
  onStartConversation,
  onEndConversation
}: ConversationInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [currentTranscript, setCurrentTranscript] = useState('')
  const [isAIResponding, setIsAIResponding] = useState(false)
  const [conversationActive, setConversationActive] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Handle WebSocket messages
  useEffect(() => {
    if (!socket) return

    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data)
        
        switch (data.type) {
          case 'connection_established':
            console.log('Connected to AI:', data.message)
            break
            
          case 'transcript':
            if (data.is_final) {
              // Add user message to conversation
              const userMessage: Message = {
                id: Date.now().toString(),
                type: 'user',
                text: data.text,
                timestamp: new Date()
              }
              setMessages(prev => [...prev, userMessage])
              setCurrentTranscript('')
            } else {
              // Update current transcript (partial or streaming)
              const transcriptText = data.text || data.partial_transcript || ''
              setCurrentTranscript(transcriptText)
            }
            break
            
          case 'ai_response':
            setIsAIResponding(false)
            
            // Add AI message to conversation
            const aiMessage: Message = {
              id: Date.now().toString(),
              type: 'ai',
              text: data.text,
              timestamp: new Date(),
              audioData: data.audio_data
            }
            setMessages(prev => [...prev, aiMessage])
            
            // Play AI response audio
            if (data.audio_data) {
              playAudioResponse(data.audio_data)
            }
            break
            
          case 'conversation_started':
            setConversationActive(true)
            break
            
          case 'conversation_ended':
            setConversationActive(false)
            setCurrentTranscript('')
            break
            
          case 'error':
            console.error('AI Error:', data.message)
            break
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    socket.addEventListener('message', handleMessage)
    
    return () => {
      socket.removeEventListener('message', handleMessage)
    }
  }, [socket])

  const playAudioResponse = async (audioData: string) => {
    try {
      if (audioRef.current) {
        const audioBlob = new Blob([
          Uint8Array.from(atob(audioData), c => c.charCodeAt(0))
        ], { type: 'audio/wav' })
        
        const audioUrl = URL.createObjectURL(audioBlob)
        audioRef.current.src = audioUrl
        await audioRef.current.play()
        
        // Clean up URL after playing
        audioRef.current.onended = () => {
          URL.revokeObjectURL(audioUrl)
        }
      }
    } catch (error) {
      console.error('Error playing audio response:', error)
    }
  }

  const handleStartConversation = () => {
    onStartConversation()
    setMessages([])
    setCurrentTranscript('')
  }

  const handleEndConversation = () => {
    onEndConversation()
    setConversationActive(false)
    setCurrentTranscript('')
  }

  const clearConversation = () => {
    setMessages([])
    setCurrentTranscript('')
  }

  return (
    <div className="space-y-6">
      {/* Control Buttons */}
      <div className="flex items-center justify-center space-x-4">
        {!isConnected ? (
          <button
            onClick={onConnect}
            className="btn-primary flex items-center space-x-2"
          >
            <Phone className="w-5 h-5" />
            <span>Connect to AI</span>
          </button>
        ) : (
          <>
            {!conversationActive ? (
              <button
                onClick={handleStartConversation}
                className="btn-primary flex items-center space-x-2"
              >
                <Mic className="w-5 h-5" />
                <span>Start Conversation</span>
              </button>
            ) : (
              <button
                onClick={handleEndConversation}
                className="btn-danger flex items-center space-x-2"
              >
                <Square className="w-5 h-5" />
                <span>End Conversation</span>
              </button>
            )}
            
            <button
              onClick={onDisconnect}
              className="btn-secondary flex items-center space-x-2"
            >
              <PhoneOff className="w-5 h-5" />
              <span>Disconnect</span>
            </button>
            
            {messages.length > 0 && (
              <button
                onClick={clearConversation}
                className="btn-secondary"
              >
                Clear Chat
              </button>
            )}
          </>
        )}
      </div>

      {/* Conversation Display */}
      <div className="bg-gray-50 rounded-lg p-4 min-h-[400px] max-h-[600px] overflow-y-auto">
        {messages.length === 0 && !currentTranscript ? (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <Mic className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p>Start a conversation to see messages here</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                    message.type === 'user'
                      ? 'bg-primary-600 text-white'
                      : 'bg-white text-gray-800 shadow-sm border'
                  }`}
                >
                  <p className="text-sm">{message.text}</p>
                  <p className={`text-xs mt-1 ${
                    message.type === 'user' ? 'text-primary-200' : 'text-gray-500'
                  }`}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
            
            {/* Current transcript (live) */}
            {currentTranscript && (
              <div className="flex justify-end">
                <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-primary-200 text-primary-800 border-2 border-primary-300 border-dashed">
                  <p className="text-sm">{currentTranscript}</p>
                  <p className="text-xs mt-1 text-primary-600">Speaking...</p>
                </div>
              </div>
            )}
            
            {/* AI thinking indicator */}
            {isAIResponding && (
              <div className="flex justify-start">
                <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-white text-gray-800 shadow-sm border">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                    </div>
                    <span className="text-sm text-gray-600">AI is thinking...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Recording Status */}
      {conversationActive && (
        <div className="text-center">
          <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full ${
            isRecording ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-600'
          }`}>
            {isRecording ? (
              <>
                <Mic className="w-4 h-4" />
                <span className="text-sm font-medium">Listening...</span>
              </>
            ) : (
              <>
                <MicOff className="w-4 h-4" />
                <span className="text-sm font-medium">Ready to listen</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Hidden audio element for playing AI responses */}
      <audio ref={audioRef} className="hidden" />
    </div>
  )
}
