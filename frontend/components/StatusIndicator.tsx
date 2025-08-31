'use client'

import { Wifi, WifiOff, Mic, MicOff, Volume2 } from 'lucide-react'
import AudioVisualizer from './AudioVisualizer'

interface StatusIndicatorProps {
  connectionStatus: 'disconnected' | 'connecting' | 'connected'
  isRecording: boolean
  audioLevel: number
}

export default function StatusIndicator({ 
  connectionStatus, 
  isRecording, 
  audioLevel 
}: StatusIndicatorProps) {
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'text-green-600 bg-green-100'
      case 'connecting':
        return 'text-yellow-600 bg-yellow-100'
      case 'disconnected':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-gray-600 bg-gray-100'
    }
  }

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected':
        return 'Connected'
      case 'connecting':
        return 'Connecting...'
      case 'disconnected':
        return 'Disconnected'
      default:
        return 'Unknown'
    }
  }

  return (
    <div className="flex items-center justify-center space-x-8">
      {/* Connection Status */}
      <div className="flex items-center space-x-2">
        <div className={`p-2 rounded-full ${getConnectionStatusColor()}`}>
          {connectionStatus === 'connected' ? (
            <Wifi className="w-5 h-5" />
          ) : (
            <WifiOff className="w-5 h-5" />
          )}
        </div>
        <div>
          <p className="text-sm font-medium text-gray-900">Connection</p>
          <p className={`text-xs ${
            connectionStatus === 'connected' ? 'text-green-600' :
            connectionStatus === 'connecting' ? 'text-yellow-600' :
            'text-red-600'
          }`}>
            {getConnectionStatusText()}
          </p>
        </div>
      </div>

      {/* Recording Status */}
      <div className="flex items-center space-x-2">
        <div className={`p-2 rounded-full ${
          isRecording ? 'text-red-600 bg-red-100 recording-pulse' : 'text-gray-600 bg-gray-100'
        }`}>
          {isRecording ? (
            <Mic className="w-5 h-5" />
          ) : (
            <MicOff className="w-5 h-5" />
          )}
        </div>
        <div>
          <p className="text-sm font-medium text-gray-900">Microphone</p>
          <p className={`text-xs ${isRecording ? 'text-red-600' : 'text-gray-600'}`}>
            {isRecording ? 'Recording' : 'Inactive'}
          </p>
        </div>
      </div>

      {/* Audio Level Visualizer */}
      {isRecording && (
        <div className="flex items-center space-x-2">
          <div className="p-2 rounded-full text-blue-600 bg-blue-100">
            <Volume2 className="w-5 h-5" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900">Audio Level</p>
            <AudioVisualizer audioLevel={audioLevel} isActive={isRecording} />
          </div>
        </div>
      )}
    </div>
  )
}
