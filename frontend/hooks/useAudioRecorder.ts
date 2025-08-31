'use client'

import { useState, useRef, useCallback, useEffect } from 'react'

interface UseAudioRecorderOptions {
  onAudioData?: (audioData: string) => void
  sampleRate?: number
  chunkSize?: number
}

export function useAudioRecorder({
  onAudioData,
  sampleRate = 16000,
  chunkSize = 1024
}: UseAudioRecorderOptions) {
  const [isRecording, setIsRecording] = useState(false)
  const [audioLevel, setAudioLevel] = useState(0)
  const [hasPermission, setHasPermission] = useState<boolean | null>(null)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // Check microphone permission on mount
  useEffect(() => {
    checkMicrophonePermission()
  }, [])

  const checkMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      setHasPermission(true)
      stream.getTracks().forEach(track => track.stop()) // Stop the test stream
    } catch (error) {
      console.error('Microphone permission denied:', error)
      setHasPermission(false)
    }
  }

  const requestMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: sampleRate,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      })
      setHasPermission(true)
      return stream
    } catch (error) {
      console.error('Failed to get microphone access:', error)
      setHasPermission(false)
      throw error
    }
  }

  const startRecording = useCallback(async () => {
    try {
      if (isRecording) return

      // Request microphone access
      const stream = await requestMicrophonePermission()
      streamRef.current = stream

      // Create audio context
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: sampleRate
      })

      const audioContext = audioContextRef.current
      const source = audioContext.createMediaStreamSource(stream)

      // Create analyser for audio level monitoring
      analyserRef.current = audioContext.createAnalyser()
      analyserRef.current.fftSize = 256
      source.connect(analyserRef.current)

      // Create script processor for real-time audio processing
      processorRef.current = audioContext.createScriptProcessor(chunkSize, 1, 1)
      
      processorRef.current.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer
        const inputData = inputBuffer.getChannelData(0)
        
        // Convert to base64 and send
        if (onAudioData) {
          const audioData = convertToBase64(inputData)
          onAudioData(audioData)
        }
      }

      source.connect(processorRef.current)
      processorRef.current.connect(audioContext.destination)

      // Start audio level monitoring
      startAudioLevelMonitoring()

      setIsRecording(true)
      console.log('Recording started')

    } catch (error) {
      console.error('Failed to start recording:', error)
      throw error
    }
  }, [isRecording, onAudioData, sampleRate, chunkSize])

  const stopRecording = useCallback(() => {
    try {
      setIsRecording(false)

      // Stop audio level monitoring
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current)
        animationFrameRef.current = null
      }

      // Clean up audio context
      if (processorRef.current) {
        processorRef.current.disconnect()
        processorRef.current = null
      }

      if (analyserRef.current) {
        analyserRef.current.disconnect()
        analyserRef.current = null
      }

      if (audioContextRef.current) {
        audioContextRef.current.close()
        audioContextRef.current = null
      }

      // Stop media stream
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
        streamRef.current = null
      }

      setAudioLevel(0)
      console.log('Recording stopped')

    } catch (error) {
      console.error('Error stopping recording:', error)
    }
  }, [])

  const startAudioLevelMonitoring = () => {
    if (!analyserRef.current) return

    const analyser = analyserRef.current
    const bufferLength = analyser.frequencyBinCount
    const dataArray = new Uint8Array(bufferLength)

    const updateAudioLevel = () => {
      if (!analyser || !isRecording) return

      analyser.getByteFrequencyData(dataArray)
      
      // Calculate average volume
      let sum = 0
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i]
      }
      const average = sum / bufferLength
      const normalizedLevel = average / 255 // Normalize to 0-1

      setAudioLevel(normalizedLevel)
      
      animationFrameRef.current = requestAnimationFrame(updateAudioLevel)
    }

    updateAudioLevel()
  }

  const convertToBase64 = (audioData: Float32Array): string => {
    try {
      // Convert Float32Array to Int16Array (16-bit PCM)
      const int16Array = new Int16Array(audioData.length)
      for (let i = 0; i < audioData.length; i++) {
        int16Array[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32767))
      }

      // Convert to bytes
      const bytes = new Uint8Array(int16Array.buffer)
      
      // Convert to base64
      let binary = ''
      for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i])
      }
      
      return btoa(binary)
    } catch (error) {
      console.error('Error converting audio to base64:', error)
      return ''
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (isRecording) {
        stopRecording()
      }
    }
  }, [isRecording, stopRecording])

  return {
    isRecording,
    audioLevel,
    hasPermission,
    startRecording,
    stopRecording,
    requestMicrophonePermission
  }
}
