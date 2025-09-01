"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Mic, MicOff, Volume2, AlertCircle } from "lucide-react";

interface TranscriptionMessage {
  type: "transcription" | "status";
  text?: string;
  is_final?: boolean;
  message?: string;
}

const PYTHON_BACKEND_URL = "192.168.0.176:8000";

export function SpeechToTextApp() {
  const [isRecording, setIsRecording] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [transcription, setTranscription] = useState("");
  const [partialTranscription, setPartialTranscription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [permissionGranted, setPermissionGranted] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const clientIdRef = useRef<string>(Math.random().toString(36).substring(7));

  // Request microphone permission
  const requestMicrophonePermission = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        } 
      });
      setPermissionGranted(true);
      setError(null);
      
      // Store the stream for later use
      streamRef.current = stream;
      
      return stream;
    } catch (err) {
      console.error("Microphone permission denied:", err);
      setError("Microphone permission is required for speech recognition");
      setPermissionGranted(false);
      return null;
    }
  }, []);

  // Initialize WebSocket connection
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(`ws://${PYTHON_BACKEND_URL}/ws/${clientIdRef.current}`);
    
    ws.onopen = () => {
      console.log("WebSocket connected");
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const data: TranscriptionMessage = JSON.parse(event.data);
        
        if (data.type === "transcription") {
          if (data.is_final) {
            // Final transcription - add to main text
            setTranscription(prev => prev + " " + (data.text || ""));
            setPartialTranscription("");
          } else {
            // Partial transcription - show as preview
            setPartialTranscription(data.text || "");
          }
        } else if (data.type === "status") {
          console.log("Status:", data.message);
        }
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setIsConnected(false);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setError("Connection to speech recognition server failed");
      setIsConnected(false);
    };

    wsRef.current = ws;
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    if (!permissionGranted) {
      const stream = await requestMicrophonePermission();
      if (!stream) return;
    }

    if (!isConnected) {
      connectWebSocket();
      // Wait a bit for connection to establish
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    try {
      const stream = streamRef.current || await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        } 
      });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          // Send audio data to backend
          wsRef.current.send(event.data);
        }
      };

      mediaRecorder.start(100); // Send data every 100ms
      mediaRecorderRef.current = mediaRecorder;
      
      // Send start recording message
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "start_recording" }));
      }

      setIsRecording(true);
      setError(null);
    } catch (err) {
      console.error("Error starting recording:", err);
      setError("Failed to start recording");
    }
  }, [permissionGranted, isConnected, connectWebSocket, requestMicrophonePermission]);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop_recording" }));
    }

    setIsRecording(false);
  }, []);

  // Toggle recording
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  // Clear transcription
  const clearTranscription = useCallback(() => {
    setTranscription("");
    setPartialTranscription("");
  }, []);

  // Initialize on component mount
  useEffect(() => {
    requestMicrophonePermission();
    connectWebSocket();

    return () => {
      // Cleanup
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [requestMicrophonePermission, connectWebSocket]);

  return (
    <div className="max-w-4xl mx-auto">
      {/* Status indicators */}
      <div className="flex justify-center gap-4 mb-6">
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
          permissionGranted 
            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" 
            : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
        }`}>
          <Mic className="w-4 h-4" />
          {permissionGranted ? "Microphone Ready" : "Microphone Access Needed"}
        </div>
        
        <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
          isConnected 
            ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200" 
            : "bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200"
        }`}>
          <Volume2 className="w-4 h-4" />
          {isConnected ? "Connected" : "Connecting..."}
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="flex items-center gap-2 p-4 mb-6 bg-red-100 border border-red-300 rounded-lg text-red-800 dark:bg-red-900 dark:border-red-700 dark:text-red-200">
          <AlertCircle className="w-5 h-5" />
          {error}
        </div>
      )}

      {/* Talk button */}
      <div className="flex justify-center mb-8">
        <button
          onClick={toggleRecording}
          disabled={!permissionGranted || !isConnected}
          className={`relative w-32 h-32 rounded-full border-4 transition-all duration-200 ${
            isRecording
              ? "bg-red-500 border-red-600 hover:bg-red-600 shadow-lg shadow-red-500/50"
              : "bg-blue-500 border-blue-600 hover:bg-blue-600 shadow-lg shadow-blue-500/50"
          } ${
            !permissionGranted || !isConnected 
              ? "opacity-50 cursor-not-allowed" 
              : "hover:scale-105 active:scale-95"
          }`}
        >
          {isRecording ? (
            <MicOff className="w-12 h-12 text-white mx-auto" />
          ) : (
            <Mic className="w-12 h-12 text-white mx-auto" />
          )}
          
          {/* Recording indicator */}
          {isRecording && (
            <div className="absolute inset-0 rounded-full border-4 border-red-300 animate-ping"></div>
          )}
        </button>
      </div>

      <div className="text-center mb-4">
        <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
          {isRecording ? "Listening... Speak now!" : "Click the button to start talking"}
        </p>
      </div>

      {/* Transcription display */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 min-h-[200px]">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            Transcription
          </h2>
          <button
            onClick={clearTranscription}
            className="px-4 py-2 text-sm bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg transition-colors"
          >
            Clear
          </button>
        </div>
        
        <div className="space-y-2">
          {/* Final transcription */}
          <div className="text-gray-900 dark:text-white leading-relaxed">
            {transcription || (
              <span className="text-gray-500 dark:text-gray-400 italic">
                Your speech will appear here...
              </span>
            )}
          </div>
          
          {/* Partial transcription */}
          {partialTranscription && (
            <div className="text-gray-600 dark:text-gray-400 italic border-l-2 border-blue-500 pl-3">
              {partialTranscription}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
