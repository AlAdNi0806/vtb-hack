'use client';

import { Button } from '@/components/ui/button';
import useTranscribe from '@/hooks/use-transcribe';
import { isActiveState } from '@/lib/ario';
import { nanoid } from 'nanoid';
import { useState } from 'react';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

// Helper to render a chat bubble
const ChatBubble = ({ message }: { message: ChatMessage }) => {
  const isUser = message.role === 'user';
  const bubbleClasses = isUser
    ? 'bg-blue-500 text-white self-end'
    : 'bg-gray-200 text-gray-800 self-start';
  return (
    <div className={`max-w-md rounded-lg px-4 py-2 ${bubbleClasses}`}>
      <p>{message.content}</p>
    </div>
  );
};

export default function Transcribe() {
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isRecording, setIsRecording] = useState(false);

  const handleStreamResponse = async (text: string) => {
    // Add user message to history
    setChatHistory((prev) => [...prev, { id: nanoid(8), role: 'user', content: text }]);

    // Add a placeholder for the assistant's response
    const assistantMessageId = nanoid(8);
    setChatHistory((prev) => [...prev, { id: assistantMessageId, role: 'assistant', content: '...' }]);

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });

    if (!response.body) return;

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let streamedContent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      // Process Server-Sent Events (SSE) from the stream
      const lines = chunk.split('\n\n').filter(Boolean);
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const dataContent = line.substring(6);
          if (dataContent.trim() === '[DONE]') continue;
          try {
            const json = JSON.parse(dataContent);
            const token = json.choices[0]?.delta?.content || '';
            streamedContent += token;
            // Update the assistant's message in the chat history
            setChatHistory((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId ? { ...msg, content: streamedContent } : msg
              )
            );
          } catch (e) {
            console.error('Failed to parse stream chunk:', dataContent);
          }
        }
      }
    }
  };

  const { state, finalTokens, nonFinalTokens, startTranscription, stopTranscription } = useTranscribe({
    apiKey: "4f7bdb6be03c50f473bf1bf90f929cf329f108447225452b5b831079581a66de",
    onEndOfSpeech: async (fullText) => {
      console.log('End of speech detected. Full text:', fullText);
      if (fullText.trim()) {
        console.log("_________________end_____________________")
        stopTranscription()
        await handleStreamResponse(fullText);
        startTranscription()
      }
    },
  });

  const toggleRecording = () => {
    if (isRecording) {
      stopTranscription();
    } else {
      startTranscription();
    }
    setIsRecording(!isRecording);
  };

  // Combine final and non-final tokens for live display
  const liveTranscript = [...finalTokens, ...nonFinalTokens].map((t) => t.text).join('');

  return (
    <div className="flex h-screen w-full flex-col">
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {chatHistory.map((msg) => (
          <ChatBubble key={msg.id} message={msg} />
        ))}
        {isRecording && liveTranscript && (
          <div className="max-w-md self-end rounded-lg bg-blue-500/80 px-4 py-2 text-white">
            <p>{liveTranscript}</p>
          </div>
        )}
      </div>
      <div className="border-t p-4 text-center">
        <button
          onClick={toggleRecording}
          className={`rounded-full px-6 py-3 font-bold text-white ${isRecording ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'
            }`}
        >
          {isRecording ? 'Stop Listening' : 'Start Listening'}
        </button>
        <p className="mt-2 text-sm text-gray-500">Recorder State: {state}</p>
      </div>
    </div>
  );
}