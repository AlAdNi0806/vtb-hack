"use client";

import { SpeechToTextApp } from "@/components/SpeechToTextApp";

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
            Real-time Speech to Text
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Press the Talk button and start speaking to see live transcription
          </p>
        </div>
        <SpeechToTextApp />
      </div>
    </div>
  );
}
