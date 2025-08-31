'use client'

import { Mic, MessageCircle } from 'lucide-react'

export default function Header() {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
              <MessageCircle className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-gray-900">Conversational AI</h1>
              <p className="text-sm text-gray-600">Real-time voice conversation</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <Mic className="w-5 h-5 text-gray-400" />
            <span className="text-sm text-gray-600">Voice Enabled</span>
          </div>
        </div>
      </div>
    </header>
  )
}
