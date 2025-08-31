'use client'

import { useEffect, useState } from 'react'

interface AudioVisualizerProps {
  audioLevel: number
  isActive: boolean
}

export default function AudioVisualizer({ audioLevel, isActive }: AudioVisualizerProps) {
  const [bars, setBars] = useState<number[]>(new Array(8).fill(0))

  useEffect(() => {
    if (isActive) {
      // Simulate audio bars based on audio level
      const newBars = bars.map((_, index) => {
        const baseHeight = Math.random() * 0.3 + 0.1 // Base random height
        const levelMultiplier = audioLevel * 0.7 // Scale with actual audio level
        return Math.min(baseHeight + levelMultiplier, 1) // Cap at 1
      })
      setBars(newBars)
    } else {
      setBars(new Array(8).fill(0))
    }
  }, [audioLevel, isActive])

  return (
    <div className="audio-visualizer">
      {bars.map((height, index) => (
        <div
          key={index}
          className={`audio-bar ${isActive ? 'active' : ''}`}
          style={{
            height: `${Math.max(height * 32, 4)}px`,
            animationDelay: `${index * 0.1}s`
          }}
        />
      ))}
    </div>
  )
}
