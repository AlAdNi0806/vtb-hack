'use client'

import { cn } from '@/lib/utils'
import React, { memo, useCallback, useEffect, useRef } from 'react'
import Avatar from './avatar'
import { Button } from './ui/button';
import { MicIcon, MicOffIcon, PhoneIcon, PhoneOff, PhoneOffIcon } from 'lucide-react';
import useChat from '@/hooks/use-chat';
import { nanoid } from 'nanoid';
import useTranscribe from '@/hooks/use-transcribe';

function AiCall() {

  const { messages, addMessage, clearMessages, editMessage } = useChat();

  const [joinedCall, setJoinedCall] = React.useState(false);

  const [isBotSpeaking, setIsBotSpeaking] = React.useState(false);
  const [isUserSpeaking, setIsUserSpeaking] = React.useState(false);

  const userMessageIdRef = useRef<string | null>(null);
  const botMessageIdRef = useRef<string | null>(null);

  const [isMicOn, setIsMicOn] = React.useState(false);

  function toggleJoinCall() {
    if (!joinedCall === true) {
      addMessage({
        id: nanoid(8),
        role: 'system',
        content: 'You started the call'
      })
      startTranscription()
    } else {
      stopTranscription()
      addMessage({
        id: nanoid(8),
        role: 'system',
        content: 'You left the call'
      })
    }
    setJoinedCall(!joinedCall);
  }

  useEffect(() => {
    addMessage({
      id: nanoid(8),
      role: 'system',
      content: 'You joined the room'
    })
  }, [])

  const { state, finalTokens, nonFinalTokens, startTranscription, stopTranscription } = useTranscribe({
    apiKey: "4f7bdb6be03c50f473bf1bf90f929cf329f108447225452b5b831079581a66de",
    onEndOfSpeech: async (fullText) => {
      userMessageIdRef.current = null
      console.log('End of speech detected. Full text:', fullText);
      if (fullText.trim()) {
        console.log("_________________end_____________________")
        // await handleStreamResponse(fullText);
      }
      stopTranscription()
    },
    onFinished: () => {
      console.log("_________________finished_____________________")
      if (joinedCall) {
        startTranscription();
      }
    },
    onPartialResult: (finalTokens, nonFinalTokens) => {
      const liveTranscript = [...finalTokens, ...nonFinalTokens].map((t) => t.text).join('');
      console.log('Live transcript:', liveTranscript);
      if (liveTranscript.trim()) {
        if (!userMessageIdRef.current) {
          userMessageIdRef.current = nanoid(8)
          addMessage({
            id: userMessageIdRef.current,
            role: 'user',
            content: liveTranscript
          })
        } else {
          editMessage(userMessageIdRef.current, liveTranscript)
        }
      }
    },
  });











  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);

  const stopAudioAnalysis = useCallback(() => {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    setIsUserSpeaking(false);
  }, []);

  const startAudioAnalysis = useCallback(async () => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        streamRef.current = stream;

        const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
        audioContextRef.current = audioContext;

        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        analyserRef.current = analyser;

        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);

        const checkAudio = () => {
          if (!analyserRef.current) return;

          const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
          analyserRef.current.getByteFrequencyData(dataArray);

          let sum = 0;
          for (const amplitude of dataArray) {
            sum += amplitude * amplitude;
          }
          const volume = Math.sqrt(sum / dataArray.length);

          // Adjust this threshold to your needs
          const speakingThreshold = 15;
          setIsUserSpeaking(volume > speakingThreshold);

          animationFrameIdRef.current = requestAnimationFrame(checkAudio);
        };

        checkAudio();
      } catch (err) {
        console.error('Error accessing microphone:', err);
        setIsMicOn(false); // Turn mic off in UI if permission is denied
      }
    }
  }, []);

  useEffect(() => {
    if (isMicOn) {
      startAudioAnalysis();
    } else {
      stopAudioAnalysis();
    }

    // Cleanup on component unmount
    return () => {
      stopAudioAnalysis();
    };
  }, [isMicOn, startAudioAnalysis, stopAudioAnalysis]);


  return (
    <div className="w-full h-full flex overflow-hidden">
      <div className=' bg-stone-900 h-full w-full overflow-hidden p-2 flex flex-col gap-2'>
        <div className='flex gap-4 flex-1'>
          <div
            className={cn(
              'bg-stone-700 h-full w-full overflow-hidden rounded-2xl flex justify-center items-center transition relative',
              isUserSpeaking && 'ring-2 ring-emerald-500'
            )}
          >
            <Avatar
              avatarType='user'
              size='lg'
            />
            <div
              className='absolute bottom-2 w-60 h-30 bg-transparent rounded-lg overflow-hidden'
            >
              <div className='relative w-full h-full flex justify-center items-end overflow-hidden'>
                <div
                  className='absolute w-full h-full z-10 bg-gradient-to-b from-stone-900/80 via-transparent to-transparent'
                />
                <p className='w-full h-full flex flex-col justify-end'>
                  <span className='text-white'>
                    {finalTokens.map((t) => t.text).join('')}
                  </span>
                  <span className='text-stone-400'>
                    {finalTokens.map((t) => t.text).join('')}
                  </span>
                </p>
              </div>
            </div>
          </div>
          <div
            className={cn(
              'bg-stone-700 h-full w-full overflow-hidden rounded-2xl flex justify-center items-center transition relative',
              isBotSpeaking && 'ring-2 ring-emerald-500'
            )}
          >
            <Avatar
              avatarType='bot'
              size='lg'
            />
          </div>
        </div>

        <div className='h-12 rounded-2xl p-3 px-2 bg-stone-700 w-max self-center flex items-center justify-center gap-2'>
          <button
            className='text-white p-3 rounded-full bg-stone-900 hover:bg-stone-800 transition cursor-pointer'
            onClick={() => {
              setIsMicOn(!isMicOn)
            }}
          >
            {isMicOn ? (
              <MicIcon className='h-4 w-4' />
            ) : (
              <MicOffIcon className='h-4 w-4' />
            )}
          </button>
          <button
            className={cn(
              'text-white p-3 rounded-full transition cursor-pointer',
              joinedCall ? 'bg-red-900 hover:bg-red-800' : 'bg-emerald-600 hover:bg-emerald-500'
            )}
            onClick={() => {
              toggleJoinCall()
            }}
          >
            {joinedCall ? (
              <PhoneOffIcon className='h-4 w-4' />
            ) : (
              <PhoneIcon className='h-4 w-4' />
            )}
          </button>
        </div>
      </div>

      <div className='bg-stone-800 h-full w-80 overflow-hidden overflow-y-auto flex flex-col p-3 gap-2'>
        {messages.map((msg) => (
          <ChatMessage key={msg.id} id={msg.id} role={msg.role} content={msg.content} />
        ))}
      </div>
    </div>
  )
}

export default AiCall

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
}

const ChatMessage = memo(({
  id,
  role,
  content
}: ChatMessage) => {

  if (role === 'system') {
    return (
      <div
        key={id}
        className='w-max self-center rounded-full bg-stone-700/80 px-3 py-0.5 text-xs text-stone-300'
      >
        {content}
      </div>
    )
  }

  return (
    <div
      key={id}
      className={cn(
        'w-max max-w-[90%] rounded-lg px-3 py-1 text-xs',
        role === 'user' ? 'bg-amber-900 self-end text-white' : 'bg-transparent self-start text-white'
      )}
    >
      {content}
    </div>
  )
})