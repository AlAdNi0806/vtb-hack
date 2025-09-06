'use client'

import { cn } from '@/lib/utils';
import { BotIcon, User2Icon } from 'lucide-react';
import React from 'react'

interface AvatarProps {
    avatarType?: 'user' | 'bot';
    size?: 'sm' | 'md' | 'lg';
}

function Avatar({ avatarType = 'user', size = 'md' }: AvatarProps): React.JSX.Element {
  return (
    <div 
        className={cn(
            'rounded-full bg-stone-600 flex justify-center items-center',
            size === 'sm' && 'h-10 w-10',
            size === 'md' && 'h-16 w-16',
            size === 'lg' && 'h-24 w-24',
        )}  
    >
        {avatarType === 'user' ? (
            <User2Icon className="text-white" />
        ) : (
            <BotIcon className="text-white" />
        )}
    </div>
  )
}

export default Avatar