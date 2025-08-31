"""
Comprehensive Audio Processing Pipeline
Orchestrates STT, turn detection, LLM response, and TTS
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from services.audio_processor import AudioProcessor
from services.ai_service import AIService
from utils.logger import setup_logger
from config import settings

logger = setup_logger(__name__)


class ConversationState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"


@dataclass
class AudioChunk:
    data: str
    timestamp: float
    sequence_id: int


@dataclass
class ProcessingResult:
    transcript: str
    is_final: bool
    turn_detected: bool
    confidence: float
    ai_response: Optional[str] = None
    tts_audio: Optional[str] = None


class AudioPipeline:
    """Orchestrates the complete audio processing pipeline"""
    
    def __init__(self, audio_processor: AudioProcessor, ai_service: AIService):
        self.audio_processor = audio_processor
        self.ai_service = ai_service
        self.state = ConversationState.IDLE
        
        # Pipeline state
        self.current_session_id: Optional[str] = None
        self.audio_buffer: List[AudioChunk] = []
        self.accumulated_transcript = ""
        self.processing_lock = asyncio.Lock()
        
        # Configuration
        self.max_buffer_size = 100  # Maximum audio chunks to keep
        self.transcript_timeout = 5.0  # Seconds to wait for transcript completion
        
    async def start_session(self, session_id: str) -> Dict[str, Any]:
        """Start a new conversation session"""
        async with self.processing_lock:
            logger.info(f"Starting audio pipeline session: {session_id}")
            
            self.current_session_id = session_id
            self.state = ConversationState.LISTENING
            self.audio_buffer.clear()
            self.accumulated_transcript = ""
            
            # Clear AI conversation history for new session
            self.ai_service.clear_conversation_history()
            
            return {
                "status": "session_started",
                "session_id": session_id,
                "state": self.state.value
            }
    
    async def end_session(self) -> Dict[str, Any]:
        """End the current conversation session"""
        async with self.processing_lock:
            logger.info(f"Ending audio pipeline session: {self.current_session_id}")
            
            session_id = self.current_session_id
            self.current_session_id = None
            self.state = ConversationState.IDLE
            self.audio_buffer.clear()
            self.accumulated_transcript = ""
            
            return {
                "status": "session_ended",
                "session_id": session_id,
                "state": self.state.value
            }
    
    async def process_audio_chunk(self, audio_data: str, timestamp: float) -> Dict[str, Any]:
        """Process incoming audio chunk through the complete pipeline"""
        if self.state == ConversationState.IDLE:
            return {"error": "No active session"}
        
        # Create audio chunk
        chunk = AudioChunk(
            data=audio_data,
            timestamp=timestamp,
            sequence_id=len(self.audio_buffer)
        )
        
        # Add to buffer (with size limit)
        self.audio_buffer.append(chunk)
        if len(self.audio_buffer) > self.max_buffer_size:
            self.audio_buffer.pop(0)
        
        try:
            # Process through audio processor
            audio_result = await self.audio_processor.process_audio_chunk(audio_data)
            
            result = {
                "transcript": audio_result.get("transcript", ""),
                "is_final": audio_result.get("is_final", False),
                "turn_detected": audio_result.get("turn_detected", False),
                "is_voice": audio_result.get("is_voice", False),
                "state": self.state.value,
                "session_id": self.current_session_id
            }
            
            # Update accumulated transcript
            if audio_result.get("transcript"):
                if audio_result.get("is_final"):
                    self.accumulated_transcript = audio_result["transcript"]
                else:
                    # For partial transcripts, we might want to show them but not accumulate
                    pass
            
            # Handle turn detection and AI response generation
            if audio_result.get("turn_detected") and self.accumulated_transcript:
                ai_response_result = await self._generate_ai_response(self.accumulated_transcript)
                result.update(ai_response_result)
                
                # Reset for next turn
                self.accumulated_transcript = ""
            
            return result
            
        except Exception as e:
            logger.error(f"Error in audio pipeline processing: {e}")
            return {
                "error": f"Processing error: {str(e)}",
                "state": self.state.value,
                "session_id": self.current_session_id
            }
    
    async def _generate_ai_response(self, user_input: str) -> Dict[str, Any]:
        """Generate AI response and TTS audio"""
        try:
            # Update state
            self.state = ConversationState.PROCESSING
            
            logger.info(f"Generating AI response for: {user_input[:100]}...")
            
            # Generate AI response
            ai_response = await self.ai_service.generate_response(user_input)

            # Update state
            self.state = ConversationState.RESPONDING

            # Generate TTS audio with streaming if enabled
            streaming_enabled = getattr(settings, 'ENABLE_STREAMING_TTS', True)
            tts_audio = await self.ai_service.text_to_speech(ai_response, streaming=streaming_enabled)

            # Return to listening state
            self.state = ConversationState.LISTENING

            return {
                "ai_response": ai_response,
                "tts_audio": tts_audio,
                "processing_complete": True,
                "streaming_used": streaming_enabled
            }
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            self.state = ConversationState.LISTENING  # Reset state
            return {
                "ai_response": "I'm sorry, I'm having trouble processing your request right now.",
                "tts_audio": "",
                "processing_complete": False,
                "error": str(e)
            }
    
    async def get_session_status(self) -> Dict[str, Any]:
        """Get current session status"""
        return {
            "session_id": self.current_session_id,
            "state": self.state.value,
            "buffer_size": len(self.audio_buffer),
            "accumulated_transcript": self.accumulated_transcript,
            "is_active": self.current_session_id is not None
        }
    
    async def force_process_transcript(self) -> Dict[str, Any]:
        """Force process current accumulated transcript (for testing/debugging)"""
        if not self.accumulated_transcript:
            return {"error": "No accumulated transcript to process"}
        
        try:
            result = await self._generate_ai_response(self.accumulated_transcript)
            self.accumulated_transcript = ""
            return result
        except Exception as e:
            logger.error(f"Error in forced transcript processing: {e}")
            return {"error": str(e)}
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        return {
            "current_state": self.state.value,
            "session_active": self.current_session_id is not None,
            "buffer_size": len(self.audio_buffer),
            "accumulated_transcript_length": len(self.accumulated_transcript),
            "audio_processor_ready": self.audio_processor.is_ready(),
            "ai_service_ready": self.ai_service.is_ready()
        }
