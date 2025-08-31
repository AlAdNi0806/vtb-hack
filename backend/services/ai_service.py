"""
AI Service for handling LLM interactions and TTS generation
Enhanced with RealtimeTTS for streaming speech synthesis
"""

import asyncio
import base64
import io
import logging
from typing import Optional, Dict, Any, Callable
import httpx
import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AIService:
    """Service for AI model interactions including LLM and TTS"""

    def __init__(self):
        self.cerebras_client = None
        self.realtime_tts = None
        self.tts_engine = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ready = False

        # Streaming TTS state
        self.streaming_active = False
        self.audio_callback = None

        # Conversation context
        self.conversation_history = []
        self.system_prompt = """You are a helpful AI assistant engaged in a natural voice conversation.
        Keep your responses conversational, concise, and engaging. Respond as if you're talking to someone in person.
        Avoid overly long responses since this is a real-time conversation."""
    
    async def initialize(self):
        """Initialize AI models and services"""
        try:
            logger.info("Initializing AI Service with RealtimeTTS...")

            # Initialize Cerebras HTTP client
            self.cerebras_client = httpx.AsyncClient(
                base_url=settings.CEREBRAS_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.CEREBRAS_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )

            # Test Cerebras connection
            await self._test_cerebras_connection()

            # Initialize RealtimeTTS
            await self._initialize_realtime_tts()

            self._ready = True
            logger.info("AI Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize AI Service: {e}")
            raise
    
    async def _test_cerebras_connection(self):
        """Test connection to Cerebras API"""
        try:
            response = await self.cerebras_client.post(
                "/chat/completions",
                json={
                    "model": settings.CEREBRAS_MODEL,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                }
            )
            response.raise_for_status()
            logger.info("Cerebras API connection successful")
            
        except Exception as e:
            logger.error(f"Cerebras API connection failed: {e}")
            raise
    
    async def _initialize_realtime_tts(self):
        """Initialize RealtimeTTS for streaming speech synthesis"""
        try:
            logger.info("Initializing RealtimeTTS...")

            # Initialize TTS engine (SystemEngine for cross-platform compatibility)
            self.tts_engine = SystemEngine()

            # Initialize RealtimeTTS stream
            self.realtime_tts = TextToAudioStream(
                engine=self.tts_engine,
                on_audio_stream_start=self._on_audio_stream_start,
                on_audio_stream_stop=self._on_audio_stream_stop,
                on_audio_chunk=self._on_audio_chunk,
                level=logging.WARNING  # Reduce RealtimeTTS logging
            )

            logger.info("RealtimeTTS initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RealtimeTTS: {e}")
            logger.warning("Falling back to basic TTS...")
            await self._initialize_fallback_tts()

    async def _initialize_fallback_tts(self):
        """Initialize fallback TTS using Silero"""
        try:
            logger.info("Loading Silero TTS as fallback...")

            # Load Silero TTS model
            import silero
            self.tts_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='en',
                speaker='v3_en'
            )

            self.tts_model.to(self.device)
            self.tts_speaker = 'en_0'  # Default English speaker

            logger.info("Silero TTS fallback loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load fallback TTS: {e}")
            raise

    def _on_audio_stream_start(self):
        """Callback when audio stream starts"""
        self.streaming_active = True
        logger.debug("Audio stream started")

    def _on_audio_stream_stop(self):
        """Callback when audio stream stops"""
        self.streaming_active = False
        logger.debug("Audio stream stopped")

    def _on_audio_chunk(self, chunk):
        """Callback for each audio chunk during streaming"""
        if self.audio_callback:
            self.audio_callback(chunk)
    
    async def generate_response(self, user_input: str) -> str:
        """Generate AI response using Cerebras LLM"""
        try:
            # Add user input to conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # Prepare messages for Cerebras API
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add recent conversation history (keep last 10 exchanges)
            recent_history = self.conversation_history[-20:]  # Last 10 user-assistant pairs
            messages.extend(recent_history)
            
            # Call Cerebras API
            response = await self.cerebras_client.post(
                "/chat/completions",
                json={
                    "model": settings.CEREBRAS_MODEL,
                    "messages": messages,
                    "max_tokens": 150,  # Keep responses concise for voice conversation
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract AI response
            ai_response = result["choices"][0]["message"]["content"].strip()
            
            # Add AI response to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            logger.info(f"Generated AI response: {ai_response[:100]}...")
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return "I'm sorry, I'm having trouble processing your request right now."
    
    async def text_to_speech(self, text: str, streaming: bool = True) -> str:
        """Convert text to speech using RealtimeTTS or fallback"""
        try:
            logger.info(f"Converting text to speech: {text[:50]}...")

            if self.realtime_tts and streaming:
                return await self._text_to_speech_streaming(text)
            else:
                return await self._text_to_speech_batch(text)

        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {e}")
            return ""

    async def _text_to_speech_streaming(self, text: str) -> str:
        """Convert text to speech using RealtimeTTS streaming"""
        try:
            logger.info("Using RealtimeTTS for streaming synthesis...")

            # Collect audio chunks
            audio_chunks = []

            def collect_chunk(chunk):
                audio_chunks.append(chunk)

            # Set callback to collect chunks
            self.audio_callback = collect_chunk

            # Generate audio with RealtimeTTS
            self.realtime_tts.feed(text)
            self.realtime_tts.play()

            # Wait for synthesis to complete
            while self.streaming_active:
                await asyncio.sleep(0.01)

            # Reset callback
            self.audio_callback = None

            # Combine audio chunks and convert to base64
            if audio_chunks:
                combined_audio = b''.join(audio_chunks)
                audio_base64 = base64.b64encode(combined_audio).decode('utf-8')
                logger.info("Streaming TTS conversion completed")
                return audio_base64

            return ""

        except Exception as e:
            logger.error(f"Error in streaming TTS: {e}")
            # Fallback to batch processing
            return await self._text_to_speech_batch(text)

    async def _text_to_speech_batch(self, text: str) -> str:
        """Convert text to speech using fallback TTS"""
        try:
            if not hasattr(self, 'tts_model') or not self.tts_model:
                logger.warning("No TTS model available")
                return ""

            logger.info("Using fallback TTS for batch synthesis...")

            # Generate audio using Silero TTS
            audio = self.tts_model.apply_tts(
                text=text,
                speaker=self.tts_speaker,
                sample_rate=settings.SAMPLE_RATE
            )

            # Convert tensor to numpy array
            audio_np = audio.cpu().numpy()

            # Create audio buffer
            audio_buffer = io.BytesIO()

            # Save as WAV format
            torchaudio.save(
                audio_buffer,
                torch.from_numpy(audio_np).unsqueeze(0),
                settings.SAMPLE_RATE,
                format="wav"
            )

            # Get audio bytes and encode as base64
            audio_bytes = audio_buffer.getvalue()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            logger.info("Batch TTS conversion completed")
            return audio_base64

        except Exception as e:
            logger.error(f"Error in batch TTS conversion: {e}")
            return ""
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._ready
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up AI Service...")
        
        if self.cerebras_client:
            await self.cerebras_client.aclose()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._ready = False
        logger.info("AI Service cleanup completed")
