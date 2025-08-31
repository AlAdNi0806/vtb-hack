"""
Audio Processing Service for STT and Turn Detection
Enhanced with Parakeet-1.1b and RealtimeSTT for streaming recognition
"""

import asyncio
import base64
import io
import logging
import numpy as np
import torch
import torchaudio
import webrtcvad
from typing import Optional, Dict, Any, List, Callable
import librosa
import soundfile as sf
from RealtimeSTT import AudioToTextRecorder

# Try to import NeMo, but make it optional
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"NeMo not available: {e}")
    nemo_asr = None
    NEMO_AVAILABLE = False

from config import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioProcessor:
    """Service for audio processing including STT and turn detection"""

    def __init__(self):
        self.parakeet_model = None
        self.realtime_stt = None
        self.turn_detector = None
        self.vad = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ready = False

        # Audio buffers and state
        self.audio_buffer = []
        self.silence_buffer = []
        self.is_speaking = False
        self.silence_duration = 0.0
        self.last_transcript = ""

        # Streaming state
        self.streaming_active = False
        self.partial_transcript = ""
        self.transcript_callback = None
    
    async def initialize(self):
        """Initialize audio processing models"""
        try:
            logger.info("Initializing Audio Processor with Parakeet and RealtimeSTT...")

            # Initialize Voice Activity Detection
            self.vad = webrtcvad.Vad(settings.VAD_AGGRESSIVENESS)

            # Initialize Parakeet STT model
            await self._initialize_parakeet_model()

            # Initialize RealtimeSTT for streaming
            await self._initialize_realtime_stt()

            # Initialize turn detector
            await self._initialize_turn_detector()

            self._ready = True
            logger.info("Audio Processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Audio Processor: {e}")
            raise
    
    async def _initialize_parakeet_model(self):
        """Initialize Parakeet-1.1b RNNT multilingual ASR model"""
        try:
            if not NEMO_AVAILABLE:
                logger.warning("NeMo not available, skipping Parakeet model...")
                await self._initialize_whisper_fallback()
                return

            logger.info("Loading Parakeet-1.1b RNNT multilingual ASR model...")

            # Load Parakeet model from NVIDIA NeMo
            model_name = settings.STT_MODEL_PATH  # nvidia/parakeet-1.1b-rnnt-multilingual-asr

            self.parakeet_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name=model_name,
                map_location=self.device
            )

            # Set model to evaluation mode
            self.parakeet_model.eval()

            # Configure for streaming
            self.parakeet_model.change_decoding_strategy("greedy")

            logger.info("Parakeet-1.1b model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Parakeet model: {e}")
            # Fallback to Whisper if Parakeet fails
            logger.warning("Falling back to Whisper model...")
            await self._initialize_whisper_fallback()

    async def _initialize_whisper_fallback(self):
        """Initialize Whisper as fallback STT model"""
        try:
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            model_name = "openai/whisper-base"
            self.stt_processor = AutoProcessor.from_pretrained(model_name)
            self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            )
            self.stt_model.to(self.device)
            logger.info("Whisper fallback model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper fallback: {e}")
            raise

    async def _initialize_realtime_stt(self):
        """Initialize RealtimeSTT for streaming speech recognition"""
        try:
            logger.info("Initializing RealtimeSTT for streaming...")

            # Configure RealtimeSTT
            self.realtime_stt = AudioToTextRecorder(
                model="base",  # Can be changed to larger models for better accuracy
                language="multilingual",  # Support multiple languages
                spinner=False,
                use_microphone=False,  # We'll feed audio manually
                level=logging.WARNING,  # Reduce RealtimeSTT logging
                wake_words_sensitivity=0.5,
                post_speech_silence_duration=0.3,
                min_length_of_recording=0.5,
                min_gap_between_recordings=0.1,
                enable_realtime_transcription=True,
                realtime_processing_pause=0.02,
                silero_sensitivity=0.05,
                webrtc_sensitivity=3
            )

            # Set up callbacks for streaming transcription
            self.realtime_stt.set_microphone(False)

            logger.info("RealtimeSTT initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RealtimeSTT: {e}")
            logger.warning("Continuing without RealtimeSTT streaming capabilities")
            self.realtime_stt = None
    
    async def _initialize_turn_detector(self):
        """Initialize turn detection (simplified implementation)"""
        try:
            logger.info("Initializing turn detector...")
            
            # For now, we'll use a simple silence-based turn detection
            # You can enhance this with the LiveKit turn detector model
            self.turn_detector = SimpleTurnDetector(
                silence_threshold=settings.SILENCE_THRESHOLD,
                sample_rate=settings.SAMPLE_RATE
            )
            
            logger.info("Turn detector initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize turn detector: {e}")
            raise
    
    async def process_audio_chunk(self, audio_data_base64: str) -> Dict[str, Any]:
        """Process incoming audio chunk with streaming capabilities"""
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data_base64)

            # Convert to numpy array
            audio_array = self._bytes_to_audio_array(audio_bytes)

            # Add to audio buffer
            self.audio_buffer.extend(audio_array)

            # Voice Activity Detection
            is_voice = self._detect_voice_activity(audio_array)

            # Update speaking state and detect turns
            turn_detected = self.turn_detector.process_audio(audio_array, is_voice)

            result = {
                "transcript": "",
                "is_final": False,
                "turn_detected": turn_detected,
                "is_voice": is_voice,
                "partial_transcript": ""
            }

            # Process with streaming STT if available
            if self.realtime_stt and is_voice:
                streaming_result = await self._process_streaming_stt(audio_array)
                result.update(streaming_result)

            # Fallback to batch processing for final transcripts
            if turn_detected or (len(self.audio_buffer) >= settings.SAMPLE_RATE * 1.0):
                transcript = await self._transcribe_audio_batch()
                if transcript:
                    result["transcript"] = transcript
                    result["is_final"] = turn_detected

            # If turn is detected, finalize transcript and clear buffer
            if turn_detected:
                if self.audio_buffer:
                    final_transcript = await self._transcribe_audio_batch()
                    if final_transcript:
                        result["transcript"] = final_transcript
                        result["is_final"] = True

                self._clear_audio_buffer()

            return result

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return {"transcript": "", "is_final": False, "turn_detected": False, "is_voice": False}
    
    def _bytes_to_audio_array(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array"""
        try:
            # Load audio from bytes
            audio_io = io.BytesIO(audio_bytes)
            audio_array, sample_rate = sf.read(audio_io)
            
            # Resample if necessary
            if sample_rate != settings.SAMPLE_RATE:
                audio_array = librosa.resample(
                    audio_array, 
                    orig_sr=sample_rate, 
                    target_sr=settings.SAMPLE_RATE
                )
            
            # Ensure mono
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            return audio_array.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error converting audio bytes: {e}")
            return np.array([], dtype=np.float32)
    
    def _detect_voice_activity(self, audio_array: np.ndarray) -> bool:
        """Detect voice activity in audio chunk"""
        try:
            # Convert to 16-bit PCM for WebRTC VAD
            audio_int16 = (audio_array * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # WebRTC VAD expects specific frame sizes
            frame_duration = 30  # ms
            frame_size = int(settings.SAMPLE_RATE * frame_duration / 1000)
            
            # Process in frames
            voice_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_bytes), frame_size * 2):  # 2 bytes per sample
                frame = audio_bytes[i:i + frame_size * 2]
                if len(frame) == frame_size * 2:
                    is_voice = self.vad.is_speech(frame, settings.SAMPLE_RATE)
                    if is_voice:
                        voice_frames += 1
                    total_frames += 1
            
            # Return True if majority of frames contain voice
            return voice_frames > total_frames * 0.3 if total_frames > 0 else False
            
        except Exception as e:
            logger.error(f"Error in voice activity detection: {e}")
            return False

    async def _process_streaming_stt(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """Process audio with RealtimeSTT for streaming transcription"""
        try:
            if not self.realtime_stt:
                return {"partial_transcript": ""}

            # Convert audio to the format expected by RealtimeSTT
            audio_int16 = (audio_array * 32767).astype(np.int16)

            # Feed audio to RealtimeSTT
            # Note: RealtimeSTT typically works with microphone input
            # For chunk-based processing, we need to simulate this
            partial_text = ""

            # This is a simplified implementation
            # In practice, you might need to accumulate chunks and process them
            if len(self.audio_buffer) > settings.SAMPLE_RATE * 0.5:  # 0.5 seconds
                try:
                    # Convert buffer to format for transcription
                    buffer_array = np.array(self.audio_buffer, dtype=np.float32)

                    # Use RealtimeSTT for partial transcription
                    # This is a conceptual implementation - actual RealtimeSTT integration
                    # may require different approach based on their API
                    partial_text = self._get_partial_transcript(buffer_array)

                except Exception as e:
                    logger.debug(f"Streaming STT processing error: {e}")

            return {
                "partial_transcript": partial_text,
                "streaming_active": True
            }

        except Exception as e:
            logger.error(f"Error in streaming STT: {e}")
            return {"partial_transcript": ""}

    def _get_partial_transcript(self, audio_array: np.ndarray) -> str:
        """Get partial transcript from audio buffer"""
        try:
            # This is a placeholder for RealtimeSTT integration
            # The actual implementation would depend on RealtimeSTT's API
            # For now, we'll return empty string and rely on batch processing
            return ""

        except Exception as e:
            logger.debug(f"Partial transcript error: {e}")
            return ""
    
    async def _transcribe_audio_batch(self) -> str:
        """Transcribe audio buffer to text using Parakeet or fallback model"""
        try:
            if not self.audio_buffer:
                return ""

            # Convert buffer to tensor
            audio_tensor = torch.tensor(self.audio_buffer, dtype=torch.float32)

            # Ensure minimum length
            if len(audio_tensor) < settings.SAMPLE_RATE * 0.1:  # 0.1 seconds minimum
                return ""

            transcription = ""

            # Try Parakeet first
            if self.parakeet_model:
                transcription = await self._transcribe_with_parakeet(audio_tensor)

            # Fallback to Whisper if Parakeet fails or unavailable
            if not transcription and hasattr(self, 'stt_model'):
                transcription = await self._transcribe_with_whisper(audio_tensor)

            # Clean up transcription
            transcription = transcription.strip()

            if transcription and transcription != self.last_transcript:
                self.last_transcript = transcription
                logger.info(f"Transcribed: {transcription}")
                return transcription

            return ""

        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return ""

    async def _transcribe_with_parakeet(self, audio_tensor: torch.Tensor) -> str:
        """Transcribe using Parakeet-1.1b model"""
        try:
            # Prepare audio for Parakeet
            audio_signal = audio_tensor.unsqueeze(0).to(self.device)
            audio_length = torch.tensor([audio_tensor.shape[0]], dtype=torch.long).to(self.device)

            # Transcribe with Parakeet
            with torch.no_grad():
                hypotheses = self.parakeet_model.transcribe(
                    audio=audio_signal,
                    audio_length=audio_length,
                    return_hypotheses=True
                )

            if hypotheses and len(hypotheses[0]) > 0:
                transcription = hypotheses[0][0].text
                logger.debug(f"Parakeet transcription: {transcription}")
                return transcription

            return ""

        except Exception as e:
            logger.error(f"Error in Parakeet transcription: {e}")
            return ""

    async def _transcribe_with_whisper(self, audio_tensor: torch.Tensor) -> str:
        """Transcribe using Whisper fallback model"""
        try:
            if not hasattr(self, 'stt_processor') or not hasattr(self, 'stt_model'):
                return ""

            # Process with Whisper
            inputs = self.stt_processor(
                audio_tensor,
                sampling_rate=settings.SAMPLE_RATE,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.stt_model.generate(**inputs)

            # Decode transcription
            transcription = self.stt_processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True
            )[0]

            logger.debug(f"Whisper transcription: {transcription}")
            return transcription

        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            return ""
    
    def _clear_audio_buffer(self):
        """Clear audio buffer"""
        self.audio_buffer = []
        self.last_transcript = ""
    
    def is_ready(self) -> bool:
        """Check if service is ready"""
        return self._ready
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Audio Processor...")
        
        # Clear buffers
        self.audio_buffer = []
        self.silence_buffer = []
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._ready = False
        logger.info("Audio Processor cleanup completed")


class SimpleTurnDetector:
    """Simple turn detection based on silence duration"""
    
    def __init__(self, silence_threshold: float, sample_rate: int):
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        self.silence_duration = 0.0
        self.was_speaking = False
    
    def process_audio(self, audio_array: np.ndarray, is_voice: bool) -> bool:
        """Process audio and detect turn completion"""
        chunk_duration = len(audio_array) / self.sample_rate
        
        if is_voice:
            self.silence_duration = 0.0
            self.was_speaking = True
            return False
        else:
            if self.was_speaking:
                self.silence_duration += chunk_duration
                
                if self.silence_duration >= self.silence_threshold:
                    self.was_speaking = False
                    self.silence_duration = 0.0
                    return True
        
        return False
