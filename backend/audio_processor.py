import asyncio
import logging
import threading
import queue
import numpy as np
import tempfile
import os
import soundfile as sf
from typing import Callable, Optional
from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, on_transcription: Callable[[str, bool], None]):
        self.on_transcription = on_transcription
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.processing_thread = None

        # Audio configuration
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # 500ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)

        # Audio buffer for accumulating chunks
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()

        # Model configuration
        self.parakeet_model = None
        self.fallback_recorder = None
        self.use_parakeet = True

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize Parakeet model with RealtimeSTT fallback"""
        # First try to initialize Parakeet model
        self._initialize_parakeet()

        # Always initialize fallback RealtimeSTT model
        self._initialize_fallback()

    def _initialize_parakeet(self):
        """Initialize the Parakeet multilingual ASR model with CPU fallback"""
        try:
            logger.info("Loading parakeet-1.1b-rnnt-multilingual-asr model...")

            import nemo.collections.asr as nemo_asr
            import torch

            # Force CPU usage to avoid CUDA memory issues
            device = "cpu"
            logger.info(f"Using device: {device} for Parakeet model")

            # Clear any existing CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load the parakeet model using the correct model name
            self.parakeet_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                model_name="nvidia/parakeet-rnnt-1.1b",
                map_location=device  # Force CPU
            )

            # Move model to CPU explicitly
            self.parakeet_model = self.parakeet_model.to(device)

            # Set to evaluation mode
            self.parakeet_model.eval()

            # Disable gradients to save memory
            for param in self.parakeet_model.parameters():
                param.requires_grad = False

            logger.info("✅ Parakeet model loaded successfully on CPU!")
            self.use_parakeet = True

        except Exception as e:
            logger.error(f"❌ Failed to load Parakeet model: {e}")
            logger.info("Will use RealtimeSTT fallback model")
            self.parakeet_model = None
            self.use_parakeet = False

    def _initialize_fallback(self):
        """Initialize RealtimeSTT as fallback"""
        try:
            logger.info("Initializing RealtimeSTT fallback model...")

            self.fallback_recorder = AudioToTextRecorder(
                model="tiny",  # Small, reliable model
                language="en",
                spinner=False,
                use_microphone=False,
                level=logging.ERROR,
                device="cpu",
                compute_type="int8"
            )

            logger.info("✅ RealtimeSTT fallback model initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize fallback model: {e}")
            self.fallback_recorder = None
    
    def start(self):
        """Start audio processing"""
        if not self.is_recording and (self.parakeet_model or self.fallback_recorder):
            self.is_recording = True
            self.audio_buffer = []

            # Clear any existing queue items
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Start the processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            model_info = "Parakeet" if self.use_parakeet and self.parakeet_model else "RealtimeSTT"
            logger.info(f"Audio processing started using {model_info} model")
        else:
            logger.error("Cannot start - no models available")

    def stop(self):
        """Stop audio processing"""
        if self.is_recording:
            self.is_recording = False

            # Process any remaining audio in buffer
            self._process_final_buffer()

            if self.processing_thread:
                self.processing_thread.join(timeout=3.0)
                if self.processing_thread.is_alive():
                    logger.warning("Processing thread did not stop gracefully")

            # Clear the queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            logger.info("Audio processing stopped")
    
    def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk"""
        if not self.is_recording:
            return
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            with self.buffer_lock:
                self.audio_buffer.extend(audio_np)
                
                # If we have enough data for processing, add to queue
                if len(self.audio_buffer) >= self.chunk_size:
                    chunk = np.array(self.audio_buffer[:self.chunk_size])
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                    self.audio_queue.put(chunk)
                    
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def _process_audio_loop(self):
        """Main audio processing loop with improved error handling"""
        logger.info("Audio processing loop started")
        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.is_recording:
            try:
                # Get audio chunk from queue with timeout
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                # Process the audio chunk
                if self.recorder and len(audio_chunk) > 0:
                    text = self._transcribe_chunk(audio_chunk)

                    if text and text.strip():
                        # Send partial transcription
                        self.on_transcription(text.strip(), False)
                        consecutive_errors = 0  # Reset error counter on success

            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error in audio processing loop: {e}")

                # If too many consecutive errors, stop processing
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Too many consecutive errors ({consecutive_errors}), stopping audio processing")
                    self.is_recording = False
                    break

                # Brief pause before retrying
                import time
                time.sleep(0.1)

        logger.info("Audio processing loop ended")

    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a single audio chunk using Parakeet or fallback"""
        try:
            # Ensure audio chunk is valid
            if len(audio_chunk) == 0:
                return ""

            # Limit chunk size to prevent memory issues
            max_chunk_size = self.sample_rate * 2  # 2 seconds max
            if len(audio_chunk) > max_chunk_size:
                audio_chunk = audio_chunk[:max_chunk_size]

            # Try Parakeet model first if available
            if self.use_parakeet and self.parakeet_model:
                try:
                    text = self._transcribe_with_parakeet(audio_chunk)
                    if text and text.strip():
                        return text.strip()
                except Exception as e:
                    logger.warning(f"Parakeet transcription failed, using fallback: {e}")
                    # Don't disable parakeet permanently, just use fallback for this chunk

            # Use RealtimeSTT fallback
            if self.fallback_recorder:
                text = self.fallback_recorder.transcribe(audio_chunk)
                return text.strip() if text else ""

            return ""

        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""

    def _transcribe_with_parakeet(self, audio_chunk: np.ndarray) -> str:
        """Transcribe using Parakeet model"""
        try:
            # Ensure audio is in the right format (16kHz, mono)
            if len(audio_chunk) < self.sample_rate * 0.1:  # Less than 100ms
                return ""

            # Create a temporary WAV file for Parakeet
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

                # Write audio to temporary file
                sf.write(temp_path, audio_chunk, self.sample_rate)

                # Transcribe using Parakeet
                transcription = self.parakeet_model.transcribe([temp_path])

                # Clean up temporary file
                os.unlink(temp_path)

                # Return the transcription text
                if transcription and len(transcription) > 0:
                    return transcription[0]
                return ""

        except Exception as e:
            logger.error(f"Error in Parakeet transcription: {e}")
            return ""
    
    def _process_final_buffer(self):
        """Process any remaining audio in the buffer"""
        try:
            with self.buffer_lock:
                if self.audio_buffer and len(self.audio_buffer) > 0:
                    # Only process if we have significant audio data
                    if len(self.audio_buffer) > self.sample_rate * 0.1:  # At least 100ms
                        final_chunk = np.array(self.audio_buffer, dtype=np.float32)
                        self.audio_buffer = []

                        # Transcribe final chunk
                        text = self._transcribe_chunk(final_chunk)
                        if text and text.strip():
                            # Send final transcription
                            self.on_transcription(text.strip(), True)
                    else:
                        # Clear small buffer without processing
                        self.audio_buffer = []

        except Exception as e:
            logger.error(f"Error processing final buffer: {e}")
            # Ensure buffer is cleared even on error
            with self.buffer_lock:
                self.audio_buffer = []

    def cleanup(self):
        """Clean up resources to prevent memory leaks"""
        try:
            logger.info("Cleaning up audio processor...")

            # Stop recording if active
            if self.is_recording:
                self.stop()

            # Clear buffers
            with self.buffer_lock:
                self.audio_buffer = []

            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break

            # Clean up Parakeet model
            if self.parakeet_model:
                try:
                    # Clear CUDA cache if using GPU
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Error cleaning up Parakeet model: {e}")
                finally:
                    self.parakeet_model = None

            # Clean up fallback recorder
            if self.fallback_recorder:
                try:
                    if hasattr(self.fallback_recorder, 'cleanup'):
                        self.fallback_recorder.cleanup()
                except Exception as e:
                    logger.warning(f"Error cleaning up fallback recorder: {e}")
                finally:
                    self.fallback_recorder = None

            logger.info("Audio processor cleanup completed")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
