import asyncio
import logging
import threading
import queue
import numpy as np
from typing import Callable, Optional
import torch
import torchaudio
from RealtimeSTT import AudioToTextRecorder
import nemo.collections.asr as nemo_asr

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, on_transcription: Callable[[str, bool], None]):
        self.on_transcription = on_transcription
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.processing_thread = None
        
        # Initialize the parakeet model
        self.model = None
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # 500ms chunks
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Audio buffer for accumulating chunks
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Initialize RealtimeSTT recorder
        self.recorder = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the parakeet ASR model and RealtimeSTT recorder"""
        try:
            logger.info("Loading parakeet-1.1b-rnnt-multilingual-asr model...")
            
            # Load the parakeet model from NVIDIA NeMo
            self.model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
                "nvidia/parakeet-1.1b-rnnt-multilingual-asr"
            )
            self.model.eval()
            
            # Set up RealtimeSTT recorder with custom model
            self.recorder = AudioToTextRecorder(
                model="custom",
                language="multilingual",
                spinner=False,
                use_microphone=False,  # We'll feed audio manually
                level=logging.WARNING
            )
            
            # Override the transcription method to use our parakeet model
            self.recorder.transcribe = self._transcribe_with_parakeet
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            # Fallback to default RealtimeSTT model
            try:
                self.recorder = AudioToTextRecorder(
                    model="base",
                    language="en",
                    spinner=False,
                    use_microphone=False,
                    level=logging.WARNING
                )
                logger.info("Fallback to default RealtimeSTT model")
            except Exception as fallback_error:
                logger.error(f"Fallback model initialization failed: {fallback_error}")
    
    def _transcribe_with_parakeet(self, audio_data: np.ndarray) -> str:
        """Custom transcription method using parakeet model"""
        try:
            if self.model is None:
                return ""
            
            # Convert numpy array to torch tensor
            if isinstance(audio_data, np.ndarray):
                audio_tensor = torch.from_numpy(audio_data).float()
            else:
                audio_tensor = audio_data
            
            # Ensure correct shape and sample rate
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            
            # Resample if necessary
            if audio_tensor.shape[-1] != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=audio_tensor.shape[-1], 
                    new_freq=self.sample_rate
                )
                audio_tensor = resampler(audio_tensor)
            
            # Transcribe using parakeet model
            with torch.no_grad():
                transcription = self.model.transcribe([audio_tensor])
                if transcription and len(transcription) > 0:
                    return transcription[0]
                return ""
                
        except Exception as e:
            logger.error(f"Error in parakeet transcription: {e}")
            return ""
    
    def start(self):
        """Start audio processing"""
        if not self.is_recording:
            self.is_recording = True
            self.audio_buffer = []
            
            if self.recorder:
                # Start the processing thread
                self.processing_thread = threading.Thread(target=self._process_audio_loop)
                self.processing_thread.daemon = True
                self.processing_thread.start()
                logger.info("Audio processing started")
    
    def stop(self):
        """Stop audio processing"""
        if self.is_recording:
            self.is_recording = False
            
            # Process any remaining audio in buffer
            self._process_final_buffer()
            
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            
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
        """Main audio processing loop"""
        while self.is_recording:
            try:
                # Get audio chunk from queue with timeout
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process the audio chunk
                if self.recorder:
                    # Use RealtimeSTT for real-time processing
                    text = self._transcribe_chunk(audio_chunk)
                    
                    if text and text.strip():
                        # Send partial transcription
                        self.on_transcription(text.strip(), False)
                
            except Exception as e:
                logger.error(f"Error in audio processing loop: {e}")
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe a single audio chunk"""
        try:
            if self.model:
                return self._transcribe_with_parakeet(audio_chunk)
            elif self.recorder:
                # Fallback to default RealtimeSTT
                return self.recorder.transcribe(audio_chunk)
            return ""
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""
    
    def _process_final_buffer(self):
        """Process any remaining audio in the buffer"""
        try:
            with self.buffer_lock:
                if self.audio_buffer:
                    final_chunk = np.array(self.audio_buffer)
                    self.audio_buffer = []
                    
                    # Transcribe final chunk
                    text = self._transcribe_chunk(final_chunk)
                    if text and text.strip():
                        # Send final transcription
                        self.on_transcription(text.strip(), True)
                        
        except Exception as e:
            logger.error(f"Error processing final buffer: {e}")
