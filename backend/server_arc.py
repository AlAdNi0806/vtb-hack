import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures
import time
import logging
import re
from huggingface_hub import login, HfFolder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ASR_Server")

# --- Configuration ---
ASR_MODEL_NAME = "nvidia/stt_ru_conformer_transducer_large"
# CORRECTED MODEL NAME - NVIDIA uses "nemo_" prefix for VAD models
VAD_MODEL_NAME = "nvidia/nemo_vad_multilingual_1.1_small"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
MAX_UTTERANCE_LENGTH = 30.0  # Maximum utterance length in seconds
SILENCE_DURATION = 0.8  # Seconds of silence to consider end of utterance
VAD_WINDOW_SIZE = 0.05  # VAD processes audio in 50ms windows

# --- Hugging Face Authentication ---
def setup_hf_authentication():
    """Set up Hugging Face authentication for gated models."""
    token = HfFolder.get_token()
    
    if not token:
        logger.warning("""
        No Hugging Face token found. You need to:
        1. Create an account at https://huggingface.co
        2. Visit https://huggingface.co/nvidia/nemo_vad_multilingual_1.1_small and agree to the terms
        3. Get your token from https://huggingface.co/settings/tokens
        4. Run 'huggingface-cli login' and enter your token
        
        Alternatively, set the HF_TOKEN environment variable.
        """)
        return False
    
    try:
        # Verify the token works
        from huggingface_hub import whoami
        user = whoami(token=token)
        logger.info(f"Authenticated with Hugging Face as {user['name']}")
        return True
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        logger.warning("""
        Your Hugging Face token is invalid or doesn't have access to the VAD model.
        Please make sure you've accepted the terms at:
        https://huggingface.co/nvidia/nemo_vad_multilingual_1.1_small
        """)
        return False

# --- Global Resources ---
logger.info("Setting up Hugging Face authentication...")
if not setup_hf_authentication():
    logger.warning("Continuing without VAD model authentication - may fail to load VAD model")

logger.info("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(ASR_MODEL_NAME)

logger.info("Loading NeMo VAD model...")
try:
    # Try to load VAD model with explicit token handling
    vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(
        VAD_MODEL_NAME,
        # Pass token explicitly if available
        **({"token": HfFolder.get_token()} if HfFolder.get_token() else {})
    )
    logger.info("VAD model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load VAD model: {e}")
    logger.warning("Falling back to basic chunking without VAD - transcription quality may suffer")
    vad_model = None

pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
logger.info("All models loaded and ready.")

# --------------------------------------------------------------
# VAD processing function to detect speech segments
def detect_voice_activity(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE):
    """Detects voice activity in audio buffer using NeMo VAD."""
    if vad_model is None:
        # Fallback to simple energy-based VAD if model failed to load
        logger.warning("Using simple energy-based VAD fallback")
        if not pcm_bytes:
            return False, []
        
        # Simple energy-based VAD (threshold on audio amplitude)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        energy = np.mean(np.abs(audio))
        return energy > 0.05, [[0, len(audio)]]
    
    if not pcm_bytes:
        return False, []
    
    # Convert bytes to numpy array
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    try:
        # Process through VAD model
        with torch.no_grad():
            # VAD model expects audio tensor and length tensor
            audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dimension
            audio_len = torch.tensor([len(audio)])
            
            # Get VAD predictions
            vad_probs = vad_model.classify(audio_signal=audio_tensor, length=audio_len)
            
            # Extract speech probabilities
            # VAD output structure: [batch, time, classes] - we want class 1 (speech)
            if isinstance(vad_probs, tuple):
                vad_probs = vad_probs[0]
            
            # Convert to numpy and extract speech probabilities (class 1)
            vad_probs_np = vad_probs.cpu().numpy()[0, :, 1]  # [time, speech_prob]
            
            # Detect speech segments
            speech_segments = []
            current_segment = None
            
            # Calculate window size in samples
            window_samples = int(VAD_WINDOW_SIZE * sample_rate)
            
            for i, prob in enumerate(vad_probs_np):
                is_speech = prob > VAD_THRESHOLD
                
                if is_speech and current_segment is None:
                    # Start of speech segment
                    current_segment = [i * window_samples, i * window_samples]
                elif is_speech and current_segment is not None:
                    # Extend current segment
                    current_segment[1] = (i + 1) * window_samples
                elif not is_speech and current_segment is not None:
                    # End of speech segment
                    speech_segments.append(current_segment)
                    current_segment = None
            
            # Add last segment if exists
            if current_segment is not None:
                speech_segments.append(current_segment)
                
            return len(speech_segments) > 0, speech_segments
            
    except Exception as e:
        logger.error(f"VAD processing error: {e}", exc_info=True)
        # If VAD fails, assume there's speech to avoid breaking the pipeline
        return True, [[0, len(audio)]]


# --------------------------------------------------------------
# Rule-based punctuation restoration for Russian
def rule_based_punctuation_restoration(text: str) -> str:
    """Apply rule-based punctuation restoration for Russian text."""
    if not text:
        return ""
    
    # Clean up text
    text = text.strip()
    if not text:
        return ""
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Ensure text ends with proper punctuation
    if not text.endswith(('.', '!', '?', '...')):
        # Simple heuristic: if text contains question words, add question mark
        question_words = ['кто', 'что', 'где', 'когда', 'почему', 'зачем', 'как', 'какой', 'ли']
        if any(word in text.lower().split() for word in question_words):
            text += '?'
        else:
            text += '.'
    
    return text

# --------------------------------------------------------------
# Transcription function with buffer management
def transcribe_utterance(pcm_bytes: bytes) -> str:
    """Transcribes a complete utterance with punctuation restoration."""
    if not pcm_bytes:
        return ""
    
    # Convert bytes to normalized float32 array
    waveform_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    try:
        # Transcribe with ASR model
        with torch.no_grad():
            hyps = asr_model.transcribe([waveform_np], batch_size=1)
        
        if hyps and len(hyps) > 0:
            hypothesis = hyps[0]
            text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
            
            # Clean up the transcription
            text = re.sub(r'\s+', ' ', text).strip()
            if not text:
                return ""
            
            # Apply punctuation restoration
            punctuated_text = rule_based_punctuation_restoration(text)
            
            return punctuated_text
        
        return ""
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return ""

# --------------------------------------------------------------
async def recognize(websocket):
    """Handles a single client connection with VAD-based utterance detection."""
    logger.info("Client connected.")
    loop = asyncio.get_running_loop()
    audio_buffer = bytearray()
    last_speech_time = time.time()
    is_speaking = False
    utterance_start_time = None
    partial_transcript = ""
    last_partial_update = time.time()
    
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                # Add new audio to buffer
                audio_buffer.extend(msg)
                
                current_time = time.time()
                
                # Only run VAD processing periodically to reduce load
                if current_time - last_speech_time > 0.1 or not is_speaking:
                    # Process buffer with VAD in a thread
                    has_speech, _ = await loop.run_in_executor(
                        pool, partial(detect_voice_activity, bytes(audio_buffer))
                    )
                    
                    # Update speaking state
                    if has_speech:
                        if not is_speaking:
                            # Speech just started
                            is_speaking = True
                            utterance_start_time = current_time
                            logger.info("Speech detected - starting utterance")
                            partial_transcript = ""
                        
                        # Reset silence timer
                        last_speech_time = current_time
                    else:
                        if is_speaking:
                            # Check if silence duration exceeds threshold
                            silence_duration = current_time - last_speech_time
                            
                            # Check if we've reached max utterance length
                            utterance_duration = current_time - utterance_start_time
                            
                            if silence_duration >= SILENCE_DURATION or utterance_duration >= MAX_UTTERANCE_LENGTH:
                                # End of utterance detected
                                logger.info(f"End of utterance detected after {silence_duration:.2f}s silence")
                                
                                # Process the complete utterance
                                utterance = bytes(audio_buffer)
                                audio_buffer.clear()
                                
                                # Transcribe in thread
                                text = await loop.run_in_executor(
                                    pool, partial(transcribe_utterance, utterance)
                                )
                                
                                if text:
                                    # Send the complete utterance
                                    await websocket.send(json.dumps({
                                        "transcript": text,
                                        "is_final": True
                                    }))
                                    logger.info(f"Sent final transcript: {text}")
                                
                                # Reset state
                                is_speaking = False
                                utterance_start_time = None
                                partial_transcript = ""
                        
                # Optional: Send partial results for UI feedback (every 1.5 seconds)
                if is_speaking and current_time - last_speech_time < SILENCE_DURATION * 0.7:
                    if current_time - last_partial_update > 1.5:
                        # Create a partial transcription (without final punctuation)
                        partial_audio = bytes(audio_buffer)
                        partial_text = await loop.run_in_executor(
                            pool, partial(transcribe_utterance, partial_audio)
                        )
                        
                        if partial_text and partial_text != partial_transcript:
                            # Remove any ending punctuation for partial results
                            partial_text = partial_text.rstrip('.!?')
                            if partial_text:
                                await websocket.send(json.dumps({
                                    "transcript": partial_text,
                                    "is_final": False
                                }))
                                partial_transcript = partial_text
                                last_partial_update = current_time
                
                # Prevent buffer from growing too large
                max_buffer_size = int(MAX_UTTERANCE_LENGTH * SAMPLE_RATE * 2)
                if len(audio_buffer) > max_buffer_size:
                    audio_buffer = audio_buffer[-max_buffer_size:]
            
            # Handle control messages
            elif isinstance(msg, str):
                try:
                    data = json.loads(msg)
                    if data.get("action") == "stop":
                        break
                    elif data.get("action") == "status":
                        # Send system status
                        await websocket.send(json.dumps({
                            "status": "running",
                            "models_loaded": True,
                            "vad_active": vad_model is not None
                        }))
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {msg}")

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"Connection closed: {e.reason} (code: {e.code})")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Process any remaining audio on disconnect
        if audio_buffer and is_speaking:
            logger.info("Processing remaining audio on disconnect...")
            text = await loop.run_in_executor(
                pool, partial(transcribe_utterance, bytes(audio_buffer))
            )
            if text:
                await websocket.send(json.dumps({
                    "transcript": text,
                    "is_final": True
                }))
        logger.info("Client disconnected.")

# --------------------------------------------------------------
async def main():
    """Starts the WebSocket server."""
    logger.info(f"Starting ASR WebSocket server on ws://{SERVER_HOST}:{SERVER_PORT}")
    logger.info(f"Models loaded:")
    logger.info(f"  ASR: {ASR_MODEL_NAME}")
    logger.info(f"  VAD: {VAD_MODEL_NAME} {'(loaded)' if vad_model else '(not loaded - using fallback)'}")
    
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        logger.info(f"ASR WebSocket server started successfully!")
        logger.info(f"Connect to: ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user.")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)