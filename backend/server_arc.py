import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures
import time

# --- Configuration ---
ASR_MODEL_NAME = "nvidia/stt_ru_conformer_transducer_large"
VAD_MODEL_NAME = "nvidia/vad_multilingual_1.1_small"  # Official VAD model
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
MAX_UTTERANCE_LENGTH = 30.0
SILENCE_DURATION = 0.8
VAD_WINDOW_LENGTH_IN_SEC = 0.5  # VAD processes in 0.5s chunks
VAD_SHIFT_MS = 10  # Stride for VAD processing

# --- Global Resources ---
print("Loading NeMo ASR and VAD models...")

# Load ASR model
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(ASR_MODEL_NAME)

# Load VAD model (correct class)
vad_model = nemo_asr.models.EncDecClassificationModel.from_pretrained(VAD_MODEL_NAME)

# Move models to GPU if available
if torch.cuda.is_available():
    asr_model = asr_model.to('cuda')
    vad_model = vad_model.to('cuda')

# Set both models to eval mode
asr_model.eval()
vad_model.eval()

pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
print("All models loaded and ready.")

# --------------------------------------------------------------
# VAD processing function to detect speech segments
def detect_voice_activity(pcm_bytes: bytes):
    """Detects voice activity in audio buffer using NeMo VAD."""
    if not pcm_bytes or len(pcm_bytes) == 0:
        return False

    # Convert bytes to numpy array
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]

    # Create a tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dim
    if torch.cuda.is_available():
        audio_tensor = audio_tensor.to('cuda')

    # VAD model expects fixed window sizes; we'll process in chunks
    window_size = int(VAD_WINDOW_LENGTH_IN_SEC * SAMPLE_RATE)
    
    # If audio is shorter than window, pad it
    if audio_tensor.shape[1] < window_size:
        padding = torch.zeros((1, window_size - audio_tensor.shape[1]))
        if torch.cuda.is_available():
            padding = padding.to('cuda')
        audio_tensor = torch.cat([audio_tensor, padding], dim=1)
    
    # Split into overlapping windows if longer
    num_windows = max(1, int((audio_tensor.shape[1] - window_size) / (VAD_SHIFT_MS / 1000 * SAMPLE_RATE)) + 1)
    speech_detected = False

    with torch.no_grad():
        for i in range(num_windows):
            start_idx = int(i * (VAD_SHIFT_MS / 1000 * SAMPLE_RATE))
            end_idx = start_idx + window_size
            if end_idx > audio_tensor.shape[1]:
                break
            chunk = audio_tensor[:, start_idx:end_idx]

            # Run VAD
            logits = vad_model.forward(input_signal=chunk, input_signal_length=torch.tensor([chunk.shape[1]]).to(chunk.device))
            prob = torch.sigmoid(logits).cpu().numpy()[0][0]

            if prob > VAD_THRESHOLD:
                speech_detected = True
                break  # Early exit if speech found

    return speech_detected

# --------------------------------------------------------------
# Rule-based punctuation restoration for Russian
def rule_based_punctuation_restoration(text: str) -> str:
    """Apply rule-based punctuation restoration for Russian text."""
    if not text:
        return ""
    
    text = text.strip()
    if not text:
        return ""
    
    # Capitalize first letter
    text = text[0].upper() + text[1:] if text else text
    
    # Add ending punctuation
    if not text.endswith(('.', '!', '?')):
        question_words = ['кто', 'что', 'где', 'когда', 'почему', 'зачем', 'как', 'какой', 'не', 'ни']
        if any(text.lower().startswith(word) for word in question_words):
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
    waveform_np = np.expand_dims(waveform_np, axis=0)  # Add batch dim
    
    try:
        with torch.no_grad():
            # Transcribe
            if torch.cuda.is_available():
                # Move to GPU
                waveform_np = torch.tensor(waveform_np).to('cuda')
                hyps = asr_model.transcribe(waveform_np, batch_size=1)
            else:
                hyps = asr_model.transcribe([waveform_np], batch_size=1)
        
        if hyps and len(hyps) > 0:
            text = hyps[0]
            if hasattr(text, 'text'):
                text = text.text
            text = text.strip()
            
            # Apply punctuation
            return rule_based_punctuation_restoration(text)
        
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# --------------------------------------------------------------
async def recognize(websocket):
    """Handles a single client connection with VAD-based utterance detection."""
    print("Client connected.")
    loop = asyncio.get_running_loop()
    audio_buffer = bytearray()
    last_speech_time = time.time()
    is_speaking = False
    utterance_start_time = None
    
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                # Add new audio to buffer
                audio_buffer.extend(msg)
                
                current_time = time.time()
                
                # Only check VAD every 100ms to reduce load
                if current_time - last_speech_time >= 0.1:
                    # Convert buffer to bytes for VAD
                    audio_chunk = bytes(audio_buffer[-int(SAMPLE_RATE * 0.5):])  # Last 500ms
                    
                    has_speech = await loop.run_in_executor(
                        pool, detect_voice_activity, audio_chunk
                    )
                    
                    if has_speech:
                        if not is_speaking:
                            is_speaking = True
                            utterance_start_time = current_time
                            print("🎤 Speech detected - starting utterance")
                        
                        last_speech_time = current_time
                    else:
                        if is_speaking:
                            silence_duration = current_time - last_speech_time
                            utterance_duration = current_time - utterance_start_time
                            
                            if silence_duration >= SILENCE_DURATION or utterance_duration >= MAX_UTTERANCE_LENGTH:
                                print(f"🛑 Utterance ended after {silence_duration:.2f}s silence")
                                
                                # Process full utterance
                                utterance = bytes(audio_buffer)
                                audio_buffer.clear()
                                
                                text = await loop.run_in_executor(
                                    pool, partial(transcribe_utterance, utterance)
                                )
                                
                                if text:
                                    await websocket.send(json.dumps({
                                        "transcript": text,
                                        "is_final": True
                                    }))
                                    print(f"✅ Final transcript: {text}")
                                
                                is_speaking = False
                                utterance_start_time = None
                    
                    # Prevent buffer overflow
                    max_buf = int(MAX_UTTERANCE_LENGTH * SAMPLE_RATE * 2)
                    if len(audio_buffer) > max_buf:
                        audio_buffer = audio_buffer[-max_buf:]
            
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e.reason}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if audio_buffer and is_speaking:
            print("Processing remaining audio...")
            text = await loop.run_in_executor(
                pool, partial(transcribe_utterance, bytes(audio_buffer))
            )
            if text:
                await websocket.send(json.dumps({
                    "transcript": text,
                    "is_final": True
                }))
        print("Client disconnected.")

# --------------------------------------------------------------
async def main():
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"🎙️ ASR WebSocket server running on ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())