import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures
import webrtcvad
from transformers import AutoTokenizer, AutoModelForTokenClassification

# --- Configuration ---
MODEL_NAME = "nvidia/stt_ru_conformer_transducer_large"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
CHUNK_DURATION_SECONDS = 3  # Process audio in 3-second chunks
CHUNK_SIZE_BYTES = CHUNK_DURATION_SECONDS * SAMPLE_RATE * 2  # 16-bit PCM

# VAD Configuration
VAD_AGGRESSIVENESS = 3  # Can be 0, 1, 2, or 3 (most aggressive)
VAD_FRAME_DURATION_MS = 30  # Duration of a frame in milliseconds
VAD_SAMPLE_RATE = SAMPLE_RATE  # Should match the audio sample rate

# --- Global Resources ---
print("Loading NeMo ASR model...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Initialize VAD
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

# Load punctuation model
punctuation_model_name = "RUPunct/RUPunct_big"  # Replace with actual model name or path
print("Loading punctuation model...")
punctuation_tokenizer = AutoTokenizer.from_pretrained(punctuation_model_name)
punctuation_model = AutoModelForTokenClassification.from_pretrained(punctuation_model_name)
punctuation_model.eval()  # Set the model to evaluation mode
print("Punctuation model loaded.")

print("Model loaded and ready.")

def transcribe_chunk(pcm_bytes: bytes) -> str:
    """
    Transcribes an audio chunk directly from memory.
    This function is designed to be run in a separate thread.
    """
    if not pcm_bytes:
        return ""
    # Convert bytes to a NumPy array of 16-bit integers
    pcm_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)

    # Convert to a normalized float32 NumPy array, as expected by the model
    waveform_np = pcm_i16.astype(np.float32) / 32768.0
    try:
        # Run inference in a no_grad context for efficiency
        with torch.no_grad():
            hyps = asr_model.transcribe([waveform_np], batch_size=1)

        # Check if we got a valid, non-empty result
        if hyps and len(hyps) > 0:
            hypothesis = hyps[0]

            # THE FIX IS HERE: Check if it's the Hypothesis object and get its .text attribute
            if hasattr(hypothesis, 'text'):
                text = hypothesis.text
                print(f"Transcription result: '{text}'")
                # Add punctuation
                punctuated_text = add_punctuation(text)
                return punctuated_text
            # Fallback in case the model returns a plain string
            elif isinstance(hypothesis, str):
                print(f"Transcription result: '{hypothesis}'")
                punctuated_text = add_punctuation(hypothesis)
                return punctuated_text
        # Return an empty string if transcription result is empty
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

def add_punctuation(text: str) -> str:
    """
    Adds punctuation to the transcribed text using the punctuation model.
    """
    try:
        # Tokenize the input text
        inputs = punctuation_tokenizer(text, return_tensors="pt", truncation=True)

        # Get predictions
        with torch.no_grad():
            outputs = punctuation_model(**inputs)

        # Convert predictions to punctuated text
        predictions = torch.argmax(outputs.logits, dim=-1)
        punctuated_text = punctuation_tokenizer.decode(predictions[0], skip_special_tokens=True)

        return punctuated_text
    except Exception as e:
        print(f"Punctuation error: {e}")
        return text  # Return original text if punctuation fails

async def recognize(websocket):
    """Handles a single client connection."""
    print("Client connected.")
    loop = asyncio.get_running_loop()
    buffer = bytearray()

    # VAD state
    vad_buffer = bytearray()
    is_speaking = False
    frame_duration = VAD_FRAME_DURATION_MS * SAMPLE_RATE // 1000 * 2  # bytes per frame

    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                buffer.extend(msg)
                vad_buffer.extend(msg)

                # Process audio with VAD
                while len(vad_buffer) >= frame_duration:
                    frame = vad_buffer[:frame_duration]
                    vad_buffer = vad_buffer[frame_duration:]

                    # Check if the frame contains speech
                    is_speech = vad.is_speech(frame, sample_rate=VAD_SAMPLE_RATE)

                    if is_speech:
                        if not is_speaking:
                            is_speaking = True
                            print("Speech detected.")
                    else:
                        if is_speaking:
                            is_speaking = False
                            print("Speech ended.")

                            # Process the accumulated speech
                            if len(buffer) > 0:
                                current_chunk = bytes(buffer)
                                buffer.clear()  # Clear buffer to start accumulating the next chunk
                                # Run the CPU-bound ASR in the thread pool
                                text = await loop.run_in_executor(
                                    pool, partial(transcribe_chunk, current_chunk)
                                )

                                if text:
                                    await websocket.send(json.dumps({"transcript": text}))

            # (Optional) Handle JSON messages for control, e.g., stop
            elif isinstance(msg, str):
                data = json.loads(msg)
                if data.get("action") == "stop":
                    break
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e.reason} (code: {e.code})")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Process any remaining audio in the buffer when the client disconnects
        if buffer:
            print("Processing remaining audio chunk...")
            text = await loop.run_in_executor(
                pool, partial(transcribe_chunk, bytes(buffer))
            )
            if text:
                await websocket.send(json.dumps({"final_transcript": text}))
        print("Client disconnected.")

async def main():
    """Starts the WebSocket server."""
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"ASR WebSocket server started on ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
