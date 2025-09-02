import asyncio
import websockets
import json
import torch
import numpy as np
import nemo.collections.asr as nemo_asr
from functools import partial
import concurrent.futures
import time
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

# --- Configuration ---
ASR_MODEL_NAME = "nvidia/stt_ru_conformer_transducer_large"
VAD_MODEL_NAME = "nvidia/vad_multilingual_1.1_small"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8765
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.5
MAX_UTTERANCE_LENGTH = 30.0
SILENCE_DURATION = 0.8

# --- Global Resources ---
print("Loading NeMo ASR and VAD models...")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(ASR_MODEL_NAME)
vad_model = nemo_asr.models.SpeakerDiarizationACAModel.from_pretrained(VAD_MODEL_NAME)
pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Load Russian BERT for punctuation restoration
print("Loading Russian BERT for punctuation restoration...")
try:
    # Using a multilingual BERT model that can handle Russian punctuation
    # You can also use a specifically trained Russian punctuation model
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModelForTokenClassification.from_pretrained("cointegrated/rubert-tiny2")
    
    # Create punctuation restoration pipeline
    punct_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )
    print("Russian BERT model loaded for punctuation restoration.")
except Exception as e:
    print(f"Could not load BERT punctuation model: {e}")
    print("Falling back to rule-based punctuation.")
    punct_pipeline = None

print("All models loaded and ready.")

# --------------------------------------------------------------
# VAD processing function to detect speech segments
def detect_voice_activity(pcm_bytes: bytes, sample_rate: int = SAMPLE_RATE):
    """Detects voice activity in audio buffer using NeMo VAD."""
    if not pcm_bytes:
        return False, []
    
    # Convert bytes to numpy array
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    
    try:
        # Process through VAD model
        with torch.no_grad():
            # Reshape for VAD model (this is simplified - actual implementation may vary)
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            lengths = torch.tensor([len(audio)])
            
            # Get VAD predictions
            vad_probs = vad_model.forward(input_signal=audio_tensor, input_signal_length=lengths)
            
            # Extract speech probabilities
            if isinstance(vad_probs, tuple):
                vad_probs = vad_probs[0]
            
            # Convert to numpy
            vad_probs = vad_probs.cpu().numpy()[0]
            
            # Detect speech segments
            speech_segments = []
            current_segment = None
            
            for i, prob in enumerate(vad_probs):
                is_speech = prob > VAD_THRESHOLD
                
                if is_speech and current_segment is None:
                    current_segment = [i, i]
                elif is_speech and current_segment is not None:
                    current_segment[1] = i
                elif not is_speech and current_segment is not None:
                    speech_segments.append(current_segment)
                    current_segment = None
            
            # Add last segment if exists
            if current_segment is not None:
                speech_segments.append(current_segment)
                
            return len(speech_segments) > 0, speech_segments
            
    except Exception as e:
        print(f"VAD processing error: {e}")
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
        question_words = ['кто', 'что', 'где', 'когда', 'почему', 'зачем', 'как', 'какой']
        if any(word in text.lower() for word in question_words):
            text += '?'
        else:
            text += '.'
    
    return text

# --------------------------------------------------------------
# BERT-based punctuation restoration function
def bert_punctuation_restoration(text: str) -> str:
    """Use BERT to restore punctuation in text."""
    if not text or not punct_pipeline:
        return rule_based_punctuation_restoration(text)
    
    try:
        # Clean the text
        text = text.strip().lower()
        if not text:
            return ""
        
        # Use the BERT pipeline for punctuation restoration
        # This is a simplified implementation - you might need a specialized model
        predictions = punct_pipeline(text)
        
        # Sort predictions by start position
        predictions.sort(key=lambda x: x['start'])
        
        # Reconstruct text with punctuation
        result_parts = []
        last_pos = 0
        
        for pred in predictions:
            # Add text before this prediction
            result_parts.append(text[last_pos:pred['start']])
            last_pos = pred['end']
            
            # Add the predicted entity (punctuation)
            if pred['entity_group'] in ['PERIOD', 'COMMA', 'QUESTION', 'EXCLAMATION']:
                punct_map = {
                    'PERIOD': '.',
                    'COMMA': ',',
                    'QUESTION': '?',
                    'EXCLAMATION': '!'
                }
                result_parts.append(punct_map.get(pred['entity_group'], ''))
        
        # Add remaining text
        result_parts.append(text[last_pos:])
        
        result = ''.join(result_parts).strip()
        
        # Apply final formatting
        if result:
            # Capitalize first letter
            result = result[0].upper() + result[1:]
            
            # Ensure proper ending punctuation
            if not result.endswith(('.', '!', '?')):
                # Use rule-based approach to determine appropriate ending
                question_words = ['кто', 'что', 'где', 'когда', 'почему', 'зачем', 'как', 'какой']
                if any(word in result.lower() for word in question_words):
                    result += '?'
                else:
                    result += '.'
        
        return result
        
    except Exception as e:
        print(f"BERT punctuation restoration error: {e}")
        # Fall back to rule-based if BERT fails
        return rule_based_punctuation_restoration(text)

# --------------------------------------------------------------
# Transcription function with buffer management
def transcribe_utterance(pcm_bytes: bytes) -> str:
    """Transcribes a complete utterance with BERT-based punctuation restoration."""
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
            text = text.strip()
            if not text:
                return ""
            
            # Apply BERT-based punctuation restoration
            punctuated_text = bert_punctuation_restoration(text)
            
            return punctuated_text
        
        return ""
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""

# --------------------------------------------------------------
# Enhanced sentence boundary detection using BERT embeddings
def detect_sentence_boundaries(text: str):
    """Use BERT embeddings to detect natural sentence boundaries."""
    if not text or not punct_pipeline:
        # Fall back to simple punctuation-based splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    try:
        # This is a conceptual implementation - you'd need a specialized model
        # For demonstration, we'll use the punctuation pipeline to identify boundaries
        predictions = punct_pipeline(text.lower())
        
        # Extract sentence boundaries based on punctuation predictions
        boundaries = []
        for pred in predictions:
            if pred['entity_group'] in ['PERIOD', 'QUESTION', 'EXCLAMATION']:
                # This is a potential sentence boundary
                end_pos = pred['end']
                boundaries.append(end_pos)
        
        # Split text at boundaries
        sentences = []
        start = 0
        for end in sorted(boundaries):
            sentence = text[start:end].strip()
            if sentence:
                sentences.append(sentence)
            start = end
        
        # Add final segment
        final_segment = text[start:].strip()
        if final_segment:
            sentences.append(final_segment)
            
        return sentences
        
    except Exception as e:
        print(f"Sentence boundary detection error: {e}")
        # Fall back to simple splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

# --------------------------------------------------------------
async def recognize(websocket):
    """Handles a single client connection with VAD-based utterance detection."""
    print("Client connected.")
    loop = asyncio.get_running_loop()
    audio_buffer = bytearray()
    last_speech_time = time.time()
    is_speaking = False
    utterance_start_time = None
    partial_transcript = ""
    
    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                # Add new audio to buffer
                audio_buffer.extend(msg)
                
                # Get current time
                current_time = time.time()
                
                # Only run VAD processing periodically to reduce load
                if current_time - last_speech_time > 0.2 or not is_speaking:
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
                            print("Speech detected - starting utterance")
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
                                print(f"End of utterance detected after {silence_duration:.2f}s silence")
                                
                                # Process the complete utterance
                                utterance = bytes(audio_buffer)
                                audio_buffer.clear()
                                
                                # Transcribe in thread
                                text = await loop.run_in_executor(
                                    pool, partial(transcribe_utterance, utterance)
                                )
                                
                                if text:
                                    # Detect sentence boundaries
                                    sentences = detect_sentence_boundaries(text)
                                    
                                    # Send each complete sentence
                                    for sentence in sentences:
                                        if sentence.strip():
                                            await websocket.send(json.dumps({
                                                "transcript": sentence,
                                                "is_final": True
                                            }))
                                            print(f"Sent final sentence: {sentence}")
                                
                                # Reset state
                                is_speaking = False
                                utterance_start_time = None
                                partial_transcript = ""
                        
                # Optional: Send partial results for UI feedback
                if is_speaking and current_time - last_speech_time < SILENCE_DURATION * 0.7:
                    # Only send partial updates periodically to avoid flooding
                    if current_time - last_speech_time > 1.0 and current_time % 2 < 0.1:
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
                            "vad_active": True
                        }))
                except json.JSONDecodeError:
                    print(f"Invalid JSON message: {msg}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed: {e.reason} (code: {e.code})")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Process any remaining audio on disconnect
        if audio_buffer and is_speaking:
            print("Processing remaining audio on disconnect...")
            text = await loop.run_in_executor(
                pool, partial(transcribe_utterance, bytes(audio_buffer))
            )
            if text:
                sentences = detect_sentence_boundaries(text)
                for sentence in sentences:
                    if sentence.strip():
                        await websocket.send(json.dumps({
                            "transcript": sentence,
                            "is_final": True
                        }))
        print("Client disconnected.")

# --------------------------------------------------------------
async def main():
    """Starts the WebSocket server."""
    print(f"Starting ASR WebSocket server on ws://{SERVER_HOST}:{SERVER_PORT}")
    print("Models loaded:")
    print(f"  ASR: {ASR_MODEL_NAME}")
    print(f"  VAD: {VAD_MODEL_NAME}")
    print(f"  Punctuation: {'BERT-based' if punct_pipeline else 'Rule-based'}")
    
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"ASR WebSocket server started successfully!")
        print(f"Connect to: ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
    except Exception as e:
        print(f"Failed to start server: {e}")
        import traceback
        traceback.print_exc()