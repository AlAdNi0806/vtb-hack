#!/usr/bin/env python3
"""
Ario - AI Voice Assistant Backend
Handles speech-to-text, conversation management, and AI responses via Cerebras
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from typing import Dict, List, Optional
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import soundfile as sf
from RealtimeSTT import AudioToTextRecorder
import requests
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cerebras API configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "csk-6kjk8yewdwdhkc8ndj5686rcj2te3tfyyr6dw669knd3wy33")
CEREBRAS_API_URL = "https://api.cerebras.ai/v1/chat/completions"

app = FastAPI(title="Ario - AI Voice Assistant")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
    
    def add_message(self, client_id: str, role: str, content: str):
        if client_id not in self.conversations:
            self.conversations[client_id] = []
        
        self.conversations[client_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Keep only last 20 messages to manage memory
        if len(self.conversations[client_id]) > 20:
            self.conversations[client_id] = self.conversations[client_id][-20:]
    
    def get_conversation(self, client_id: str) -> List[Dict]:
        return self.conversations.get(client_id, [])
    
    def clear_conversation(self, client_id: str):
        if client_id in self.conversations:
            del self.conversations[client_id]

class AudioProcessor:
    def __init__(self, on_transcription_callback):
        self.on_transcription = on_transcription_callback
        self.recorder = None
        self.is_recording = False
        self.sample_rate = 16000
        self.audio_buffer = []
        self.silence_threshold = 0.01
        self.silence_duration = 0
        self.max_silence_duration = 2.0  # 2 seconds of silence
        
        self._initialize_recorder()
    
    def _initialize_recorder(self):
        """Initialize the speech recognition model"""
        try:
            logger.info("Initializing speech recognition...")
            
            # Use a fast, reliable model for real-time processing
            self.recorder = AudioToTextRecorder(
                model="tiny.en",  # Fast English model
                language="en",
                spinner=False,
                use_microphone=False,
                level=logging.ERROR,
                device="cpu",
                compute_type="int8"
            )
            
            logger.info("✅ Speech recognition initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize speech recognition: {e}")
            self.recorder = None
    
    def start_recording(self):
        """Start recording session"""
        self.is_recording = True
        self.audio_buffer = []
        self.silence_duration = 0
        logger.info("🎤 Recording started")
    
    def stop_recording(self):
        """Stop recording session"""
        self.is_recording = False
        logger.info("⏹️ Recording stopped")
    
    def process_audio_chunk(self, audio_data: bytes):
        """Process incoming audio chunk"""
        if not self.is_recording or not self.recorder:
            return
        
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            if len(audio_np) == 0:
                return
            
            # Add to buffer
            self.audio_buffer.extend(audio_np)
            
            # Process if we have enough audio (0.5 seconds)
            chunk_size = int(self.sample_rate * 0.5)
            if len(self.audio_buffer) >= chunk_size:
                chunk = np.array(self.audio_buffer[:chunk_size], dtype=np.float32)
                self.audio_buffer = self.audio_buffer[chunk_size:]
                
                # Transcribe chunk
                text = self._transcribe_chunk(chunk)
                
                if text and text.strip():
                    # Add punctuation and clean up text
                    cleaned_text = self._clean_and_punctuate(text.strip())
                    self.on_transcription(cleaned_text, False)
                    self.silence_duration = 0
                else:
                    # Track silence
                    self.silence_duration += 0.5
                    
                    # If silence detected for too long, finalize transcription
                    if self.silence_duration >= self.max_silence_duration:
                        self._finalize_transcription()
                        
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
    
    def _transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        """Transcribe audio chunk"""
        try:
            if not self.recorder or len(audio_chunk) == 0:
                return ""
            
            # Create temporary file for transcription
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                sf.write(temp_path, audio_chunk, self.sample_rate)
                
                # Transcribe
                text = self.recorder.transcribe([temp_path])
                
                # Clean up
                os.unlink(temp_path)
                
                return text[0] if text and len(text) > 0 else ""
                
        except Exception as e:
            logger.error(f"Error transcribing chunk: {e}")
            return ""
    
    def _clean_and_punctuate(self, text: str) -> str:
        """Clean and add basic punctuation to text"""
        if not text:
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        
        # Add punctuation based on common patterns
        if text and not text[-1] in '.!?':
            # Add period if it looks like a statement
            if any(word in text.lower() for word in ['is', 'are', 'was', 'were', 'have', 'has', 'will', 'would', 'can', 'could']):
                text += '.'
            # Add question mark if it looks like a question
            elif any(text.lower().startswith(word) for word in ['what', 'where', 'when', 'why', 'how', 'who', 'which', 'do', 'does', 'did', 'can', 'could', 'would', 'should']):
                text += '?'
            else:
                text += '.'
        
        return text
    
    def _finalize_transcription(self):
        """Finalize current transcription"""
        if self.audio_buffer:
            # Process remaining audio
            remaining_audio = np.array(self.audio_buffer, dtype=np.float32)
            self.audio_buffer = []
            
            if len(remaining_audio) > 0:
                text = self._transcribe_chunk(remaining_audio)
                if text and text.strip():
                    cleaned_text = self._clean_and_punctuate(text.strip())
                    self.on_transcription(cleaned_text, True)
        
        self.silence_duration = 0

class CerebrasClient:
    def __init__(self):
        self.api_key = CEREBRAS_API_KEY
        self.api_url = CEREBRAS_API_URL
    
    async def get_ai_response(self, messages: List[Dict], websocket: WebSocket):
        """Get streaming AI response from Cerebras"""
        try:
            # Prepare the request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Add system prompt for voice assistant behavior
            system_prompt = {
                "role": "system",
                "content": "You are Ario, a helpful AI voice assistant. Provide concise, conversational responses. Keep answers brief but informative, as this is a voice conversation. Be friendly and natural."
            }
            
            # Prepare messages with system prompt
            full_messages = [system_prompt] + messages[-10:]  # Last 10 messages for context
            
            payload = {
                "model": "llama3.1-8b",  # Use Cerebras model
                "messages": full_messages,
                "stream": True,
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            # Send streaming request
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Cerebras API error: {response.status_code} - {response.text}")
                await websocket.send_text(json.dumps({
                    "type": "ai_response",
                    "text": "Sorry, I'm having trouble connecting to my AI brain right now. Please try again.",
                    "is_start": True
                }))
                await websocket.send_text(json.dumps({"type": "ai_response_complete"}))
                return
            
            # Process streaming response
            full_response = ""
            is_first_chunk = True
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        
                        if data.strip() == '[DONE]':
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    full_response += content
                                    
                                    # Send chunk to frontend
                                    await websocket.send_text(json.dumps({
                                        "type": "ai_response",
                                        "text": full_response,
                                        "is_start": is_first_chunk
                                    }))
                                    
                                    is_first_chunk = False
                                    
                        except json.JSONDecodeError:
                            continue
            
            # Send completion signal
            await websocket.send_text(json.dumps({"type": "ai_response_complete"}))
            
            return full_response.strip()
            
        except Exception as e:
            logger.error(f"Error getting AI response: {e}")
            await websocket.send_text(json.dumps({
                "type": "ai_response",
                "text": "Sorry, I encountered an error while processing your request. Please try again.",
                "is_start": True
            }))
            await websocket.send_text(json.dumps({"type": "ai_response_complete"}))
            return None

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_processors: Dict[str, AudioProcessor] = {}
        self.conversation_manager = ConversationManager()
        self.cerebras_client = CerebrasClient()

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Create audio processor for this client
        self.audio_processors[client_id] = AudioProcessor(
            on_transcription_callback=lambda text, is_final: asyncio.create_task(
                self.send_transcription(client_id, text, is_final)
            )
        )
        
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_processors:
            del self.audio_processors[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def send_transcription(self, client_id: str, text: str, is_final: bool):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(
                    json.dumps({
                        "type": "transcription",
                        "text": text,
                        "is_final": is_final
                    })
                )
            except Exception as e:
                logger.error(f"Error sending transcription to {client_id}: {e}")

    async def handle_ai_query(self, client_id: str, text: str, conversation_history: List[Dict]):
        """Handle AI query and stream response"""
        if client_id not in self.active_connections:
            return
        
        # Add user message to conversation
        self.conversation_manager.add_message(client_id, "user", text)
        
        # Get conversation context
        messages = []
        for msg in self.conversation_manager.get_conversation(client_id):
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get AI response
        websocket = self.active_connections[client_id]
        ai_response = await self.cerebras_client.get_ai_response(messages, websocket)
        
        if ai_response:
            # Add AI response to conversation
            self.conversation_manager.add_message(client_id, "assistant", ai_response)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive()
            
            if "bytes" in data:
                # Audio data received
                audio_data = data["bytes"]
                if client_id in manager.audio_processors:
                    manager.audio_processors[client_id].process_audio_chunk(audio_data)
            
            elif "text" in data:
                # Control message received
                message = json.loads(data["text"])
                
                if message.get("type") == "start_recording":
                    if client_id in manager.audio_processors:
                        manager.audio_processors[client_id].start_recording()
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "Recording started"
                        }))
                
                elif message.get("type") == "stop_recording":
                    if client_id in manager.audio_processors:
                        manager.audio_processors[client_id].stop_recording()
                        await websocket.send_text(json.dumps({
                            "type": "status",
                            "message": "Recording stopped"
                        }))
                
                elif message.get("type") == "ai_query":
                    # Handle AI query
                    text = message.get("text", "")
                    conversation_history = message.get("conversation_history", [])
                    
                    if text.strip():
                        await manager.handle_ai_query(client_id, text, conversation_history)
                        
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Ario AI Voice Assistant"}

if __name__ == "__main__":
    # Check for Cerebras API key
    if CEREBRAS_API_KEY == "your-cerebras-api-key-here":
        logger.warning("⚠️ Please set your CEREBRAS_API_KEY environment variable")
        logger.info("💡 Export it like: export CEREBRAS_API_KEY='your-actual-key'")
    
    logger.info("🚀 Starting Ario AI Voice Assistant...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
