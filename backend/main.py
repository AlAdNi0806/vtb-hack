"""
Real-time Conversational AI Backend Service
Main FastAPI application with WebSocket support for real-time audio processing
"""

import asyncio
import json
import logging
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from services.ai_service import AIService
from services.audio_processor import AudioProcessor
from services.audio_pipeline import AudioPipeline
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Conversational AI Backend",
    description="Real-time voice conversation with AI using STT, TTS, and turn detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
ai_service: AIService = None
audio_processor: AudioProcessor = None
audio_pipeline: AudioPipeline = None

# Active WebSocket connections and sessions
active_connections: Dict[str, WebSocket] = {}
active_sessions: Dict[str, str] = {}  # connection_id -> session_id


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global ai_service, audio_processor, audio_pipeline

    logger.info("Starting Conversational AI Backend Service...")

    try:
        # Initialize AI service
        ai_service = AIService()
        await ai_service.initialize()

        # Initialize audio processor
        audio_processor = AudioProcessor()
        await audio_processor.initialize()

        # Initialize audio pipeline
        audio_pipeline = AudioPipeline(audio_processor, ai_service)

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down services...")
    
    if ai_service:
        await ai_service.cleanup()
    
    if audio_processor:
        await audio_processor.cleanup()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Conversational AI Backend is running", "status": "healthy"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    health_status = {
        "status": "healthy",
        "services": {
            "ai_service": ai_service.is_ready() if ai_service else False,
            "audio_processor": audio_processor.is_ready() if audio_processor else False,
            "audio_pipeline": audio_pipeline is not None,
        },
        "active_connections": len(active_connections)
    }
    
    return JSONResponse(content=health_status)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time audio communication"""
    await websocket.accept()
    
    # Generate unique connection ID
    connection_id = f"conn_{len(active_connections)}"
    active_connections[connection_id] = websocket
    
    logger.info(f"New WebSocket connection: {connection_id}")
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "connection_id": connection_id,
            "message": "Connected to Conversational AI"
        })
        
        # Handle incoming messages
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                await handle_websocket_message(websocket, connection_id, data)
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Cleanup connection
        if connection_id in active_connections:
            del active_connections[connection_id]


async def handle_websocket_message(websocket: WebSocket, connection_id: str, data: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    message_type = data.get("type")

    if message_type == "audio_chunk":
        # Process audio chunk through pipeline
        audio_data = data.get("audio_data")
        if audio_data:
            await process_audio_chunk(websocket, connection_id, audio_data)

    elif message_type == "start_conversation":
        # Start conversation session
        session_id = f"session_{connection_id}_{int(asyncio.get_event_loop().time())}"
        active_sessions[connection_id] = session_id

        result = await audio_pipeline.start_session(session_id)

        await websocket.send_json({
            "type": "conversation_started",
            "session_id": session_id,
            "message": "Ready to listen"
        })

    elif message_type == "end_conversation":
        # End conversation session
        if connection_id in active_sessions:
            await audio_pipeline.end_session()
            del active_sessions[connection_id]

        await websocket.send_json({
            "type": "conversation_ended",
            "message": "Conversation ended"
        })

    else:
        await websocket.send_json({
            "type": "error",
            "message": f"Unknown message type: {message_type}"
        })


async def process_audio_chunk(websocket: WebSocket, connection_id: str, audio_data: str):
    """Process incoming audio chunk through the AI pipeline"""
    try:
        # Check if session is active
        if connection_id not in active_sessions:
            await websocket.send_json({
                "type": "error",
                "message": "No active conversation session"
            })
            return

        # Process audio through the pipeline
        import time
        result = await audio_pipeline.process_audio_chunk(audio_data, time.time())

        # Send transcript if available
        if result.get("transcript"):
            await websocket.send_json({
                "type": "transcript",
                "text": result["transcript"],
                "is_final": result.get("is_final", False),
                "turn_detected": result.get("turn_detected", False)
            })

        # Send AI response if available
        if result.get("ai_response"):
            await websocket.send_json({
                "type": "ai_response",
                "text": result["ai_response"],
                "audio_data": result.get("tts_audio", "")
            })

    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}")
        await websocket.send_json({
            "type": "error",
            "message": f"Audio processing error: {str(e)}"
        })


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
