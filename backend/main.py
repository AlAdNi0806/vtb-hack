import asyncio
import json
import logging
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from audio_processor import AudioProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real-time Speech-to-Text API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.audio_processors: Dict[str, AudioProcessor] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.audio_processors[client_id] = AudioProcessor(
            on_transcription=lambda text, is_final: asyncio.create_task(
                self.send_transcription(client_id, text, is_final)
            )
        )
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.audio_processors:
            try:
                self.audio_processors[client_id].cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up audio processor for {client_id}: {e}")
            finally:
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

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            try:
                # Receive audio data or control messages with timeout
                data = await asyncio.wait_for(websocket.receive(), timeout=30.0)

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
                            manager.audio_processors[client_id].start()
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": "Recording started"
                            }))

                    elif message.get("type") == "stop_recording":
                        if client_id in manager.audio_processors:
                            manager.audio_processors[client_id].stop()
                            await websocket.send_text(json.dumps({
                                "type": "status",
                                "message": "Recording stopped"
                            }))

                    elif message.get("type") == "ping":
                        # Respond to ping to keep connection alive
                        await websocket.send_text(json.dumps({
                            "type": "pong"
                        }))

            except asyncio.TimeoutError:
                # Send ping to check if connection is still alive
                try:
                    await websocket.send_text(json.dumps({
                        "type": "ping"
                    }))
                except:
                    break
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
    finally:
        manager.disconnect(client_id)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
