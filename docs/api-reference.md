# API Reference

This document provides detailed information about the Conversational AI API endpoints and WebSocket messages.

## REST API Endpoints

### Health Check
**GET** `/health`

Returns the health status of the backend services.

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "ai_service": true,
    "audio_processor": true,
    "audio_pipeline": true
  },
  "active_connections": 2
}
```

### Root Endpoint
**GET** `/`

Basic endpoint that returns service information.

**Response:**
```json
{
  "message": "Conversational AI Backend is running",
  "status": "healthy"
}
```

## WebSocket API

### Connection
**WebSocket** `/ws`

Establishes a WebSocket connection for real-time audio communication.

### Message Types

#### 1. Connection Established
**Direction:** Server → Client

```json
{
  "type": "connection_established",
  "connection_id": "conn_0",
  "message": "Connected to Conversational AI"
}
```

#### 2. Start Conversation
**Direction:** Client → Server

```json
{
  "type": "start_conversation"
}
```

**Response:**
```json
{
  "type": "conversation_started",
  "session_id": "session_conn_0_1234567890",
  "message": "Ready to listen"
}
```

#### 3. Audio Chunk
**Direction:** Client → Server

```json
{
  "type": "audio_chunk",
  "audio_data": "base64_encoded_audio_data"
}
```

#### 4. Transcript
**Direction:** Server → Client

```json
{
  "type": "transcript",
  "text": "Hello, how are you?",
  "is_final": true,
  "turn_detected": false
}
```

#### 5. AI Response
**Direction:** Server → Client

```json
{
  "type": "ai_response",
  "text": "I'm doing well, thank you! How can I help you today?",
  "audio_data": "base64_encoded_tts_audio"
}
```

#### 6. End Conversation
**Direction:** Client → Server

```json
{
  "type": "end_conversation"
}
```

**Response:**
```json
{
  "type": "conversation_ended",
  "message": "Conversation ended"
}
```

#### 7. Error
**Direction:** Server → Client

```json
{
  "type": "error",
  "message": "Error description"
}
```

## Audio Format Specifications

### Input Audio (Client → Server)
- **Format:** WAV (base64 encoded)
- **Sample Rate:** 16,000 Hz
- **Channels:** Mono (1 channel)
- **Bit Depth:** 16-bit
- **Encoding:** PCM

### Output Audio (Server → Client)
- **Format:** WAV (base64 encoded)
- **Sample Rate:** 16,000 Hz
- **Channels:** Mono (1 channel)
- **Bit Depth:** 16-bit
- **Encoding:** PCM

## Configuration Parameters

### Backend Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CEREBRAS_API_KEY` | string | required | Cerebras API key for LLM |
| `HOST` | string | `0.0.0.0` | Server host address |
| `PORT` | integer | `8000` | Server port |
| `DEBUG` | boolean | `True` | Debug mode |
| `SAMPLE_RATE` | integer | `16000` | Audio sample rate |
| `CHUNK_SIZE` | integer | `1024` | Audio chunk size |
| `LOG_LEVEL` | string | `INFO` | Logging level |

### Frontend Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | string | `http://localhost:8000` | Backend API URL |
| `NEXT_PUBLIC_WS_URL` | string | `ws://localhost:8000/ws` | WebSocket URL |
| `NEXT_PUBLIC_SAMPLE_RATE` | integer | `16000` | Audio sample rate |
| `NEXT_PUBLIC_DEBUG` | boolean | `true` | Debug mode |

## Error Codes

### WebSocket Errors

| Code | Message | Description |
|------|---------|-------------|
| `INVALID_JSON` | "Invalid JSON format" | Malformed JSON in message |
| `UNKNOWN_TYPE` | "Unknown message type: {type}" | Unsupported message type |
| `NO_SESSION` | "No active conversation session" | Audio sent without active session |
| `PROCESSING_ERROR` | "Audio processing error: {error}" | Error in audio pipeline |

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `500` | Internal server error |
| `503` | Service unavailable |

## Rate Limits

- **WebSocket connections:** 10 per IP address
- **Audio chunks:** No limit (real-time streaming)
- **API requests:** 100 per minute per IP

## Authentication

Currently, the API uses the Cerebras API key for LLM access. No additional authentication is required for the WebSocket connection.

## Examples

### JavaScript Client Example

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Handle connection
ws.onopen = () => {
  console.log('Connected to AI');
  
  // Start conversation
  ws.send(JSON.stringify({
    type: 'start_conversation'
  }));
};

// Handle messages
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch (data.type) {
    case 'transcript':
      console.log('Transcript:', data.text);
      break;
    case 'ai_response':
      console.log('AI Response:', data.text);
      // Play audio response
      playAudio(data.audio_data);
      break;
  }
};

// Send audio chunk
function sendAudioChunk(audioData) {
  ws.send(JSON.stringify({
    type: 'audio_chunk',
    audio_data: audioData
  }));
}
```

### Python Client Example

```python
import asyncio
import websockets
import json
import base64

async def client():
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Start conversation
        await websocket.send(json.dumps({
            "type": "start_conversation"
        }))
        
        # Listen for messages
        async for message in websocket:
            data = json.loads(message)
            
            if data["type"] == "transcript":
                print(f"Transcript: {data['text']}")
            elif data["type"] == "ai_response":
                print(f"AI: {data['text']}")

# Run client
asyncio.run(client())
```

## SDK and Libraries

### Recommended Libraries

**Frontend:**
- `socket.io-client` - WebSocket client
- `recorder.js` - Audio recording
- `web-audio-api` - Audio processing

**Backend:**
- `fastapi` - Web framework
- `websockets` - WebSocket server
- `torch` - AI model inference
- `librosa` - Audio processing

## Support

For API support and questions:
- Check the [troubleshooting guide](deployment.md#troubleshooting)
- Review the [quick start guide](quick-start.md)
- Contact the development team
