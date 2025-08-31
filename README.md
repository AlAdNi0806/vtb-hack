# Real-time Conversational AI Web Application

A real-time voice conversation application with AI using speech-to-text, text-to-speech, and turn detection.

## Architecture

```
arc/
├── frontend/          # Next.js web application
├── backend/           # Python AI service
├── docs/             # Documentation
└── docker/           # Docker configurations
```

## Technology Stack

### Frontend
- **Next.js** - React framework for web application
- **WebSocket** - Real-time communication
- **Web Audio API** - Audio capture and playback

### Backend
- **Python** - AI model interactions and audio processing
- **FastAPI** - Web framework for API endpoints
- **WebSocket** - Real-time communication
- **parakeet-1.1b-rnnt-multilingual-asr** - Speech-to-text
- **Silero TTS** - Text-to-speech
- **LiveKit Turn Detector** - Turn detection
- **Cerebras API** - LLM for conversational responses

## Features

- Real-time voice conversation with AI
- Multilingual speech recognition
- Natural voice synthesis
- Intelligent turn detection
- Modular architecture for scalability

## Quick Start

### Prerequisites
- Node.js 18+ and npm/yarn
- Python 3.9+
- Ubuntu server (for deployment)

### Development Setup

1. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   python main.py
   ```

### Environment Variables

Create `.env` files in both frontend and backend directories:

**Backend (.env)**
```
CEREBRAS_API_KEY=csk-6kjk8yewdwdhkc8ndj5686rcj2te3tfyyr6dw669knd3wy33
```

## Deployment

See [docs/deployment.md](docs/deployment.md) for Ubuntu server deployment instructions.

## API Documentation

- Backend API: http://localhost:8000/docs (FastAPI auto-generated docs)
- WebSocket endpoint: ws://localhost:8000/ws

## License

MIT License
