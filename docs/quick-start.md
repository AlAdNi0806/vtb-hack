# Quick Start Guide

This guide will help you get the Conversational AI application running quickly for development and testing.

## Prerequisites

- Python 3.9+
- Node.js 18+
- Git
- Cerebras API key: `csk-6kjk8yewdwdhkc8ndj5686rcj2te3tfyyr6dw669knd3wy33`

## Development Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd arc
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env and add your Cerebras API key

# Test the setup
python run_tests.py

# Start the backend
python main.py
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

Open a new terminal:

```bash
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local if needed (default values should work for local development)

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Testing the Application

1. Open your browser and go to `http://localhost:3000`
2. Click "Connect to AI" to establish WebSocket connection
3. Click "Start Conversation" to begin voice interaction
4. Allow microphone access when prompted
5. Speak into your microphone
6. The AI will respond with both text and voice

## API Endpoints

- **Health Check**: `GET http://localhost:8000/health`
- **WebSocket**: `ws://localhost:8000/ws`
- **API Documentation**: `http://localhost:8000/docs`

## Troubleshooting

### Common Issues

1. **Microphone not working**: Check browser permissions
2. **WebSocket connection fails**: Ensure backend is running on port 8000
3. **Audio not playing**: Check browser audio settings
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode

Enable debug mode by setting `DEBUG=True` in backend `.env` and `NEXT_PUBLIC_DEBUG=true` in frontend `.env.local`.

## Next Steps

- See [deployment.md](deployment.md) for production deployment
- Check [api-reference.md](api-reference.md) for detailed API documentation
- Review [architecture.md](architecture.md) for system design details
