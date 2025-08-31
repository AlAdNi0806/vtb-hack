# Real-time Speech-to-Text Application

A modern web application that provides real-time speech-to-text transcription using Next.js frontend and Python backend with the parakeet-1.1b-rnnt-multilingual-asr model.

## Features

- **Real-time transcription**: Live speech-to-text conversion as you speak
- **Multilingual support**: Powered by parakeet-1.1b-rnnt-multilingual-asr model
- **Modern UI**: Clean, responsive interface built with Next.js and Tailwind CSS
- **WebSocket communication**: Low-latency real-time data streaming
- **Visual feedback**: Recording indicators and connection status
- **Error handling**: Robust error handling for microphone access and connection issues

## Architecture

- **Frontend**: Next.js with TypeScript, Tailwind CSS, and Lucide React icons
- **Backend**: Python FastAPI server with WebSocket support
- **Speech Recognition**: RealtimeSTT library with parakeet model integration
- **Communication**: WebSocket for real-time audio streaming and transcription

## Prerequisites

### Ubuntu/Linux
- Python 3.8+
- Node.js 18+
- npm or yarn

### System Dependencies (Ubuntu)
```bash
sudo apt update
sudo apt install -y python3-dev python3-pip python3-venv portaudio19-dev libasound2-dev libsndfile1-dev ffmpeg git
```

## Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd arc
```

### 2. Backend Setup (Ubuntu)

#### Option A: Automated Setup
```bash
cd backend
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

#### Option B: Manual Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd frontend
npm install
```

## Running the Application

### 1. Start the Backend Server
```bash
cd backend
source venv/bin/activate  # If not already activated
python main.py
```
The backend will start on `http://localhost:8000`

### 2. Start the Frontend Development Server
```bash
cd frontend
npm run dev
```
The frontend will start on `http://localhost:3000`

### 3. Access the Application
Open your browser and navigate to `http://localhost:3000`

## Usage

1. **Grant Microphone Permission**: When prompted, allow microphone access
2. **Check Connection Status**: Ensure both "Microphone Ready" and "Connected" indicators are green
3. **Start Recording**: Click the blue microphone button to start recording
4. **Speak**: Begin speaking - you'll see live transcription appear
5. **Stop Recording**: Click the red button to stop recording
6. **Clear Text**: Use the "Clear" button to reset the transcription

## API Endpoints

### WebSocket
- `ws://localhost:8000/ws/{client_id}` - WebSocket endpoint for real-time communication

### HTTP
- `GET /health` - Health check endpoint

## WebSocket Message Format

### Client to Server
```json
{
  "type": "start_recording"
}
```
```json
{
  "type": "stop_recording"
}
```

### Server to Client
```json
{
  "type": "transcription",
  "text": "transcribed text",
  "is_final": false
}
```
```json
{
  "type": "status",
  "message": "status message"
}
```

## Troubleshooting

### Common Issues

1. **Microphone Permission Denied**
   - Ensure your browser has microphone permissions
   - Check system audio settings

2. **Connection Failed**
   - Verify the backend server is running on port 8000
   - Check firewall settings

3. **No Transcription**
   - Ensure the parakeet model is properly loaded
   - Check backend logs for errors

4. **Audio Quality Issues**
   - Use a good quality microphone
   - Minimize background noise
   - Speak clearly and at a moderate pace

### Backend Logs
Check the backend terminal for detailed error messages and model loading status.

### Browser Console
Open browser developer tools to check for JavaScript errors or WebSocket connection issues.

## Development

### Project Structure
```
arc/
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # App router pages
│   │   └── components/      # React components
│   └── package.json
├── backend/                 # Python backend
│   ├── main.py             # FastAPI server
│   ├── audio_processor.py  # Audio processing logic
│   ├── requirements.txt    # Python dependencies
│   └── setup_ubuntu.sh     # Ubuntu setup script
└── README.md
```

### Adding Features
- Modify `frontend/src/components/SpeechToTextApp.tsx` for UI changes
- Update `backend/audio_processor.py` for audio processing improvements
- Extend `backend/main.py` for new API endpoints

## License

This project is licensed under the MIT License.
