"""
Configuration settings for the Conversational AI Backend
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=True, description="Debug mode")
    
    # API Keys
    CEREBRAS_API_KEY: str = Field(..., description="Cerebras API key for LLM")
    
    # Audio Configuration
    SAMPLE_RATE: int = Field(default=16000, description="Audio sample rate")
    CHUNK_SIZE: int = Field(default=1024, description="Audio chunk size")
    AUDIO_FORMAT: str = Field(default="wav", description="Audio format")
    
    # Model Configuration
    STT_MODEL_PATH: str = Field(
        default="nvidia/parakeet-1.1b-rnnt-multilingual-asr",
        description="Speech-to-text model path"
    )
    TTS_MODEL_PATH: str = Field(
        default="silero_tts",
        description="Text-to-speech model path"
    )
    TURN_DETECTOR_MODEL_PATH: str = Field(
        default="livekit/turn-detector",
        description="Turn detector model path"
    )
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FILE: Optional[str] = Field(default="app.log", description="Log file path")
    
    # Cerebras Configuration
    CEREBRAS_BASE_URL: str = Field(
        default="https://api.cerebras.ai/v1",
        description="Cerebras API base URL"
    )
    CEREBRAS_MODEL: str = Field(
        default="llama3.1-8b",
        description="Cerebras model to use"
    )
    
    # Audio Processing
    VAD_AGGRESSIVENESS: int = Field(
        default=2,
        description="Voice Activity Detection aggressiveness (0-3)"
    )
    SILENCE_THRESHOLD: float = Field(
        default=0.5,
        description="Silence threshold in seconds for turn detection"
    )
    
    # Performance
    MAX_AUDIO_BUFFER_SIZE: int = Field(
        default=10,
        description="Maximum audio buffer size in seconds"
    )
    STT_BATCH_SIZE: int = Field(
        default=1,
        description="STT processing batch size"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()
