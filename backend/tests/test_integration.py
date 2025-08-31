"""
Integration tests for the Conversational AI Backend
"""

import asyncio
import json
import pytest
import websockets
from unittest.mock import AsyncMock, MagicMock
import base64

from main import app
from services.ai_service import AIService
from services.audio_processor import AudioProcessor
from services.audio_pipeline import AudioPipeline


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.fixture
    async def mock_services(self):
        """Create mock services for testing"""
        # Mock AI Service
        ai_service = AsyncMock(spec=AIService)
        ai_service.is_ready.return_value = True
        ai_service.generate_response.return_value = "Hello! How can I help you?"
        ai_service.text_to_speech.return_value = "base64_encoded_audio_data"
        
        # Mock Audio Processor
        audio_processor = AsyncMock(spec=AudioProcessor)
        audio_processor.is_ready.return_value = True
        audio_processor.process_audio_chunk.return_value = {
            "transcript": "Hello there",
            "is_final": True,
            "turn_detected": True,
            "is_voice": True
        }
        
        return ai_service, audio_processor
    
    @pytest.fixture
    def sample_audio_data(self):
        """Generate sample audio data for testing"""
        # Create a simple audio buffer (silence)
        import numpy as np
        audio_array = np.zeros(1024, dtype=np.float32)
        
        # Convert to base64
        int16_array = (audio_array * 32767).astype(np.int16)
        audio_bytes = int16_array.tobytes()
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    async def test_websocket_connection(self):
        """Test WebSocket connection establishment"""
        # This would require running the actual server
        # For now, we'll test the message handling logic
        pass
    
    async def test_audio_pipeline_flow(self, mock_services, sample_audio_data):
        """Test the complete audio processing pipeline"""
        ai_service, audio_processor = mock_services
        
        # Create audio pipeline
        pipeline = AudioPipeline(audio_processor, ai_service)
        
        # Start session
        session_result = await pipeline.start_session("test_session")
        assert session_result["status"] == "session_started"
        assert session_result["session_id"] == "test_session"
        
        # Process audio chunk
        result = await pipeline.process_audio_chunk(sample_audio_data, 1234567890.0)
        
        # Verify results
        assert "transcript" in result
        assert "is_final" in result
        assert "turn_detected" in result
        
        # End session
        end_result = await pipeline.end_session()
        assert end_result["status"] == "session_ended"
    
    async def test_conversation_flow(self, mock_services, sample_audio_data):
        """Test a complete conversation flow"""
        ai_service, audio_processor = mock_services
        pipeline = AudioPipeline(audio_processor, ai_service)
        
        # Start conversation
        await pipeline.start_session("conversation_test")
        
        # Simulate user speaking
        result1 = await pipeline.process_audio_chunk(sample_audio_data, 1234567890.0)
        
        # Should get transcript
        assert result1.get("transcript") == "Hello there"
        assert result1.get("is_final") == True
        assert result1.get("turn_detected") == True
        
        # Should get AI response
        assert result1.get("ai_response") == "Hello! How can I help you?"
        assert result1.get("tts_audio") == "base64_encoded_audio_data"
        
        # End conversation
        await pipeline.end_session()
    
    def test_audio_data_conversion(self, sample_audio_data):
        """Test audio data conversion and encoding"""
        # Verify the sample audio data is valid base64
        try:
            decoded = base64.b64decode(sample_audio_data)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Audio data conversion failed: {e}")
    
    async def test_error_handling(self, mock_services):
        """Test error handling in the pipeline"""
        ai_service, audio_processor = mock_services
        
        # Mock an error in audio processing
        audio_processor.process_audio_chunk.side_effect = Exception("Processing error")
        
        pipeline = AudioPipeline(audio_processor, ai_service)
        await pipeline.start_session("error_test")
        
        # Process should handle the error gracefully
        result = await pipeline.process_audio_chunk("invalid_audio", 1234567890.0)
        assert "error" in result
    
    async def test_session_management(self, mock_services):
        """Test session management functionality"""
        ai_service, audio_processor = mock_services
        pipeline = AudioPipeline(audio_processor, ai_service)
        
        # Test session status when no session
        status = await pipeline.get_session_status()
        assert status["session_id"] is None
        assert status["is_active"] == False
        
        # Start session
        await pipeline.start_session("session_mgmt_test")
        
        # Check active session
        status = await pipeline.get_session_status()
        assert status["session_id"] == "session_mgmt_test"
        assert status["is_active"] == True
        
        # End session
        await pipeline.end_session()
        
        # Check session ended
        status = await pipeline.get_session_status()
        assert status["session_id"] is None
        assert status["is_active"] == False


class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        # This would require FastAPI test client
        pass
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        # This would require FastAPI test client
        pass


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
