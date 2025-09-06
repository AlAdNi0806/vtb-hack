import aiohttp
from pipecat.services.piper.tts import PiperTTSService
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner

# Setup session and service (assume Piper server running at http://localhost:5000/api/tts)
session = aiohttp.ClientSession()
tts = PiperTTSService(
    base_url="http://localhost:5000/api/tts",
    aiohttp_session=session,
    voice="ru_RU-diana-medium",  # Russian voice
    sample_rate=22050
)

# Example pipeline for streaming TTS (integrate with STT/LLM for bidirectional)
pipeline = Pipeline([tts])  # Add input/LLM/output as needed
task = PipelineTask(pipeline)
runner = PipelineRunner()

# Run with text input
await task.queue_frame(TextFrame("Это потоковый тест на русском."))
await runner.run(task)