import nemo.collections.asr as nemo_asr
import time 

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="nvidia/parakeet-rnnt-1.1b",
    map_location="cpu"
)

audio_file = './examples_english_english.wav'

_ = asr_model.transcribe([audio_file])

start = time.time()
transcription = asr_model.transcribe([audio_file])
end = time.time()

print("Transcript:", transcription[0])
print(f"Time taken: {end - start:.2f} seconds")
