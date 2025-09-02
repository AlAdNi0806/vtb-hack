import nemo.collections.asr as nemo_asr

# Load a pretrained ASR model (English example, replace with Russian model if needed)
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_ru_fastconformer_hybrid_large_pc")

# Path to your audio file (wav, 16 kHz recommended)
audio_file = "./examples_english_english.wav"

# Transcribe audio file
transcripts = asr_model.transcribe([audio_file])

print("Transcription:")
print(transcripts[0])


# models:
# nvidia/stt_en_fastconformer_transducer_large
# nvidia/stt_ru_fastconformer_hybrid_large_pc - with spaces periods commas and other
# nvidia/stt_ru_conformer_transducer_large