import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

# Load pretrained ASR model (example English model)
asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_transducer_large")

# Update decoding config to enable punctuation and timestamp computation
decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.preserve_alignments = True
    decoding_cfg.compute_timestamps = True
    decoding_cfg.segment_seperators = [".", "?", "!"]
    decoding_cfg.word_seperator = " "

asr_model.change_decoding_strategy(decoding_cfg)

# Transcribe audio with timestamps and punctuation
hypotheses = asr_model.transcribe(["./examples_english_english.wav"], return_hypotheses=True)

# For RNNT models, extract best hypotheses if needed
if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
    hypotheses = hypotheses[0]

# Get word and segment timestamps
word_timestamps = hypotheses[0].timestep['word']
segment_timestamps = hypotheses[0].timestep['segment']

time_stride = 8 * asr_model.cfg.preprocessor.window_stride  # duration of one timestep in seconds

print("Transcription with punctuation and timestamps:")
for stamp in segment_timestamps:
    start = stamp['start_offset'] * time_stride
    end = stamp['end_offset'] * time_stride
    segment = stamp['segment']
    print(f"{start:.2f}s - {end:.2f}s: {segment}")

print("\nWords with timestamps:")
for stamp in word_timestamps:
    start = stamp['start_offset'] * time_stride
    end = stamp['end_offset'] * time_stride
    word = stamp.get('word', '')
    print(f"{start:.2f}s - {end:.2f}s: {word}")
