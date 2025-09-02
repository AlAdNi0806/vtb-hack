import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

asr_model = nemo_asr.models.ASRModel.from_pretrained("nvidia/stt_en_fastconformer_transducer_large")

decoding_cfg = asr_model.cfg.decoding
with open_dict(decoding_cfg):
    decoding_cfg.preserve_alignments = True
    decoding_cfg.compute_timestamps = True
    # Specify segment separators correctly (use strings, e.g. ".", "?", "!")
    decoding_cfg.segment_separators = [".", "?", "!"]

asr_model.change_decoding_strategy(decoding_cfg)

hypotheses = asr_model.transcribe(["./rec.wav"], return_hypotheses=True)

word_timestamps = hypotheses[0].timestamp['word']
segment_timestamps = hypotheses[0].timestamp['segment']

time_stride = 8 * asr_model.cfg.preprocessor.window_stride

print("Segments:")
for seg in segment_timestamps:
    start = seg['start_offset'] * time_stride
    end = seg['end_offset'] * time_stride
    print(f"{start:.2f} - {end:.2f}: {seg['segment']}")

print("Words:")
for word in word_timestamps:
    start = word['start_offset'] * time_stride
    end = word['end_offset'] * time_stride
    print(f"{start:.2f} - {end:.2f}: {word['word']}")
