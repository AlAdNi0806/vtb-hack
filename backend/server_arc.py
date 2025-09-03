"""
Streaming ASR + punctuation + smart end-pointing + Cerebras answer stream
"""
import asyncio, websockets, json, torch, numpy as np, nemo.collections.asr as nemo_asr
import webrtcvad, time, concurrent.futures, re, httpx
from transformers import pipeline          #  <── optional for punctuation

# -------------------- CONFIG --------------------
MODEL_NAME          = "nvidia/stt_ru_conformer_transducer_large"
PUNC_MODEL          = "RUPunct/RUPunct_medium"   # optional
SERVER_HOST         = "0.0.0.0"
SERVER_PORT         = 8765
SAMPLE_RATE         = 16_000
FRAME_DURATION_MS   = 30
VAD_AGGRESSIVENESS  = 2
SILENCE_TIMEOUT     = 1.5
MIN_SPEECH_DURATION = 0.5
PARTIAL_INTERVAL    = 0.40
SENTENCE_END_TOKENS = {".", "?", "!", "…"}          # triggers faster endpoint
CEREBRAS_URL        = "https://api.cerebras.ai/v1/chat/completions"
CEREBRAS_KEY        = "csk-6kjk8yewdwdhkc8ndj5686rcj2te3tfyyr6dw669knd3wy33"        # export or hard-code
SYSTEM_PROMPT       = "You are a helpful assistant. Answer concisely."

print("Loading ASR model…")
asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(MODEL_NAME)

print("Loading punctuation model…")
punctuator = pipeline("token-classification", PUNC_MODEL, aggregation_strategy="none") \
             if PUNC_MODEL else None

pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
vad  = webrtcvad.Vad(VAD_AGGRESSIVENESS)
print("Ready.")
# ------------------------------------------------

def punctuate(text: str) -> str:
    """Add punctuation in ~10 ms; fallback to original if model missing."""
    if not punctuator:
        return text
    try:
        # quick heuristic: if already punctuated, skip
        if any(t in text for t in SENTENCE_END_TOKENS):
            return text
        out = punctuator(text)
        # out is list of dicts with 'word', 'entity' (COMMA, PERIOD, …)
        # we just insert the predicted punctuation after the word
        puncted = []
        for d in out:
            puncted.append(d["word"])
            if d["entity"].endswith("PERIOD"):
                puncted.append(".")
            elif d["entity"].endswith("COMMA"):
                puncted.append(",")
            elif d["entity"].endswith("QUESTION"):
                puncted.append("?")
            elif d["entity"].endswith("EXCLAMATION"):
                puncted.append("!")
        return "".join(puncted).replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    except Exception:
        return text

def _audio_to_np(pcm: bytes) -> np.ndarray:
    pcm_i16 = np.frombuffer(pcm, dtype=np.int16)
    return pcm_i16.astype(np.float32) / 32768.0

def transcribe(pcm: bytes) -> str:
    if not pcm: return ""
    waveform = _audio_to_np(pcm)
    with torch.no_grad():
        hyps = asr_model.transcribe([waveform], batch_size=1)
    text = hyps[0] if isinstance(hyps[0], str) else hyps[0].text
    return text or ""

# ------------------------------------------------
class SmartBuffer:
    def __init__(self):
        self.buffer = bytearray()
        self.speech_start_t = None
        self.last_voice_t   = None
        self.is_speaking    = False
        self.last_text      = ""        # for duplicate partial suppression

    def add(self, data): self.buffer.extend(data)

    def process(self):
        events = []
        while len(self.buffer) >= FRAME_SIZE*2:
            frame = self.buffer[:FRAME_SIZE*2]
            self.buffer = self.buffer[FRAME_SIZE*2:]
            speech = vad.is_speech(frame, SAMPLE_RATE)
            now = time.time()

            if speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start_t = now
                self.last_voice_t = now
            else:
                if self.is_speaking:
                    silence = now - self.last_voice_t
                    if silence > SILENCE_TIMEOUT:
                        self.is_speaking = False
                        if self.last_voice_t - self.speech_start_t >= MIN_SPEECH_DURATION:
                            events.append("speech_end")
                            continue
                    # shorter silence, but sentence end token seen
                    if silence > 0.4 and self.last_text and self.last_text[-1] in SENTENCE_END_TOKENS:
                        events.append("speech_end")
                        continue

            if self.is_speaking:
                events.append(("frame", frame))
        return events

    def clear(self):
        self.buffer.clear()
        self.is_speaking = False
        self.speech_start_t = None
        self.last_voice_t = None
        self.last_text = ""
# ------------------------------------------------
async def cerebras_stream(prompt: str):
    """Yield words from Cerebras chat completion stream."""
    headers = {"Authorization": f"Bearer {CEREBRAS_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": "llama3.1-8b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        "stream": True,
        "max_tokens": 256
    }
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", CEREBRAS_URL, headers=headers, json=payload) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]": break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                        if delta:
                            yield delta
                    except Exception:
                        continue
# ------------------------------------------------
async def recognize(websocket):
    print("Client connected.")
    loop = asyncio.get_running_loop()
    buf   = SmartBuffer()
    speech = bytearray()
    conversation = []   # list of dict {role, text}

    def build_prompt():
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in conversation[-10:]:        # sliding window
            msgs.append(m)
        return msgs

    try:
        async for msg in websocket:
            if isinstance(msg, bytes):
                buf.add(msg)
                for ev in buf.process():
                    if ev == "frame":
                        speech.extend(ev[1])
                    elif ev == "speech_end":
                        if not speech: continue
                        audio = bytes(speech)
                        speech.clear()
                        raw = await loop.run_in_executor(pool, transcribe, audio)
                        text = punctuate(raw).strip()
                        if not text: continue
                        # send final ASR
                        await websocket.send(json.dumps({"type": "asr_final", "text": text}))
                        conversation.append({"role": "user", "content": text})
                        # start LLM stream
                        full_ai = ""
                        async for token in cerebras_stream(text):
                            full_ai += token
                            await websocket.send(json.dumps({"type": "ai_partial", "text": token}))
                        conversation.append({"role": "assistant", "content": full_ai})
                        await websocket.send(json.dumps({"type": "ai_final", "text": full_ai}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print("Client disconnected.")
# ------------------------------------------------
async def main():
    async with websockets.serve(recognize, SERVER_HOST, SERVER_PORT):
        print(f"Server ready at ws://{SERVER_HOST}:{SERVER_PORT}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())