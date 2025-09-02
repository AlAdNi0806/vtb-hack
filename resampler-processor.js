// resampler-processor.js

/**
 * An AudioWorkletProcessor for downsampling audio to a target sample rate.
 * It converts the audio to 16-bit PCM format and sends it to the main thread.
 */
class ResamplerProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // The target sample rate is passed from the main thread
    this.targetRate = options.processorOptions.targetRate;
    this.downsampleRatio = sampleRate / this.targetRate; // sampleRate is a global in AudioWorklet
    console.log(`AudioWorklet: Mic SR=${sampleRate}, Target SR=${this.targetRate}`);
  }

  /**
   * Called by the browser's audio engine with new audio data.
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0][0]; // Input is Float32Array PCM data

    if (!input) {
      return true; // Keep processor alive
    }

    const downsampledLen = Math.floor(input.length / this.downsampleRatio);
    const pcm16 = new Int16Array(downsampledLen);

    for (let i = 0; i < downsampledLen; i++) {
      const srcIndex = Math.floor(i * this.downsampleRatio);
      // Clamp value between -1 and 1, then convert to 16-bit integer
      pcm16[i] = Math.max(-1, Math.min(1, input[srcIndex])) * 0x7FFF;
    }
    
    // Send the processed 16-bit PCM audio buffer back to the main thread.
    // The second argument [pcm16.buffer] is a "transferable object",
    // which efficiently transfers ownership without copying data.
    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);

    return true; // Indicate that the processor should continue running
  }
}

registerProcessor('resampler-processor', ResamplerProcessor);