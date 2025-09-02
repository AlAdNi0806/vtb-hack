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
    this.buffer = new Float32Array(0);
    this.frameSize = 1024; // Process in smaller chunks for better real-time performance
    
    console.log(`AudioWorklet: Mic SR=${sampleRate}, Target SR=${this.targetRate}, Ratio=${this.downsampleRatio}`);
  }

  /**
   * Called by the browser's audio engine with new audio data.
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0][0]; // Input is Float32Array PCM data

    if (!input) {
      return true; // Keep processor alive
    }

    // Append new data to buffer
    const newBuffer = new Float32Array(this.buffer.length + input.length);
    newBuffer.set(this.buffer);
    newBuffer.set(input, this.buffer.length);
    this.buffer = newBuffer;

    // Process in chunks to maintain real-time performance
    while (this.buffer.length >= this.frameSize) {
      const frame = this.buffer.slice(0, this.frameSize);
      this.buffer = this.buffer.slice(this.frameSize);
      
      // Downsample the frame
      const downsampledLen = Math.floor(frame.length / this.downsampleRatio);
      const pcm16 = new Int16Array(downsampledLen);

      for (let i = 0; i < downsampledLen; i++) {
        const srcIndex = Math.floor(i * this.downsampleRatio);
        if (srcIndex < frame.length) {
          // Convert to 16-bit integer with proper scaling
          pcm16[i] = Math.max(-32768, Math.min(32767, frame[srcIndex] * 32768));
        }
      }
      
      // Send to main thread
      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }

    return true; // Keep processor alive
  }
}

registerProcessor('resampler-processor', ResamplerProcessor);