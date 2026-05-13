/**
 * Streams mono input chunks to the main thread for wake / VAD.
 */
class PcmTapProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch0 = inputs[0]?.[0];
    if (ch0 && ch0.length) {
      const copy = new Float32Array(ch0.length);
      copy.set(ch0);
      this.port.postMessage({ pcm: copy });
    }
    return true;
  }
}

registerProcessor("pcm-tap-processor", PcmTapProcessor);
