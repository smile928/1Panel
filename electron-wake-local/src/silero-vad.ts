import * as ort from "onnxruntime-web";

/**
 * Silero VAD on 1280-sample @16kHz chunks (same layout as openWakeWord-JS runVAD).
 * Used only for TTS barge-in; state is reset when leaving "speaking" mode.
 */
export class SileroChunkVad {
  private session: ort.InferenceSession | null = null;
  private vadStateH = new Float32Array(2 * 1 * 64).fill(0);
  private vadStateC = new Float32Array(2 * 1 * 64).fill(0);
  private readonly sampleRate = 16000;

  async init(modelPath: string, wasmPaths: string): Promise<void> {
    ort.env.wasm.wasmPaths = wasmPaths;
    this.session = await ort.InferenceSession.create(modelPath);
  }

  reset(): void {
    this.vadStateH.fill(0);
    this.vadStateC.fill(0);
  }

  /**
   * @param chunk — PCM in int16 amplitude space (same convention as openWakeWord Model.predict)
   */
  async score(chunk: Float32Array): Promise<number> {
    if (!this.session) throw new Error("SileroChunkVad not initialized");
    if (chunk.length !== 1280) {
      throw new Error(`SileroChunkVad expects 1280 samples, got ${chunk.length}`);
    }
    const normalized = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) normalized[i] = chunk[i] / 32768.0;

    const srTensor = new ort.Tensor("int64", BigInt64Array.from([BigInt(this.sampleRate)]), [1]);
    const hTensor = new ort.Tensor("float32", this.vadStateH, [2, 1, 64]);
    const cTensor = new ort.Tensor("float32", this.vadStateC, [2, 1, 64]);
    const inputTensor = new ort.Tensor("float32", normalized, [1, chunk.length]);

    const feeds: Record<string, ort.Tensor> = {
      [this.session.inputNames[0]]: inputTensor,
      [this.session.inputNames[1]]: srTensor,
      [this.session.inputNames[2]]: hTensor,
      [this.session.inputNames[3]]: cTensor,
    };

    const results = await this.session.run(feeds);
    this.vadStateH = results[this.session.outputNames[1]].data as Float32Array;
    this.vadStateC = results[this.session.outputNames[2]].data as Float32Array;
    return results[this.session.outputNames[0]].data[0] as number;
  }
}
