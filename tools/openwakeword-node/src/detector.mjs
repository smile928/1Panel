import { EventEmitter } from 'node:events';
import path from 'node:path';
import { OpenWakeWordModel } from './openwakeword-model.mjs';

/**
 * Streams 16 kHz mono PCM (s16le) through {@link OpenWakeWordModel}.
 * Emits `scores` after each `write()` with the latest per-model scores (same semantics as openwakeword-js `predict`).
 */
export class WakeWordDetector extends EventEmitter {
  /**
   * @param {object} opts
   * @param {string} opts.modelsDir directory containing `melspectrogram.onnx`, `embedding_model.onnx`, optional `silero_vad.onnx`, and wake `.onnx` files
   * @param {string[]} opts.wakewordBasenames model base names without `.onnx` (e.g. `hey_rhasspy_v0.1`) or filenames including `.onnx`
   * @param {number} [opts.vadThreshold=0] Silero VAD gate; `0` disables
   */
  constructor({ modelsDir, wakewordBasenames, vadThreshold = 0 }) {
    super();
    if (!wakewordBasenames?.length) {
      throw new Error('WakeWordDetector requires at least one wake word model basename');
    }
    this.modelsDir = modelsDir;
    this.wakewordBasenames = wakewordBasenames;
    this.vadThreshold = vadThreshold;
    /** @type {OpenWakeWordModel | null} */
    this.model = null;
  }

  async start() {
    if (this.model) throw new Error('WakeWordDetector already started');

    const wakewordModels = this.wakewordBasenames.map((n) =>
      n.endsWith('.onnx') ? path.join(this.modelsDir, n) : path.join(this.modelsDir, `${n}.onnx`),
    );

    this.model = new OpenWakeWordModel({
      wakewordModels,
      melspectrogramModelPath: path.join(this.modelsDir, 'melspectrogram.onnx'),
      embeddingModelPath: path.join(this.modelsDir, 'embedding_model.onnx'),
      vadModelPath: path.join(this.modelsDir, 'silero_vad.onnx'),
      vadThreshold: this.vadThreshold,
      inferenceFramework: 'onnx',
    });
    await this.model.init();
  }

  /**
   * @param {Buffer | Uint8Array} pcmS16leChunk even byte length; odd trailing byte is ignored
   */
  async write(pcmS16leChunk) {
    if (!this.model) throw new Error('WakeWordDetector is not started');
    const buf = Buffer.isBuffer(pcmS16leChunk) ? pcmS16leChunk : Buffer.from(pcmS16leChunk);
    const len = buf.length - (buf.length % 2);
    if (len === 0) return;
    const int16 = new Int16Array(buf.buffer, buf.byteOffset, len / 2);
    const scores = await this.model.predict(int16);
    this.emit('scores', scores);
  }

  reset() {
    this.model?.reset();
  }
}
