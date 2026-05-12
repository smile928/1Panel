import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { EventEmitter } from 'node:events';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PYTHON_SCRIPT = path.join(__dirname, '..', 'python', 'stream_stdin.py');

/**
 * Streams 16 kHz mono PCM (Int16) to a Python openWakeWord (ONNX) worker.
 * Emits {@link WakeWordDetectorEventMap} `scores` for every 80 ms frame.
 */
export class WakeWordDetector extends EventEmitter {
  /** @type {import('node:child_process').ChildProcess | null} */
  #child = null;

  /**
   * @param {object} opts
   * @param {string[]} opts.models openWakeWord model ids (e.g. `hey_rhasspy_v0.1`) or paths to `.onnx`
   * @param {number} [opts.vadThreshold=0] Silero VAD gate; `0` disables
   * @param {string} [opts.pythonPath] defaults to `process.env.PYTHON` or `python3`
   */
  constructor({ models, vadThreshold = 0, pythonPath = process.env.PYTHON ?? 'python3' }) {
    super();
    if (!models?.length) {
      throw new Error('WakeWordDetector requires at least one model name or path');
    }
    this.models = models;
    this.vadThreshold = vadThreshold;
    this.pythonPath = pythonPath;
  }

  start() {
    if (this.#child) {
      throw new Error('WakeWordDetector already started');
    }
    const args = [PYTHON_SCRIPT, this.models.join(','), String(this.vadThreshold)];
    this.#child = spawn(this.pythonPath, args, {
      stdio: ['pipe', 'pipe', 'inherit'],
    });

    this.#child.on('error', (err) => this.emit('error', err));
    this.#child.on('exit', (code, signal) => {
      this.emit('exit', code, signal);
      this.#child = null;
    });

    const rl = createInterface({ input: this.#child.stdout });
    rl.on('line', (line) => {
      try {
        const row = JSON.parse(line);
        if (row.scores && typeof row.scores === 'object') {
          this.emit('scores', row.scores);
        }
      } catch (e) {
        this.emit('error', e);
      }
    });
  }

  /**
   * @param {Buffer | Uint8Array} pcmS16leChunk raw s16le bytes (any length; caller may split for lower latency)
   */
  write(pcmS16leChunk) {
    if (!this.#child?.stdin?.writable) {
      throw new Error('WakeWordDetector is not running');
    }
    this.#child.stdin.write(pcmS16leChunk);
  }

  end() {
    this.#child?.stdin?.end();
  }

  kill(signal = 'SIGTERM') {
    this.#child?.kill(signal);
    this.#child = null;
  }
}
