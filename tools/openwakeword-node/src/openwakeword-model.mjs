/**
 * OpenWakeWord-style streaming inference for Node.js (onnxruntime-node).
 * Logic aligned with https://github.com/Firojpaudel/OpenWakeWord_npm_porting (Apache-2.0),
 * models from https://github.com/dscripka/openWakeWord (Apache-2.0).
 */

import * as ort from 'onnxruntime-node';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/** @typedef {'onnx'} InferenceFramework */

/**
 * @typedef {object} ModelOptions
 * @property {string[]} wakewordModels filesystem paths to `.onnx` wake-word classifiers
 * @property {string} melspectrogramModelPath
 * @property {string} embeddingModelPath
 * @property {string} [vadModelPath]
 * @property {number} [vadThreshold] 0 disables VAD
 * @property {Record<string, number>} [patience]
 * @property {Record<string, number>} [thresholds]
 * @property {number} [debounceTime]
 * @property {InferenceFramework} inferenceFramework
 * @property {string} [wasmPaths] ignored in Node (kept for API parity with openwakeword-js)
 */

export class OpenWakeWordModel {
  /** @param {ModelOptions} options */
  constructor(options) {
    this.options = options;
    this.melSession = null;
    this.embeddingSession = null;
    this.vadSession = null;
    /** @type {Map<string, import('onnxruntime-node').InferenceSession>} */
    this.customSessions = new Map();
    /** @type {Map<string, number>} */
    this.embeddingWindowSizes = new Map();

    this.melBuffer = [];
    this.embeddingBuffers = [];
    /** @type {Map<string, number[]>} */
    this.predictionBuffers = new Map();
    this.vadBuffer = [];
    this.rawAudioRemainder = new Float32Array(0);
    this.melContextBuffer = new Float32Array(OpenWakeWordModel.MEL_CONTEXT);

    this.noiseSeededEmbeddings = [];

    this.vadStateH = new Float32Array(2 * 1 * 64).fill(0);
    this.vadStateC = new Float32Array(2 * 1 * 64).fill(0);

    this.isLoaded = false;
  }

  static CHUNK_SIZE = 1280;
  static MEL_CONTEXT = 480;
  static SAMPLE_RATE = 16000;
  static MEL_BINS = 32;
  static FRAMES_PER_CHUNK = 8;
  static MEL_WINDOW_SIZE = 76;
  static EMBEDDING_WINDOW_SIZE = 24;
  static MAX_MEL_FRAMES = 970;
  static INITIAL_FRAMES_SUPPRESS = 5;
  static PREDICTION_BUFFER_MAX = 30;
  static GLOBAL_MAX_EMBEDDING_WINDOW = 50;

  static extractModelName(modelPath) {
    const p = modelPath.startsWith('file:') ? fileURLToPath(modelPath) : modelPath;
    return path.basename(p, path.extname(p));
  }

  async init() {
    try {
      this.melSession = await ort.InferenceSession.create(this.options.melspectrogramModelPath);
      this.embeddingSession = await ort.InferenceSession.create(this.options.embeddingModelPath);

      if (this.options.vadModelPath && this.options.vadThreshold && this.options.vadThreshold > 0) {
        this.vadSession = await ort.InferenceSession.create(this.options.vadModelPath);
      }

      this.melBuffer = Array.from({ length: OpenWakeWordModel.MEL_WINDOW_SIZE }, () =>
        new Float32Array(OpenWakeWordModel.MEL_BINS).fill(1.0),
      );

      const warmNoise = new Float32Array(OpenWakeWordModel.SAMPLE_RATE * 4);
      for (let i = 0; i < warmNoise.length; i++) warmNoise[i] = Math.random() * 2000 - 1000;

      const tempMelContext = new Float32Array(OpenWakeWordModel.MEL_CONTEXT).fill(0);
      const generatedEmbeddings = [];

      for (let i = 0; i <= warmNoise.length - OpenWakeWordModel.CHUNK_SIZE; i += OpenWakeWordModel.CHUNK_SIZE) {
        const chunk = warmNoise.subarray(i, i + OpenWakeWordModel.CHUNK_SIZE);
        const melInput = new Float32Array(OpenWakeWordModel.CHUNK_SIZE + OpenWakeWordModel.MEL_CONTEXT);
        melInput.set(tempMelContext);
        melInput.set(chunk, OpenWakeWordModel.MEL_CONTEXT);
        tempMelContext.set(chunk.subarray(OpenWakeWordModel.CHUNK_SIZE - OpenWakeWordModel.MEL_CONTEXT));

        const melOutput = await this.runMelSpectrogram(melInput);
        for (let f = 0; f < OpenWakeWordModel.FRAMES_PER_CHUNK; f++) {
          const frame = new Float32Array(OpenWakeWordModel.MEL_BINS);
          for (let b = 0; b < OpenWakeWordModel.MEL_BINS; b++) {
            const idx = f * OpenWakeWordModel.MEL_BINS + b;
            frame[b] = melOutput[idx] / 10.0 + 2.0;
          }
          this.melBuffer.push(frame);
        }
        while (this.melBuffer.length > OpenWakeWordModel.MAX_MEL_FRAMES) this.melBuffer.shift();

        const emb = await this.runEmbeddingModel();
        generatedEmbeddings.push(emb);
      }

      this.noiseSeededEmbeddings = generatedEmbeddings
        .slice(-OpenWakeWordModel.GLOBAL_MAX_EMBEDDING_WINDOW)
        .map((e) => new Float32Array(e));
      this.embeddingBuffers = this.noiseSeededEmbeddings.map((e) => new Float32Array(e));

      for (const modelPath of this.options.wakewordModels) {
        const session = await ort.InferenceSession.create(modelPath);
        const name = OpenWakeWordModel.extractModelName(modelPath);
        this.customSessions.set(name, session);

        const inputName = session.inputNames[0];
        let windowSize = 24;

        try {
          const dummyTensor = new ort.Tensor(
            'float32',
            new Float32Array(24 * 96),
            [1, 24, 96],
          );
          await session.run({ [inputName]: dummyTensor });
        } catch (e) {
          const msg = String(e);
          const match = msg.match(/Got: \d+ Expected: (\d+)/);
          if (match) windowSize = parseInt(match[1], 10);
        }

        this.embeddingWindowSizes.set(name, windowSize);
        this.predictionBuffers.set(name, []);
      }

      this.isLoaded = true;
    } catch (error) {
      console.error('Failed to initialize OpenWakeWord models:', error);
      throw error;
    }
  }

  /**
   * @param {Float32Array | Int16Array} audio
   * @returns {Promise<Record<string, number>>}
   */
  async predict(audio) {
    if (!this.isLoaded) throw new Error('Model not initialized');

    let pcmAudio;
    if (audio instanceof Int16Array) {
      pcmAudio = new Float32Array(audio.length);
      for (let i = 0; i < audio.length; i++) pcmAudio[i] = audio[i];
    } else {
      let max = 0;
      for (let i = 0; i < Math.min(audio.length, 1000); i++) {
        const abs = Math.abs(audio[i]);
        if (abs > max) max = abs;
      }
      if (max <= 1.0) {
        pcmAudio = new Float32Array(audio.length);
        for (let i = 0; i < audio.length; i++) pcmAudio[i] = audio[i] * 32768.0;
      } else {
        pcmAudio = audio;
      }
    }

    const combinedAudio = new Float32Array(this.rawAudioRemainder.length + pcmAudio.length);
    combinedAudio.set(this.rawAudioRemainder);
    combinedAudio.set(pcmAudio, this.rawAudioRemainder.length);

    /** @type {Record<string, number>} */
    const scores = {};
    for (const name of this.customSessions.keys()) scores[name] = 0.0;

    let offset = 0;
    while (offset + OpenWakeWordModel.CHUNK_SIZE <= combinedAudio.length) {
      const chunk = combinedAudio.subarray(offset, offset + OpenWakeWordModel.CHUNK_SIZE);
      offset += OpenWakeWordModel.CHUNK_SIZE;

      const melInput = new Float32Array(OpenWakeWordModel.CHUNK_SIZE + OpenWakeWordModel.MEL_CONTEXT);
      melInput.set(this.melContextBuffer);
      melInput.set(chunk, OpenWakeWordModel.MEL_CONTEXT);
      this.melContextBuffer.set(chunk.subarray(OpenWakeWordModel.CHUNK_SIZE - OpenWakeWordModel.MEL_CONTEXT));

      if (this.vadSession && this.options.vadThreshold) {
        const vadScore = await this.runVAD(chunk);
        this.vadBuffer.push(vadScore);
        while (this.vadBuffer.length > 30) this.vadBuffer.shift();
      }

      const melOutput = await this.runMelSpectrogram(melInput);
      for (let f = 0; f < OpenWakeWordModel.FRAMES_PER_CHUNK; f++) {
        const frame = new Float32Array(OpenWakeWordModel.MEL_BINS);
        for (let b = 0; b < OpenWakeWordModel.MEL_BINS; b++) {
          const idx = f * OpenWakeWordModel.MEL_BINS + b;
          frame[b] = melOutput[idx] / 10.0 + 2.0;
        }
        this.melBuffer.push(frame);
      }
      while (this.melBuffer.length > OpenWakeWordModel.MAX_MEL_FRAMES) this.melBuffer.shift();

      const embedding = await this.runEmbeddingModel();
      this.embeddingBuffers.push(embedding);
      while (this.embeddingBuffers.length > OpenWakeWordModel.GLOBAL_MAX_EMBEDDING_WINDOW) {
        this.embeddingBuffers.shift();
      }

      for (const [name, session] of this.customSessions.entries()) {
        const windowSize = this.embeddingWindowSizes.get(name) ?? 24;

        let score = await this.runClassifier(name, session, windowSize);

        if (this.vadSession && this.options.vadThreshold) {
          const window = this.vadBuffer.slice(-7, -4);
          const maxVAD = window.length > 0 ? Math.max(...window) : 0;
          if (maxVAD < this.options.vadThreshold) score = 0.0;
        }

        const predBuf = this.predictionBuffers.get(name);
        predBuf.push(score);
        while (predBuf.length > OpenWakeWordModel.PREDICTION_BUFFER_MAX) predBuf.shift();

        if (predBuf.length < OpenWakeWordModel.INITIAL_FRAMES_SUPPRESS) {
          score = 0.0;
        } else if (
          (this.options.patience && this.options.patience[name]) ||
          (this.options.debounceTime && this.options.debounceTime > 0)
        ) {
          const threshold = this.options.thresholds?.[name] ?? 0.5;
          if (this.options.patience?.[name]) {
            const p = this.options.patience[name];
            const recentScores = predBuf.slice(-p);
            const countAbove = recentScores.filter((s) => s >= threshold).length;
            if (countAbove < p) score = 0.0;
          } else if (this.options.debounceTime) {
            const framesToWait = Math.ceil(this.options.debounceTime / 0.08);
            const recentScores = predBuf.slice(-framesToWait - 1, -1);
            const alreadyTriggered = recentScores.some((s) => s >= threshold);
            if (score >= threshold && alreadyTriggered) score = 0.0;
          }
        }
        scores[name] = Math.max(scores[name], score);
      }
    }

    this.rawAudioRemainder = combinedAudio.slice(offset);
    return scores;
  }

  /** @param {Float32Array} input */
  async runMelSpectrogram(input) {
    const inputTensor = new ort.Tensor('float32', input, [1, input.length]);
    const results = await this.melSession.run({ [this.melSession.inputNames[0]]: inputTensor });
    return /** @type {Float32Array} */ (results[this.melSession.outputNames[0]].data);
  }

  async runEmbeddingModel() {
    const windowData = new Float32Array(OpenWakeWordModel.MEL_WINDOW_SIZE * OpenWakeWordModel.MEL_BINS);
    const startIdx = this.melBuffer.length - OpenWakeWordModel.MEL_WINDOW_SIZE;
    for (let t = 0; t < OpenWakeWordModel.MEL_WINDOW_SIZE; t++) {
      windowData.set(this.melBuffer[startIdx + t], t * OpenWakeWordModel.MEL_BINS);
    }
    const windowTensor = new ort.Tensor('float32', windowData, [
      1,
      OpenWakeWordModel.MEL_WINDOW_SIZE,
      OpenWakeWordModel.MEL_BINS,
      1,
    ]);
    const results = await this.embeddingSession.run({
      [this.embeddingSession.inputNames[0]]: windowTensor,
    });
    const output = /** @type {Float32Array} */ (results[this.embeddingSession.outputNames[0]].data);
    const embedding = new Float32Array(96);
    for (let i = 0; i < 96; i++) {
      let v = output[i] ?? 0;
      if (Number.isNaN(v) || !Number.isFinite(v)) v = 0;
      embedding[i] = v;
    }
    return embedding;
  }

  /**
   * @param {string} name
   * @param {import('onnxruntime-node').InferenceSession} session
   * @param {number} windowSize
   */
  async runClassifier(name, session, windowSize) {
    const predData = new Float32Array(windowSize * 96);
    const startIdx = this.embeddingBuffers.length - windowSize;

    for (let t = 0; t < windowSize; t++) {
      predData.set(this.embeddingBuffers[startIdx + t], t * 96);
    }

    const predTensor = new ort.Tensor('float32', predData, [1, windowSize, 96]);
    const results = await session.run({ [session.inputNames[0]]: predTensor });
    const out = results[session.outputNames[0]].data;
    return /** @type {number} */ (out[0]);
  }

  /** @param {Float32Array} chunk */
  async runVAD(chunk) {
    const normalized = new Float32Array(chunk.length);
    for (let i = 0; i < chunk.length; i++) normalized[i] = chunk[i] / 32768.0;
    const srTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(OpenWakeWordModel.SAMPLE_RATE)]), [1]);
    const hTensor = new ort.Tensor('float32', this.vadStateH, [2, 1, 64]);
    const cTensor = new ort.Tensor('float32', this.vadStateC, [2, 1, 64]);
    const inputTensor = new ort.Tensor('float32', normalized, [1, chunk.length]);
    const feeds = {
      [this.vadSession.inputNames[0]]: inputTensor,
      [this.vadSession.inputNames[1]]: srTensor,
      [this.vadSession.inputNames[2]]: hTensor,
      [this.vadSession.inputNames[3]]: cTensor,
    };

    const results = await this.vadSession.run(feeds);
    this.vadStateH = /** @type {Float32Array} */ (results[this.vadSession.outputNames[1]].data);
    this.vadStateC = /** @type {Float32Array} */ (results[this.vadSession.outputNames[2]].data);
    return /** @type {number} */ (results[this.vadSession.outputNames[0]].data[0]);
  }

  reset() {
    this.melBuffer = Array.from({ length: OpenWakeWordModel.MEL_WINDOW_SIZE }, () =>
      new Float32Array(OpenWakeWordModel.MEL_BINS).fill(1.0),
    );
    this.rawAudioRemainder = new Float32Array(0);
    this.melContextBuffer.fill(0);
    this.vadBuffer = [];
    this.vadStateH.fill(0);
    this.vadStateC.fill(0);
    this.embeddingBuffers = this.noiseSeededEmbeddings.map((e) => new Float32Array(e));
    for (const name of this.customSessions.keys()) {
      this.predictionBuffers.set(name, []);
    }
  }
}
