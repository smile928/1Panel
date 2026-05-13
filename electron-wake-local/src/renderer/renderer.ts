import { Model } from "openwakeword-js";
import * as ort from "onnxruntime-web";
import { SileroChunkVad } from "../silero-vad";

declare global {
  interface Window {
    wakeBridge: {
      getModelPaths: () => Promise<{
        modelsDir: string;
        melspectrogram: string;
        embedding: string;
        sileroVad: string;
        wakeword: string;
        wasmDir: string;
      }>;
      onStopTts: (cb: () => void) => () => void;
      notifyWake: () => void;
      notifyBargeIn: () => void;
    };
  }
}

const WAKE_MODEL_KEY = "xiaolanghua";
const CHUNK = 1280;
const SAMPLE_RATE = 16000;

type Mode = "idle" | "speaking" | "cooldown";

function el(id: string): HTMLElement {
  const e = document.getElementById(id);
  if (!e) throw new Error(`Missing #${id}`);
  return e;
}

function logLine(text: string): void {
  const box = el("log") as HTMLPreElement;
  box.textContent = `${new Date().toISOString().slice(11, 23)} ${text}\n${box.textContent ?? ""}`.slice(0, 8000);
}

function floatToPcm16Scale(channel: Float32Array): Float32Array {
  const out = new Float32Array(channel.length);
  for (let i = 0; i < channel.length; i++) {
    const s = Math.max(-1, Math.min(1, channel[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

/** Simple polyphase-ish resampler: nativeRate → 16k mono */
function resampleTo16k(input: Float32Array, nativeRate: number): Float32Array {
  if (nativeRate === SAMPLE_RATE) return input;
  const ratio = nativeRate / SAMPLE_RATE;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcPos = i * ratio;
    const j = Math.floor(srcPos);
    const frac = srcPos - j;
    const a = input[j] ?? 0;
    const b = input[j + 1] ?? a;
    out[i] = a + frac * (b - a);
  }
  return out;
}

async function main(): Promise<void> {
  const paths = await window.wakeBridge.getModelPaths();
  const wasmPath = paths.wasmDir.replace(/\/?$/, "/");

  ort.env.wasm.wasmPaths = wasmPath;

  const wakeModel = new Model({
    wakewordModels: [paths.wakeword],
    melspectrogramModelPath: paths.melspectrogram,
    embeddingModelPath: paths.embedding,
    vadModelPath: paths.sileroVad,
    vadThreshold: 0.45,
    thresholds: { [WAKE_MODEL_KEY]: 0.55 },
    debounceTime: 1.0,
    inferenceFramework: "onnx",
    wasmPaths: wasmPath,
  });

  const bargeVad = new SileroChunkVad();
  await bargeVad.init(paths.sileroVad, wasmPath);
  await wakeModel.init();

  let mode: Mode = "idle";
  let pcmCarry = new Float32Array(0);
  let vadSpeechStreak = 0;
  let vadSilenceStreak = 0;
  const BARGE_VAD_THRESHOLD = 0.55;
  const BARGE_STREAK_FRAMES = 2;
  const COOLDOWN_MS = 450;

  const status = el("status");
  const scoreWake = el("score-wake") as HTMLSpanElement;
  const scoreVad = el("score-vad") as HTMLSpanElement;

  function setMode(m: Mode): void {
    mode = m;
    status.textContent =
      m === "idle"
        ? "空闲：监听唤醒「小浪花，小浪花」"
        : m === "speaking"
          ? "播放中：VAD 打断已开启"
          : "冷却中…";
    if (m === "speaking") {
      wakeModel.reset();
      bargeVad.reset();
      vadSpeechStreak = 0;
      vadSilenceStreak = 0;
    }
    if (m === "idle") {
      wakeModel.reset();
      bargeVad.reset();
      vadSpeechStreak = 0;
      vadSilenceStreak = 0;
    }
  }

  window.wakeBridge.onStopTts(() => {
    speechSynthesis.cancel();
    logLine("已打断 TTS（VAD）");
    setMode("cooldown");
    setTimeout(() => setMode("idle"), COOLDOWN_MS);
  });

  (el("btn-tts") as HTMLButtonElement).onclick = () => {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(
      "这是一段用于测试打断的语音。你可以随时说话，系统会用静音检测打断播放。小浪花，小浪花。"
    );
    u.lang = "zh-CN";
    u.onstart = () => {
      setMode("speaking");
      logLine("TTS 开始");
    };
    u.onend = () => {
      logLine("TTS 结束");
      setMode("cooldown");
      setTimeout(() => setMode("idle"), COOLDOWN_MS);
    };
    speechSynthesis.speak(u);
  };

  (el("btn-mic") as HTMLButtonElement).onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const ctx = new AudioContext();
    const nativeRate = ctx.sampleRate;
    logLine(`AudioContext sampleRate=${nativeRate}`);

    await ctx.audioWorklet.addModule(new URL("pcm-tap-processor.js", import.meta.url).href);

    const source = ctx.createMediaStreamSource(stream);
    const tap = new AudioWorkletNode(ctx, "pcm-tap-processor");

    tap.port.onmessage = async (ev: MessageEvent<{ pcm: Float32Array }>) => {
      const chunk = ev.data.pcm;
      if (!chunk?.length) return;

      let pcm = floatToPcm16Scale(chunk);
      pcm = resampleTo16k(pcm, nativeRate);

      const merged = new Float32Array(pcmCarry.length + pcm.length);
      merged.set(pcmCarry);
      merged.set(pcm, pcmCarry.length);
      pcmCarry = merged;

      while (pcmCarry.length >= CHUNK) {
        const frame = pcmCarry.subarray(0, CHUNK);
        pcmCarry = pcmCarry.slice(CHUNK);

        if (mode === "idle") {
          const scores = await wakeModel.predict(frame);
          const s = scores[WAKE_MODEL_KEY] ?? 0;
          scoreWake.textContent = s.toFixed(3);
          scoreVad.textContent = "—";
          if (s >= 0.55) {
            logLine(`唤醒命中 score=${s.toFixed(3)}`);
            window.wakeBridge.notifyWake();
          }
        } else if (mode === "speaking") {
          const v = await bargeVad.score(frame);
          scoreWake.textContent = "—";
          scoreVad.textContent = v.toFixed(3);
          if (v >= BARGE_VAD_THRESHOLD) {
            vadSpeechStreak++;
            vadSilenceStreak = 0;
            if (vadSpeechStreak >= BARGE_STREAK_FRAMES) {
              vadSpeechStreak = 0;
              speechSynthesis.cancel();
              window.wakeBridge.notifyBargeIn();
            }
          } else {
            vadSilenceStreak++;
            if (vadSilenceStreak >= 3) vadSpeechStreak = 0;
          }
        } else {
          scoreWake.textContent = "—";
          scoreVad.textContent = "—";
        }
      }
    };

    source.connect(tap);
    tap.connect(ctx.destination);
    logLine("麦克风已启动");
  };

  setMode("idle");
  logLine("模型就绪。先点「启动麦克风」，再点「播放测试 TTS」体验打断。");
}

main().catch((e) => {
  console.error(e);
  const pre = document.getElementById("log");
  if (pre) pre.textContent = String(e);
});
