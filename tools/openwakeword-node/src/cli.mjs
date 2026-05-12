#!/usr/bin/env node
/**
 * CLI for streaming openWakeWord scores from Node.js via a Python ONNX worker.
 *
 * Examples:
 *   npm run setup
 *   node src/cli.mjs --models hey_rhasspy_v0.1 --wav recording.wav
 *   ffmpeg -f alsa -i default -f s16le -ac 1 -ar 16000 - | node src/cli.mjs --models hey_rhasspy_v0.1 --stdin
 */

import { WakeWordDetector } from './detector.mjs';
import { readWavPcmS16le, stereoToMonoS16le } from './wav-pcm.mjs';

function parseArgs(argv) {
  const out = { models: [], vad: 0, trigger: 0.5, wav: null, stdin: false, mic: false };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--models' && argv[i + 1]) out.models = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (a === '--vad' && argv[i + 1]) out.vad = Number(argv[++i]);
    else if (a === '--trigger' && argv[i + 1]) out.trigger = Number(argv[++i]);
    else if (a === '--wav' && argv[i + 1]) out.wav = argv[++i];
    else if (a === '--stdin') out.stdin = true;
    else if (a === '--mic') out.mic = true;
  }
  return out;
}

function pcmMustBe16kMono(pcm, sampleRate, channels) {
  if (sampleRate !== 16000) {
    throw new Error(`Expected 16 kHz WAV, got ${sampleRate} Hz. Convert with: ffmpeg -i in.wav -ac 1 -ar 16000 out.wav`);
  }
  if (channels === 1) return pcm;
  if (channels === 2) return stereoToMonoS16le(pcm);
  throw new Error(`Expected mono or stereo WAV, got ${channels} channels`);
}

async function pumpStdin(det) {
  for await (const chunk of process.stdin) {
    det.write(chunk);
  }
  det.end();
}

function pumpBuffer(det, buf) {
  const chunk = 4096;
  for (let i = 0; i < buf.length; i += chunk) {
    det.write(buf.subarray(i, Math.min(i + chunk, buf.length)));
  }
  det.end();
}

async function main() {
  const opts = parseArgs(process.argv);
  if (!opts.models.length) {
    console.error(
      'usage: node src/cli.mjs --models <name[,name2]> [--vad 0.5] [--trigger 0.6] (--wav file.wav | --stdin | --mic)',
    );
    process.exit(2);
  }

  const det = new WakeWordDetector({
    models: opts.models,
    vadThreshold: opts.vad,
  });

  det.on('scores', (scores) => {
    const parts = Object.entries(scores)
      .map(([k, v]) => `${k}=${Number(v).toFixed(3)}`)
      .join(' ');
    process.stdout.write(`${parts}\n`);
    for (const [name, v] of Object.entries(scores)) {
      if (v >= opts.trigger) {
        process.stderr.write(`[trigger] ${name} score=${v.toFixed(3)} (threshold ${opts.trigger})\n`);
      }
    }
  });

  det.on('error', (err) => {
    console.error(err);
  });

  det.start();

  if (opts.wav) {
    const { sampleRate, channels, pcm } = readWavPcmS16le(opts.wav);
    const mono = pcmMustBe16kMono(pcm, sampleRate, channels);
    pumpBuffer(det, mono);
  } else if (opts.stdin || !process.stdin.isTTY) {
    await pumpStdin(det);
  } else if (opts.mic) {
    const { spawn } = await import('node:child_process');
    const device = process.env.MIC_DEVICE ?? 'default';
    const ff = spawn(
      'ffmpeg',
      [
        '-hide_banner',
        '-loglevel',
        'error',
        '-f',
        'alsa',
        '-i',
        device,
        '-f',
        's16le',
        '-ac',
        '1',
        '-ar',
        '16000',
        '-',
      ],
      { stdio: ['ignore', 'pipe', 'inherit'] },
    );
    ff.on('error', (e) => {
      console.error(e.message);
      console.error('Tip: install ffmpeg + ALSA, or use --stdin and pipe audio yourself.');
      det.kill();
      process.exit(1);
    });
    for await (const chunk of ff.stdout) {
      det.write(chunk);
    }
    det.end();
  } else {
    console.error('Provide --wav, --stdin, or --mic');
    det.kill();
    process.exit(2);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
