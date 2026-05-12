import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const py = process.env.PYTHON ?? 'python3';

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { encoding: 'utf8', cwd: root, ...opts });
  if (r.error) throw r.error;
  if (r.status !== 0) {
    console.error(r.stderr || r.stdout);
    throw new Error(`${cmd} ${args.join(' ')} exited ${r.status}`);
  }
  return r.stdout;
}

console.log('Installing Python dependencies (base wheels)...');
run(py, ['-m', 'pip', 'install', '-r', 'requirements.txt'], { stdio: 'inherit' });

console.log('Installing openWakeWord without tflite-runtime (use ONNX only)...');
run(py, ['-m', 'pip', 'install', 'openwakeword', '--no-deps'], { stdio: 'inherit' });

console.log('Downloading default ONNX models (hey_rhasspy_v0.1 + melspec + embedding + VAD)...');
run(py, ['-c', 'from openwakeword.utils import download_models; download_models(model_names=["hey_rhasspy_v0.1"])'], {
  stdio: 'inherit',
});

console.log('Done. Run: node src/cli.mjs --models hey_rhasspy_v0.1 --wav your_16k_mono.wav');
