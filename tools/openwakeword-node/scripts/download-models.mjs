import fs from 'node:fs';
import https from 'node:https';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, '..');
const modelsDir = path.join(root, 'models');

const BASE =
  'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/';

const FILES = [
  'melspectrogram.onnx',
  'embedding_model.onnx',
  'silero_vad.onnx',
  'hey_rhasspy_v0.1.onnx',
];

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https
      .get(url, (res) => {
        if (res.statusCode === 301 || res.statusCode === 302) {
          const loc = res.headers.location;
          file.close();
          fs.unlink(dest, () => {});
          if (!loc) return reject(new Error('Redirect without location'));
          return download(loc, dest).then(resolve).catch(reject);
        }
        if (res.statusCode !== 200) {
          file.close();
          fs.unlink(dest, () => {});
          return reject(new Error(`HTTP ${res.statusCode} for ${url}`));
        }
        res.pipe(file);
        file.on('finish', () => file.close(() => resolve()));
      })
      .on('error', (err) => {
        file.close();
        fs.unlink(dest, () => {});
        reject(err);
      });
  });
}

async function main() {
  if (!fs.existsSync(modelsDir)) fs.mkdirSync(modelsDir, { recursive: true });

  for (const name of FILES) {
    const dest = path.join(modelsDir, name);
    if (fs.existsSync(dest) && fs.statSync(dest).size > 1000) {
      console.log(`skip ${name} (exists)`);
      continue;
    }
    const url = BASE + name;
    process.stdout.write(`download ${name}... `);
    await download(url, dest);
    console.log('ok');
  }
  console.log(`Models in ${modelsDir}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
