# Wake model: 「小浪花，小浪花」

This app loads **`xiaolanghua.onnx`** — an openWakeWord classifier trained for your phrase.

## First-time setup

From `electron-wake-local/`:

```bash
npm install
npm run setup
npm start
```

`setup` runs `openwakeword-js-setup` (downloads ONNX + ORT WASM into `models/`) and `prepare-wake`, which creates `xiaolanghua.onnx` from the bundled sample **only if missing**. That sample is **not** trained on「小浪花，小浪花」; replace the file with your own export before shipping.

## Train a real model (openWakeWord)

Use the upstream [openWakeWord](https://github.com/dscripka/openWakeWord) training pipeline (see `examples/custom_model.yml` and project docs). Set your target phrase to **小浪花，小浪花** (or the exact wording you want). Export ONNX and overwrite:

`electron-wake-local/models/xiaolanghua.onnx`

The JS port expects the same ONNX format as the Python project ([openwakeword-js](https://www.npmjs.com/package/openwakeword-js)).

## Bundled base assets

These are downloaded by `openwakeword-js-setup` and are **not** phrase-specific:

- `melspectrogram.onnx`
- `embedding_model.onnx`
- `silero_vad.onnx` (used for wake gating inside openWakeWord and for **TTS barge-in** in this demo)
