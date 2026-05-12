#!/usr/bin/env python3
"""
stdin: raw PCM s16le mono 16 kHz, any length (buffered to 1280-sample frames).
stdout: one JSON object per line: {"scores": {...}}

argv: comma-separated wake model names or paths (openWakeWord Model API), optional VAD threshold as argv2.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from openwakeword.model import Model

FRAME_SAMPLES = 1280
FRAME_BYTES = FRAME_SAMPLES * 2


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: stream_stdin.py <model[,model2,...]> [vad_threshold]",
            file=sys.stderr,
        )
        sys.exit(2)

    models = [m.strip() for m in sys.argv[1].split(",") if m.strip()]
    vad = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    oww = Model(
        wakeword_models=models,
        inference_framework="onnx",
        vad_threshold=vad,
    )

    buf = bytearray()
    stdin = sys.stdin.buffer

    while True:
        chunk = stdin.read(8192)
        if not chunk:
            break
        buf.extend(chunk)
        while len(buf) >= FRAME_BYTES:
            frame = bytes(buf[:FRAME_BYTES])
            del buf[:FRAME_BYTES]
            audio = np.frombuffer(frame, dtype=np.int16)
            scores = oww.predict(audio)
            serializable = {k: float(v) for k, v in scores.items()}
            sys.stdout.write(json.dumps({"scores": serializable}, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    if len(buf) > 0:
        pad = FRAME_BYTES - len(buf)
        frame = bytes(buf) + b"\x00" * pad
        audio = np.frombuffer(frame, dtype=np.int16)
        scores = oww.predict(audio)
        serializable = {k: float(v) for k, v in scores.items()}
        sys.stdout.write(json.dumps({"scores": serializable}, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
