import fs from 'node:fs';

/**
 * Minimal PCM s16le extractor for standard 44-byte WAV headers (mono/stereo 16-bit).
 * @param {string} filePath
 * @returns {{ sampleRate: number, channels: number, pcm: Buffer }}
 */
export function readWavPcmS16le(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.length < 44 || buf.toString('ascii', 0, 4) !== 'RIFF' || buf.toString('ascii', 8, 12) !== 'WAVE') {
    throw new Error('Not a RIFF/WAVE file');
  }
  let offset = 12;
  let audioFormat = 0;
  let numChannels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  while (offset + 8 <= buf.length) {
    const id = buf.toString('ascii', offset, offset + 4);
    const size = buf.readUInt32LE(offset + 4);
    const dataStart = offset + 8;
    if (id === 'fmt ') {
      audioFormat = buf.readUInt16LE(dataStart);
      numChannels = buf.readUInt16LE(dataStart + 2);
      sampleRate = buf.readUInt32LE(dataStart + 4);
      bitsPerSample = buf.readUInt16LE(dataStart + 14);
    } else if (id === 'data') {
      const pcm = buf.subarray(dataStart, dataStart + size);
      return { sampleRate, channels: numChannels, bitsPerSample, audioFormat, pcm };
    }
    offset = dataStart + size + (size % 2);
  }
  throw new Error('WAV has no data chunk');
}

/**
 * Downmix interleaved stereo s16le to mono by averaging channels.
 * @param {Buffer} interleavedStereo length multiple of 4
 * @returns {Buffer}
 */
export function stereoToMonoS16le(interleavedStereo) {
  const n = interleavedStereo.length / 4;
  const out = Buffer.alloc(n * 2);
  for (let i = 0; i < n; i++) {
    const l = interleavedStereo.readInt16LE(i * 4);
    const r = interleavedStereo.readInt16LE(i * 4 + 2);
    const m = Math.round((l + r) / 2);
    out.writeInt16LE(m, i * 2);
  }
  return out;
}
