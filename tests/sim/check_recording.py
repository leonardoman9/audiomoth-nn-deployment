"""Checks the session recordings the simulator wrote (YYYYMMDD/HHMMSS.WAV) against the microphone source.
Usage: python check_recording.py SRC.raw SD_DIR [MAX_REL_ERROR]
For each WAV: the header matches the data size and says 32 kHz mono 16 bit; the samples are the looped source (the
simulator feeds SRC.raw in a loop), quantised like the 12-bit ADC, compared above 300 Hz (the 48 Hz DC-blocking filter
only shapes the low end) after locating the start in the source; runs of exact zeros (silence written for SD stalls)
are reported and left out of the comparison. Prints one line per file and PASS/FAIL."""
import sys, wave
from pathlib import Path
import numpy as np

src = np.fromfile(sys.argv[1], dtype=np.int16).astype(np.float64)
adc = 16.0 * np.trunc(src / 16.0)
limit = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10


def hp(x, fc=300, sr=32000):
    s = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / sr); s[f < fc] = 0
    return np.fft.irfft(s, n=len(x))


def zero_runs(x, min_len=64):
    z = np.concatenate(([0], (x == 0).astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(z))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


failed = 0
files = sorted(Path(sys.argv[2]).rglob("*.WAV"))
for f in files:
    raw = f.read_bytes()
    # chunks: RIFF/WAVE, fmt, JUNK (pads the header to 512 bytes), data
    pos, chunks = 12, {}
    while pos + 8 <= len(raw):
        cid, size = raw[pos:pos + 4], int.from_bytes(raw[pos + 4:pos + 8], "little")
        chunks[cid] = (pos + 8, size); pos += 8 + size + (size & 1)
        if cid == b"data": break
    data_at, data_size = chunks.get(b"data", (0, -1))
    n_hdr = data_size // 2
    with wave.open(str(f)) as w:
        ok_fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (32000, 1, 2)
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
    ok_size = (data_at == 512 and n_hdr == len(x) == (len(raw) - 512) // 2 and int.from_bytes(raw[4:8], "little") == 504 + data_size)
    gaps = zero_runs(x)
    # locate the start: correlate a high-passed probe from the first second without gaps against the source
    good = np.ones(len(x), bool)
    for a, b in gaps: good[a:b] = False
    start = next((i for i in range(0, len(x) - 32000, 3200) if good[i:i + 32000].all()), None)
    rel = float("nan")
    if start is not None:
        probe = hp(x[start:start + 32000])[4000:28000]
        tiled = np.concatenate([adc, adc[:64000]])
        best, off = -1.0, 0
        for o in range(0, len(adc), 16):
            seg = tiled[o + 4000:o + 28000]
            if len(seg) < len(probe): break
            c = (seg - seg.mean()) @ (probe - probe.mean()) / (np.linalg.norm(seg - seg.mean()) * np.linalg.norm(probe) + 1e-9)
            if c > best: best, off = c, o
        for o in range(max(0, off - 16), off + 17):
            seg = tiled[o + 4000:o + 28000]
            c = (seg - seg.mean()) @ (probe - probe.mean()) / (np.linalg.norm(seg - seg.mean()) * np.linalg.norm(probe) + 1e-9)
            if c > best: best, off = c, o
        # expected signal for the whole file, aligned at `start`
        idx = (off - start + np.arange(len(x))) % len(adc)
        a, b = hp(x)[good], hp(adc[idx])[good]
        a, b = a[5000:-5000], b[5000:-5000]
        rel = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9))
    ok = ok_fmt and ok_size and rel <= limit
    failed += not ok
    print(f"{f.relative_to(sys.argv[2])}: {len(x)} samples ({len(x) / 32000:.2f} s) | header {'ok' if ok_size and ok_fmt else 'BAD'}"
          f" | zero runs (SD-stall silence or digital silence): {sum(b - a for a, b in gaps)} samples in {len(gaps)}"
          f" | relative error above 300 Hz = {rel:.4f} | {'ok' if ok else 'FAIL'}")
print("PASS" if files and not failed else f"FAIL ({failed} of {len(files)} files)" if files else "FAIL (no WAV files)")
sys.exit(0 if files and not failed else 1)
