"""Synthetic test audio for the simulator when no recordings are at hand (CI): bird-like chirps in pink-ish noise.
Usage: python make_test_audio.py OUT.raw [SECONDS]   -> raw int16 mono @32 kHz. Only numpy is needed.
It exercises the whole pipeline; it is not meant to be classified correctly."""
import sys
import numpy as np
out, seconds, fs = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60, 32000
rng = np.random.default_rng(1)
n = seconds * fs
noise = np.cumsum(rng.normal(0, 1, n)); noise -= np.convolve(noise, np.ones(320) / 320, "same")  # crude 1/f, no DC
x = 0.02 * noise / np.std(noise)
t = np.arange(int(0.25 * fs)) / fs
for start in rng.uniform(0, seconds - 0.3, seconds * 2):
    f0, f1 = rng.uniform(2000, 6000), rng.uniform(2000, 8000)
    chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t * t / 0.5)) * np.hanning(len(t)) * rng.uniform(0.05, 0.3)
    i = int(start * fs); x[i:i + len(t)] += chirp
np.clip(x * 32767, -32768, 32767).astype("<i2").tofile(out)
print(f"wrote {out}: {seconds} s")
