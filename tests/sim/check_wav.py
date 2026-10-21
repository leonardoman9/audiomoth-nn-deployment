"""Checks the diagnostic WAVs written by the simulator against the microphone source: the source is quantised like the 12-bit ADC
(x/16, truncated, x16) and both signals are compared above 300 Hz (the 48 Hz DC-blocking filter only shapes the low end).
Usage: python check_wav.py SRC.raw SD_DIR"""
import glob, sys, wave
import numpy as np
src = np.fromfile(sys.argv[1], dtype=np.int16).astype(np.float64)
adc = 16.0 * np.trunc(src / 16.0)
def hp(x, sr=32000, fc=300):
    s = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / sr); s[f < fc] = 0; return np.fft.irfft(s, n=len(x))
for f in sorted(glob.glob(sys.argv[2] + "/*.WAV")):
    w = wave.open(f); n = w.getnframes(); x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float64)
    probe = hp(x)[3000:9000]; best = (-1, 0)
    for off in range(0, len(adc) - n, 32):                     # all 32-sample steps, then refine
        seg = adc[off + 3000:off + 9000]; seg = seg - seg.mean(); p = probe - probe.mean()
        c = seg @ p / (np.linalg.norm(seg) * np.linalg.norm(p) + 1e-9)
        if c > best[0]: best = (c, off)
    for o in range(max(0, best[1] - 32), best[1] + 33):
        seg = adc[o + 3000:o + 9000]; seg = seg - seg.mean(); p = probe - probe.mean()
        c = seg @ p / (np.linalg.norm(seg) * np.linalg.norm(p) + 1e-9)
        if c > best[0]: best = (c, o)
    o = best[1]
    a, b = hp(x)[5000:-5000], hp(adc[o:o + n])[5000:-5000]
    rel = np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(b ** 2)) + 1e-9)
    print(f"{f.split('/')[-1]}: header ok ({w.getframerate()} Hz, {w.getnchannels()} ch, {8*w.getsampwidth()} bit, {n} samples) | source offset {o} | relative error above 300 Hz = {rel:.4f} | rms {np.sqrt(np.mean(x**2)):.1f}")
