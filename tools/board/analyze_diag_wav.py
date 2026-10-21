"""Checks the WAV files written by the diagnostic firmware (BIRD_DIAG_RECORD) on a real AudioMoth.
Usage: python analyze_diag_wav.py FILE.WAV [FILE.WAV ...]        (only numpy is needed)

For each file it reports format, length, DC offset, RMS and peak level, clipping, dropouts (runs of identical
samples), noise floor and level per frequency band, and an estimated sound level. It ends with OK/WARN lines.

Level calibration (nominal, from datasheets, not measured): microphone SPU0410LR5H-QB -38 dBV/Pa, medium gain
x15 (op-amp 1: R2 = 15 R1, op-amp 2: R2 = R1), ADC full scale +-1.25 V; the decimation filter maps the 12-bit ADC
full scale to int16 full scale. So int16 full scale ~ 1.25 / (0.0126 * 15) = 6.6 Pa peak ~ 110 dB SPL peak.
Expect a few dB of error; use a calibrator or a reference meter for real numbers."""
import sys, wave
import numpy as np

FS_PA = 1.25 / (10 ** (-38 / 20) * 15)  # int16 full scale in pascal (peak)
P_REF = 20e-6


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def analyse(path):
    w = wave.open(path)
    rate, ch, bits, n = w.getframerate(), w.getnchannels(), 8 * w.getsampwidth(), w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64)
    warn = []
    print(f"== {path}: {rate} Hz, {ch} ch, {bits} bit, {n} samples ({n / rate:.2f} s)")
    if (rate, ch, bits) != (32000, 1, 16): warn.append("format is not 32 kHz mono 16 bit")
    if n != 96000: warn.append(f"expected 96000 samples (3 s), got {n}")
    if n == 0:
        return warn + ["empty file"]
    dc = x.mean()
    ac = x - dc
    rms, peak = np.sqrt(np.mean(ac ** 2)), np.abs(x).max()
    clip = int(np.sum(np.abs(x) >= 32760))
    # dropouts: runs of >= 32 identical samples (a stalled DMA or a dead microphone gives long flat runs)
    change = np.flatnonzero(np.diff(x) != 0)
    runs = np.diff(np.concatenate(([-1], change, [len(x) - 1])))
    longest = int(runs.max())
    frames = ac[: len(ac) // 1024 * 1024].reshape(-1, 1024)
    frame_db = db(np.sqrt(np.mean(frames ** 2, axis=1)) / 32768)
    spl = lambda r: 20 * np.log10(max(r, 1e-12) / 32768 * FS_PA / P_REF)
    print(f"   DC offset {dc:+.1f} LSB | RMS {db(rms / 32768):.1f} dBFS (~{spl(rms):.0f} dB SPL) | peak {db(peak / 32768):.1f} dBFS"
          f" | clipped samples {clip} | longest flat run {longest} samples")
    print(f"   noise floor (10th percentile of 32 ms frames) {np.percentile(frame_db, 10):.1f} dBFS"
          f" (~{spl(32768 * 10 ** (np.percentile(frame_db, 10) / 20)):.0f} dB SPL), loudest frame {frame_db.max():.1f} dBFS")
    spec = np.abs(np.fft.rfft(ac * np.hanning(len(ac)))) ** 2
    f = np.fft.rfftfreq(len(ac), 1 / rate)
    total = spec.sum() + 1e-12
    bands = [(0, 300), (300, 2000), (2000, 8000), (8000, 16000)]
    print("   energy by band: " + ", ".join(f"{a}-{b} Hz {100 * spec[(f >= a) & (f < b)].sum() / total:.1f}%" for a, b in bands))
    if abs(dc) > 200: warn.append(f"large DC offset ({dc:.0f} LSB): the 48 Hz high-pass should remove it")
    if clip > 0: warn.append(f"{clip} clipped samples: lower the gain or the source level")
    if longest >= 32: warn.append(f"flat run of {longest} samples: possible dropout or dead microphone")
    if db(rms / 32768) < -80: warn.append("signal almost zero: microphone or gain not working?")
    if spec[(f >= 0) & (f < 300)].sum() / total > 0.8: warn.append("most energy below 300 Hz: hum, wind or handling noise")
    return warn


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    problems = 0
    for p in sys.argv[1:]:
        for wmsg in analyse(p):
            print("   WARN " + wmsg); problems += 1
    print("OK: no problems found" if problems == 0 else f"{problems} warning(s)")
