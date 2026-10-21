"""Prepares a replay test for the real board: REPLAY.PCM from 3 s of audio, plus the log line the board should write.
Usage: python make_replay.py SIM_BINARY AUDIO OUT_DIR [START_SECONDS]
  SIM_BINARY  tests/sim/out/bird_logger_sim (tests/sim/build_sim.sh)
  AUDIO       a 32 kHz mono 16-bit WAV (e.g. one recorded by the diagnostic firmware) or a raw int16 32 kHz file
Writes OUT_DIR/REPLAY.PCM (copy it to the SD card) and OUT_DIR/expected.txt (the REPLAY line the simulator logged for
the same samples, i.e. the same C code on the host). After the board has run, compare with compare_replay.py."""
import os, shutil, subprocess, sys, tempfile, wave
from pathlib import Path
import numpy as np

binary, audio, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
start = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
if audio.lower().endswith(".wav"):
    w = wave.open(audio)
    if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (32000, 1, 2): sys.exit("need a 32 kHz mono 16-bit WAV")
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
else:
    pcm = np.fromfile(audio, dtype="<i2")
i = int(start * 32000)
window = pcm[i:i + 96000]
if len(window) < 96000: sys.exit(f"need 3 s of audio from {start} s, have {len(window) / 32000:.2f} s")
out.mkdir(parents=True, exist_ok=True)
data = b"RPLY" + window.astype("<i2").tobytes() + b"END!"
(out / "REPLAY.PCM").write_bytes(data)

with tempfile.TemporaryDirectory() as tmp:
    work = Path(tmp)
    (work / "sd").mkdir(); (work / "dump").mkdir()
    (work / "sd" / "CONFIG.TXT").write_text("2026-10-21 12:00:00\n")  # clock set, outside the daily window
    (work / "sd" / "REPLAY.PCM").write_bytes(data)
    r = subprocess.run([binary], env=dict(os.environ, SIM_DIR=str(work), SIM_AUDIO="/dev/null"), capture_output=True, text=True)
    log = (work / "sd" / "LOG.TXT")
    lines = [l for l in log.read_text().splitlines() if l.startswith("REPLAY,")] if log.exists() else []
    if r.returncode != 0 or not lines: sys.exit(f"simulator did not log a REPLAY line (exit {r.returncode}): {r.stdout[-300:]}")
(out / "expected.txt").write_text(lines[0] + "\n")
print(f"wrote {out / 'REPLAY.PCM'} ({len(data)} bytes) and {out / 'expected.txt'}:\n{lines[0]}")
