"""Concatenates audio files into the raw int16 mono @32 kHz stream the simulator uses as the microphone.
Usage: python make_audio.py OUT.raw FILE_OR_DIR [FILE_OR_DIR ...]   (takes up to 12 s of each, max 24 files)"""
import random, sys
from pathlib import Path
import numpy as np, soundfile, torch, torchaudio
out, srcs = Path(sys.argv[1]), sys.argv[2:]
files = []
for s in srcs:
    p = Path(s); files += (sorted(p.rglob("*.wav")) + sorted(p.rglob("*.mp3"))) if p.is_dir() else [p]
random.Random(11).shuffle(files)
chunks = []
for f in files[:24]:
    try: d, sr = soundfile.read(str(f), dtype="float32", always_2d=True)
    except Exception: continue
    w = torchaudio.functional.resample(torch.from_numpy(d.T.copy()).mean(0, keepdim=True), sr, 32000)[0, :12 * 32000]
    chunks.append(np.clip(np.rint(w.numpy() * 32768.0), -32768, 32767).astype(np.int16))
a = np.concatenate(chunks); a.tofile(out)
print("wrote", out, len(a) / 32000, "s from", len(chunks), "files")
