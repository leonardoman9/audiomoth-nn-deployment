"""Compare the C tail (int8 weights) with the PyTorch model on many clips.
Usage: python sweep_tail.py WRENNET_ROOT CHECKPOINT.pt LIB.dylib WAV_DIR N
"""
import ctypes, random, sys
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio

root, ckpt, lib, wav_dir, n = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, str(root / "tools"))
from bird_model_export_check import SR, frontend, load_model  # noqa: E402

m = load_model(ckpt)
c = ctypes.CDLL(lib)
class Tail(ctypes.Structure):
    _fields_ = [("h", ctypes.c_float * 64), ("ctx", ctypes.c_float * 64), ("mx", ctypes.c_float), ("sm", ctypes.c_float)]
c.BirdTail_Step.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
c.BirdTail_Finish.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
c.BirdTail_Reset.argtypes = [ctypes.POINTER(Tail)]

files = sorted(wav_dir.rglob("*.wav"))
random.Random(0).shuffle(files)
worst, agree, done = 0.0, 0, 0
diffs = []
for path in files:
    if done >= n: break
    try:
        data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        continue
    wav = torchaudio.functional.resample(torch.from_numpy(data.T.copy()).mean(0, keepdim=True), sr, SR)
    wav = torch.nn.functional.pad(wav, (0, max(0, 96000 - wav.shape[1])))[:, :96000]
    with torch.no_grad():
        ref = m(wav)[0].numpy()
        spec, _ = frontend(m, wav)
        feats = m.phi(spec).permute(0, 2, 1).contiguous()[0].numpy().astype(np.float32)
    t = Tail(); c.BirdTail_Reset(ctypes.byref(t))
    for row in feats:
        c.BirdTail_Step(ctypes.byref(t), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    out = (ctypes.c_float * c.BirdTail_Classes())(); c.BirdTail_Finish(ctypes.byref(t), out)
    got = np.array(out)
    d = float(np.abs(got - ref).max()); diffs.append(d); worst = max(worst, d)
    agree += int(got.argmax() == ref.argmax()); done += 1
print(f"clips={done} max|dlogit|={worst:.4f} mean={np.mean(diffs):.4f} p99={np.percentile(diffs,99):.4f} argmax agree={agree}/{done}")
