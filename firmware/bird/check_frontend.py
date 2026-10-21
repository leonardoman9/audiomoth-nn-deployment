"""Compare bird_frontend.c with the PyTorch front-end on real audio (int16 PCM, as the AudioMoth delivers).
Usage: python check_frontend.py WRENNET_ROOT CHECKPOINT.pt LIBFE FILE_OR_DIR [N] [CHUNK]
"""
import ctypes, random, sys
from pathlib import Path

import numpy as np
import soundfile
import torch
import torchaudio

root, ckpt, lib, src = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
n_max = int(sys.argv[5]) if len(sys.argv) > 5 else 1
chunk = int(sys.argv[6]) if len(sys.argv) > 6 else 512
sys.path.insert(0, str(root / "tools"))
from bird_model_export_check import SR, frontend, load_model  # noqa: E402

m = load_model(ckpt)
spec_mod = m.combined_log_linear_spec
c = ctypes.CDLL(lib)
c.BirdFE_SizeOf.restype = ctypes.c_uint32
c.BirdFE_Init.argtypes = [ctypes.c_void_p]
c.BirdFE_Push.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
c.BirdFE_FinishFloat.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
c.BirdFE_FinishFloat.restype = ctypes.c_bool
c.BirdFE_Finish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
c.BirdFE_Finish.restype = ctypes.c_bool
c.BirdFE_Reset.argtypes = [ctypes.c_void_p]
fe = ctypes.create_string_buffer(c.BirdFE_SizeOf())
c.BirdFE_Init(fe)
print("struct bytes:", c.BirdFE_SizeOf())
files = sorted(src.rglob("*.wav")) + sorted(src.rglob("*.mp3")) if src.is_dir() else [src]
random.Random(1).shuffle(files)
scale, zp = 0.028251195326447487, -25   # backbone input quantisation
worst_f, worst_q, same_frac = 0.0, 0, []
done = 0
for path in files:
    if done >= n_max: break
    try:
        data, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        continue
    wav = torchaudio.functional.resample(torch.from_numpy(data.T.copy()).mean(0, keepdim=True), sr, SR)
    wav = torch.nn.functional.pad(wav, (0, max(0, 96000 - wav.shape[1])))[:, :96000]
    pcm = np.clip(np.rint(wav[0].numpy() * 32768.0), -32768, 32767).astype(np.int16)
    with torch.no_grad():
        ref, _ = frontend(m, torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None])
    ref = ref[0].T.numpy()                                  # [301, 64]
    c.BirdFE_Reset(fe)
    for off in range(0, len(pcm), chunk):
        part = np.ascontiguousarray(pcm[off:off + chunk])
        c.BirdFE_Push(fe, part.ctypes.data, len(part))
    out_f = np.zeros((301, 64), np.float32)
    assert c.BirdFE_FinishFloat(fe, out_f.ctypes.data)
    # second pass for the int8 output (state was consumed by Finish: recompute)
    c.BirdFE_Reset(fe)
    for off in range(0, len(pcm), chunk):
        part = np.ascontiguousarray(pcm[off:off + chunk])
        c.BirdFE_Push(fe, part.ctypes.data, len(part))
    out_q = np.zeros((301, 64), np.int8)
    assert c.BirdFE_Finish(fe, out_q.ctypes.data, scale, zp)
    q_ref = np.clip(np.rint(ref / scale) + zp, -128, 127).astype(np.int8)
    df = np.abs(out_f - ref); dq = np.abs(out_q.astype(int) - q_ref.astype(int))
    worst_f = max(worst_f, float(df.max())); worst_q = max(worst_q, int(dq.max())); same_frac.append(float((dq == 0).mean()))
    done += 1
print(f"clips={done} chunk={chunk}: float spec max|diff|={worst_f:.4f} (values ~N(0,1)) | int8 max|diff|={worst_q} LSB, identical bytes mean {100*np.mean(same_frac):.2f}%")
