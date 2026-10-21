"""Export the tail of the bird model (GRU, projection, attention, classifier) as C
weights plus reference vectors for host/QEMU tests.

Usage (env with torch, torchaudio, torchsummary, soundfile):
  python gen_bird_tail.py WRENNET_ROOT CHECKPOINT.pt REFERENCE.npz OUT_DIR

REFERENCE.npz is docs/deployment/bird/bird_reference.npz (written by
tools/bird_model_export_check.py). Weights are written twice: float32 (exact
reference) and int8 with one symmetric scale per output row (flash-friendly).
"""
import sys
from pathlib import Path

import numpy as np
import torch

root, ckpt, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root / "tools"))
from bird_model_export_check import load_model  # noqa: E402

m = load_model(ckpt)
r = np.load(ref)
spec = torch.from_numpy(r["spec"])
with torch.no_grad():
    feats = m.phi(spec).permute(0, 2, 1).contiguous()[0].numpy().astype(np.float32)  # [T, 32]
    logits = m(torch.from_numpy(r["wav"])).numpy().astype(np.float32)[0]
T = feats.shape[0]
print("backbone features", feats.shape, "logits", np.round(logits, 4))
feats.tofile(out / "tail_feats.bin")
logits.tofile(out / "tail_logits.bin")
(out / "tail_dims.h").write_text(f"#define BIRD_TAIL_FRAMES {T}\n")

g = m.gru
tensors = {  # PyTorch gate order along dim 0: reset, update, new
    "w_ih": g.weight_ih_l0, "w_hh": g.weight_hh_l0, "b_ih": g.bias_ih_l0, "b_hh": g.bias_hh_l0,
    "w_p": m.projection.weight, "b_p": m.projection.bias,
    "w_a": m.keyword_attention.attention.weight, "b_a": m.keyword_attention.attention.bias,
    "w_f": m.fc.weight, "b_f": m.fc.bias,
}
tensors = {k: v.detach().numpy().astype(np.float32) for k, v in tensors.items()}


def arr(name, a, ctype, fmt):
    body = ",".join(fmt(x) for x in a.reshape(-1))
    return f"static const {ctype} {name}[{a.size}] = {{{body}}};\n"


f32 = "".join(arr("k_" + k, v, "float", lambda x: repr(float(x)) + "f") for k, v in tensors.items())
(out / "bird_tail_weights_f32.h").write_text(f32)

i8 = ""
for k, v in tensors.items():
    if k.startswith("w_"):
        rows = v.reshape(v.shape[0], -1)
        scale = np.maximum(np.abs(rows).max(axis=1) / 127.0, 1e-12).astype(np.float32)
        q = np.clip(np.rint(rows / scale[:, None]), -127, 127).astype(np.int8)
        i8 += arr("k_" + k, q, "int8_t", lambda x: str(int(x)))
        i8 += arr("k_" + k + "_scale", scale, "float", lambda x: repr(float(x)) + "f")
    else:
        i8 += arr("k_" + k, v, "float", lambda x: repr(float(x)) + "f")
(out / "bird_tail_weights_i8.h").write_text(i8)
print("params in weights:", sum(v.size for k, v in tensors.items() if k.startswith("w_")))
