"""Checks the simulator's dumped windows against the PyTorch model on the same PCM.
Usage: [MODEL_PKG=dir] python verify_log.py WRENNET_ROOT CKPT WORKDIR"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
root, ckpt, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(root / "tools"))
from bird_model_export_check import load_model
m = load_model(ckpt)
names = json.load(open(Path(os.environ.get("MODEL_PKG") or Path(__file__).resolve().parents[2] / "firmware/bird/model") / "model_config.json"))["names"]
log = (work / "sd" / "LOG.TXT").read_text().splitlines()
wins = sorted((work / "dump").glob("win_*.pcm"))
print("log lines", len(log), "dumped windows", len(wins))
agree, dmax, pct_bad, label_bad, logit_bad = 0, 0.0, 0, 0, 0
for i, w in enumerate(wins):
    pcm = np.fromfile(w, np.int16)
    assert len(pcm) == 96000, (w.name, len(pcm))
    dev = np.fromfile(w.with_suffix(".logits"), np.float32)
    with torch.no_grad(): ref = m(torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None])[0].numpy()
    if dev.argmax() != ref.argmax(): print('  flip at', w.name, 'torch confidence %.2f' % (np.exp(ref - ref.max()) / np.exp(ref - ref.max()).sum()).max())
    agree += int(dev.argmax() == ref.argmax()); dmax = max(dmax, float(np.abs(dev - ref).max()))
    e = np.exp(dev - dev.max()); p = e / e.sum()
    parts = log[len(log) - len(wins) + i].split(",")  # dumps are from the last session (one process = one boot)
    t, name, pct, yn = parts[:4]; logged = np.array([int(x) for x in parts[4:13]]) / 100.0
    logit_bad += float(np.abs(logged - dev).max()) > 0.0051
    label_bad += name != names[int(dev.argmax())]; pct_bad += abs(int(pct) - int(100 * p.max() + 0.5)) > 0
print(f"device-vs-PyTorch on the exact windows: class agree {agree}/{len(wins)}, max|dlogit| {dmax:.3f}")
print(f"log line vs device logits: wrong class names {label_bad}, wrong percentages {pct_bad}, logits not matching the device output {logit_bad}")
