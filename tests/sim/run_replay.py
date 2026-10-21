"""Replay test: builds REPLAY.PCM from a window of the reference set, boots the simulator once and compares the
logged logits with the PyTorch logits of the same window.
Usage: python run_replay.py SIM_BINARY REF.npz INDEX WORKDIR     (REF.npz: pcm, logits, names)"""
import os, subprocess, sys
from pathlib import Path
import numpy as np
binary, ref, idx, work = sys.argv[1], np.load(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
import shutil; shutil.rmtree(work, ignore_errors=True); (work / "sd").mkdir(parents=True); (work / "dump").mkdir()
(work / "sd" / "CONFIG.TXT").write_text("2026-10-21 12:00:00\n")            # time set; replay ignores the schedule
(work / "sd" / "REPLAY.PCM").write_bytes(b"RPLY" + ref["pcm"][idx].astype("<i2").tobytes() + b"END!")
env = dict(os.environ, SIM_DIR=str(work), SIM_AUDIO="/dev/null")
r = subprocess.run([binary], env=env, capture_output=True, text=True)
log = (work / "sd" / "LOG.TXT").read_text().splitlines()
print("exit", r.returncode, "| LED:", sorted(set(l for l in r.stdout.splitlines() if "LED" in l)), "| REPLAY.PCM now", (work / "sd" / "REPLAY.PCM").stat().st_size, "bytes")
print("log:", log[0][:110], "...")
f = log[0].split(",")
dev = np.array([int(x) / 100 for x in f[-9:]]); pt = ref["logits"][idx]
print("device logits(x100 rounding) vs PyTorch: max|d| = %.3f | argmax device=%d torch=%d | window %s" % (np.abs(dev - pt).max(), dev.argmax(), pt.argmax(), ref["names"][idx]))
# second boot must NOT replay again
r2 = subprocess.run([binary], env=env, capture_output=True, text=True)
print("second boot log lines:", len((work / "sd" / "LOG.TXT").read_text().splitlines()), "(expected 1)")
