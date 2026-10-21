"""Checks schedule.c (parsing, formatting, window logic, log line) against Python's datetime/numpy.
Usage: [MODEL_PKG=dir] python test_schedule.py PATH_TO_test_schedule_BINARY
  binary: cc -Ifirmware/app -I<package>/generated firmware/app/schedule.c tests/schedule/test_schedule.c -lm"""
import datetime, json, math, os, random, subprocess, sys
from pathlib import Path
import numpy as np
S = sys.argv[1]
# classes and window of the model package: MODEL_PKG, default firmware/bird/model
CFG = json.load(open(Path(os.environ.get("MODEL_PKG") or Path(__file__).resolve().parents[2] / "firmware/bird/model") / "model_config.json"))
NAMES, N, W0, WL, CONF = CFG["names"], CFG["classes"], CFG["window_start_s"], CFG["window_length_s"], CFG["confidence_pct"]
rnd = random.Random(4); cases = []; expect = []
epoch = datetime.datetime(1970, 1, 1)
def unix(dt): return int((dt - epoch).total_seconds())
for _ in range(400):
    dt = datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=rnd.randrange(0, 80 * 365 * 86400)); s = unix(dt)
    cases.append(f"parse {dt:%Y-%m-%d %H:%M:%S}\n"); expect.append(f"{s}\n")
    cases.append(f"fmt {s}\n"); expect.append(f"{dt:%Y-%m-%d %H:%M:%S}\n")
    t = s % 86400; inw = int(W0 <= t < W0 + WL); to_next = 0 if inw else (W0 - t if t < W0 else 86400 - t + W0); left = W0 + WL - t if inw else 0
    cases.append(f"win {s}\n"); expect.append(f"{inw} {to_next} {left}\n")
def r100(x):
    v = np.float32(x) * np.float32(100.0); return int(v + (np.float32(0.5) if v >= 0 else np.float32(-0.5)))
for _ in range(600):
    lg = [rnd.uniform(-9, 9) for _ in range(N)]; s = rnd.randrange(1700000000, 1900000000)
    cases.append(f"log {s} " + " ".join(f"{x:.6f}" for x in lg) + "\n")
    l32 = [np.float32(float(f"{x:.6f}")) for x in lg]; b = max(range(N), key=lambda i: l32[i])
    z = sum(math.exp(float(x - l32[b])) for x in l32); pct = int(100 / z + 0.5)
    dt = epoch + datetime.timedelta(seconds=s)
    expect.append(f"{dt:%Y-%m-%d %H:%M:%S},{NAMES[b]},{pct},{'Y' if pct >= CONF else 'N'}," + ",".join(str(r100(x)) for x in l32) + "\n")
# broken logits (NaN, inf, huge) must give a bounded, well-formed line: values clamped to +-999999, NaN -> 0
yn = lambda pct: "Y" if pct >= CONF else "N"
for lg, want in [(["nan"] * N, f"{NAMES[0]},0,N," + ",".join(["0"] * N)),
                 (["inf"] + ["0"] * (N - 1), f"{NAMES[0]},0,N,999999," + ",".join(["0"] * (N - 1))),
                 (["-1e30"] * N, f"{NAMES[0]},{int(100 / N + 0.5)},{yn(int(100 / N + 0.5))}," + ",".join(["-999999"] * N)),
                 (["1e30"] * 2 + ["-1e30"] * (N - 2), f"{NAMES[0]},50,{yn(50)}," + ",".join(["999999"] * 2 + ["-999999"] * (N - 2)))]:
    cases.append("log 1792566000 " + " ".join(lg) + "\n"); expect.append("2026-10-21 07:00:00," + want + "\n")
for bad in ["2026-02-30 07:00:00", "2026-04-31 07:00:00", "2027-02-29 07:00:00", "2026-10-21 06:58:309", "2026-10-21 06:58:30x", "2026-13-01 00:00:00", "2026-10-21T06:00:00", "abc", "2026-10-21 25:00:00", "APPLIED 2026-10-21 06:58:30"]:
    cases.append(f"parse {bad}\n"); expect.append("ERR\n")
for good, dt in [("2028-02-29 12:00:00", datetime.datetime(2028, 2, 29, 12)), ("  2026-10-21 06:58:30  ", datetime.datetime(2026, 10, 21, 6, 58, 30))]:
    cases.append(f"parse {good}\n"); expect.append(f"{unix(dt)}\n")
# recording files, WAV header, window count, STATE.TXT and SESSIONS.TXT lines
HOP = CFG.get("window_hop_samples", 96000)
for _ in range(200):
    dt = datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=rnd.randrange(0, 80 * 365 * 86400)); s = unix(dt)
    cases.append(f"path {s}\n"); expect.append(f"{dt:%Y%m%d/%H%M%S}.WAV\n")
    n = rnd.choice([0, 1, 95999, 96000, 96001, rnd.randrange(0, 40_000_000)])
    cases.append(f"count {n}\n"); expect.append(f"{0 if n < 96000 else (n - 96000) // HOP + 1}\n")
    # 512-byte header: RIFF, fmt, JUNK padding, data (the samples start on a sector)
    hdr = (b"RIFF" + (504 + 2 * n).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
           + (1).to_bytes(2, "little") + (32000).to_bytes(4, "little") + (64000).to_bytes(4, "little") + (2).to_bytes(2, "little")
           + (16).to_bytes(2, "little") + b"JUNK" + (460).to_bytes(4, "little") + bytes(460) + b"data" + (2 * n).to_bytes(4, "little"))
    assert len(hdr) == 512
    cases.append(f"wav {n}\n"); expect.append(hdr.hex() + "\n")
    p = f"{dt:%Y%m%d/%H%M%S}.WAV"; samples = rnd.randrange(96000, 40_000_000); nxt = rnd.randrange(0, (samples - 96000) // HOP + 2)
    gaps = rnd.randrange(0, samples + 1)
    tries = rnd.randrange(0, 5)
    for text in [f"REC {p} {s}", f"CLS {p} {s} {samples} {nxt} {gaps} {tries}", "DONE"]:
        cases.append(f"state {text}\n"); expect.append(text + "\n")
    w, secs = rnd.randrange(0, 10000), rnd.randrange(0, 100000)
    cases.append(f"session {s} {samples} {gaps} {w} {secs}\n"); expect.append(f"{dt:%Y-%m-%d %H:%M:%S},{p},{samples // 32000},{gaps},{w},{secs}\n")
too_far = (40_000_000 - 96000) // HOP + 2
# the longest possible record still fits
cases.append("state CLS 20261021/070000.WAV 4294967295 4294967295 4294967295 4294967295 4294967295\n"); expect.append("ERR\n")  # next > windows
big = (4294967295 - 96000) // HOP + 1
cases.append(f"state CLS 20261021/070000.WAV 4294967295 4294967295 {big} 4294967295 4294967295\n")
expect.append(f"CLS 20261021/070000.WAV 4294967295 4294967295 {big} 4294967295 4294967295\n")
for bad in ["REC 20261021/070000.WAV", "REC 20261021/07000.WAV 5", "CLS 20261021/070000.WAV 1 96000 2 0 0", f"CLS 20261021/070000.WAV 1 40000000 {too_far} 0 0",
            "CLS 20261021/070000.WAV 1 96000 0 96001 0", "CLS 20261021/070000.WAV 1 96000 0 0", "CLS 2026102x/070000.WAV 1 96000 0 0 0", "XYZ 20261021/070000.WAV 1", "REC 20261021/070000.WAV 99999999999",
            "REC 20261021/070000.TXT 1", "REC 20261021/070000.WAV 1 junk", "", "\xff\xfe"]:
    cases.append(f"state {bad}\n"); expect.append("ERR\n")
out = subprocess.run([S], input="".join(cases), capture_output=True, text=True).stdout
bad = [(c, e, o) for c, e, o in zip(cases, expect, out.splitlines(keepends=True)) if e != o]
print("cases", len(cases), "mismatches", len(bad)); print(bad[:2])
