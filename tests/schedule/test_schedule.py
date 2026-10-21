"""Checks schedule.c (parsing, formatting, window logic, log line) against Python's datetime/numpy.
Usage: python test_schedule.py PATH_TO_test_schedule_BINARY   (binary: cc -I. schedule.c test_schedule.c -lm)"""
import datetime, math, random, subprocess, sys
import numpy as np
S = sys.argv[1]
NAMES = ["Bubo bubo","Certhia familiaris","Apus apus","Certhia brachydactyla","Emberiza cia","Lophophanes cristatus","Periparus ater","Poecile montanus","non-bird"]
rnd = random.Random(4); cases = []; expect = []
epoch = datetime.datetime(1970, 1, 1)
def unix(dt): return int((dt - epoch).total_seconds())
for _ in range(400):
    dt = datetime.datetime(2020, 1, 1) + datetime.timedelta(seconds=rnd.randrange(0, 80 * 365 * 86400)); s = unix(dt)
    cases.append(f"parse {dt:%Y-%m-%d %H:%M:%S}\n"); expect.append(f"{s}\n")
    cases.append(f"fmt {s}\n"); expect.append(f"{dt:%Y-%m-%d %H:%M:%S}\n")
    t = s % 86400; inw = int(25200 <= t < 25500); to_next = 0 if inw else (25200 - t if t < 25200 else 86400 - t + 25200); left = 25500 - t if inw else 0
    cases.append(f"win {s}\n"); expect.append(f"{inw} {to_next} {left}\n")
def r100(x):
    v = np.float32(x) * np.float32(100.0); return int(v + (np.float32(0.5) if v >= 0 else np.float32(-0.5)))
for _ in range(600):
    lg = [rnd.uniform(-9, 9) for _ in range(9)]; s = rnd.randrange(1700000000, 1900000000)
    cases.append(f"log {s} " + " ".join(f"{x:.6f}" for x in lg) + "\n")
    l32 = [np.float32(float(f"{x:.6f}")) for x in lg]; b = max(range(9), key=lambda i: l32[i])
    z = sum(math.exp(float(x - l32[b])) for x in l32); pct = int(100 / z + 0.5)
    dt = epoch + datetime.timedelta(seconds=s)
    expect.append(f"{dt:%Y-%m-%d %H:%M:%S},{NAMES[b]},{pct},{'Y' if pct >= 50 else 'N'}," + ",".join(str(r100(x)) for x in l32) + "\n")
# broken logits (NaN, inf, huge) must give a bounded, well-formed line: values clamped to +-999999, NaN -> 0
for lg, want in [(["nan"] * 9, "Bubo bubo,0,N," + ",".join(["0"] * 9)),
                 (["inf"] + ["0"] * 8, "Bubo bubo,0,N,999999," + ",".join(["0"] * 8)),
                 (["-1e30"] * 9, "Bubo bubo,11,N," + ",".join(["-999999"] * 9)),
                 (["1e30"] * 4 + ["-1e30"] * 5, "Bubo bubo,25,N," + ",".join(["999999"] * 4 + ["-999999"] * 5))]:
    cases.append("log 1792566000 " + " ".join(lg) + "\n"); expect.append("2026-10-21 07:00:00," + want + "\n")
for bad in ["2026-13-01 00:00:00", "2026-10-21T06:00:00", "abc", "2026-10-21 25:00:00", "APPLIED 2026-10-21 06:58:30"]:
    cases.append(f"parse {bad}\n"); expect.append("ERR\n")
out = subprocess.run([S], input="".join(cases), capture_output=True, text=True).stdout
bad = [(c, e, o) for c, e, o in zip(cases, expect, out.splitlines(keepends=True)) if e != o]
print("cases", len(cases), "mismatches", len(bad)); print(bad[:2])
