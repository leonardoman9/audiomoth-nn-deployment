"""Compares the REPLAY line the board wrote to LOG.TXT with the one expected from the simulator (make_replay.py).
Usage: python compare_replay.py LOG.TXT expected.txt [TOLERANCE]      (TOLERANCE in logits, default 0.05)
The board and the simulator run the same C code; the int8 backbone is bit-exact, the float parts can differ in the
last bits between the Cortex-M4F and the host. In QEMU the difference was 0 at the logged precision (0.01)."""
import sys

log, expected = sys.argv[1], sys.argv[2]
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
want = open(expected).read().strip().split(",")
got = [l.strip().split(",") for l in open(log, errors="replace") if l.startswith("REPLAY,")]
if not got: sys.exit("FAIL: no REPLAY line in LOG.TXT (did the board find REPLAY.PCM? 6 red blinks = malformed file)")
got = got[-1]
dw = [int(x) / 100 for x in want[5:]]   # REPLAY,<time>,<class>,<pct>,<Y|N>,<logits...>
dg = [int(x) / 100 for x in got[5:]]
if len(dw) != len(dg): sys.exit(f"FAIL: {len(dg)} logits on the board, {len(dw)} expected (different model?)")
diff = max(abs(a - b) for a, b in zip(dw, dg))
same_class = want[2] == got[2]
print("expected:", ",".join(want[2:]))
print("board:   ", ",".join(got[2:]))
print(f"max |logit difference| = {diff:.2f} (tolerance {tol}) | same class: {same_class}")
ok = diff <= tol and same_class
print("PASS: the board computes what the simulator computes" if ok else "FAIL: the board's result differs from the simulator")
sys.exit(0 if ok else 1)
