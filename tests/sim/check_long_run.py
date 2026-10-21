"""Checks the SD card of a simulator run over one or more days (run_scenario.py) or of a real board.
Usage: [MODEL_PKG=dir] python check_long_run.py WORKDIR FIRST_DATE DAYS [MIN MAX]      (e.g. 2027-12-30 400)
WORKDIR is the folder that contains sd/. The whole daily window is recorded and then classified, so each day must have
exactly the number of 3 s windows that fit in it (MIN-MAX overrides this, e.g. for a board started mid-window), all
inside the package's daily window, in time order, without repeats, well formed and without broken (clamped) logits.
SESSIONS.TXT must have one session per day with the whole window recorded and no samples lost, and each day folder one
WAV of the right size. Prints the per-day counts and the sizes on the card."""
import collections, datetime, json, os, sys
from pathlib import Path
# classes and window of the model package: MODEL_PKG, default firmware/bird/model
CFG = json.load(open(Path(os.environ.get("MODEL_PKG") or Path(__file__).resolve().parents[2] / "firmware/bird/model") / "model_config.json"))
W0, WL = CFG["window_start_s"], CFG["window_length_s"]
HOP = CFG.get("window_hop_samples", 96000)

PER_DAY = (WL * 32000 - 96000) // HOP + 1

work, first, days = Path(sys.argv[1]), datetime.date.fromisoformat(sys.argv[2]), int(sys.argv[3])
lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (PER_DAY, PER_DAY)
sd = work / "sd"
log = sd / "LOG.TXT"
lines = [l for l in log.read_text().splitlines() if not l.startswith("REPLAY,")]
errors, per_day, prev = [], collections.OrderedDict(), None
for i, l in enumerate(lines):
    f = l.split(",")
    try:
        t = datetime.datetime.strptime(f[0], "%Y-%m-%d %H:%M:%S")
        ok = len(f) == 4 + CFG["classes"] and f[3] in "YN" and 0 <= int(f[2]) <= 100 and all(abs(int(x)) < 999999 for x in f[4:])
    except ValueError:
        ok = False
    if not ok: errors.append(f"line {i + 1} malformed: {l!r}"); continue
    sod = t.hour * 3600 + t.minute * 60 + t.second  # seconds of the day (a window may end at midnight)
    if not (W0 <= sod < W0 + WL): errors.append(f"line {i + 1} outside the window: {l[:19]}")
    # with a hop of a second or more every window has its own second: an equal time is a repeated window
    if prev and (t < prev or (t == prev and HOP >= 32000)): errors.append(f"line {i + 1} not after the previous one: {l[:19]}")
    prev = t
    per_day[t.date()] = per_day.get(t.date(), 0) + 1
expected = [first + datetime.timedelta(days=d) for d in range(days)]
missing = [d for d in expected if d not in per_day]
extra = [d for d in per_day if d not in expected]
counts = collections.Counter(per_day.values())
print(f"{len(lines)} lines over {len(per_day)} days, lines/day {dict(counts)} (expected {lo}-{hi}), LOG.TXT {log.stat().st_size} bytes")
if lines: print(f"first {lines[0][:19]}, last {lines[-1][:19]}; leap day logged: {datetime.date(2028, 2, 29) in per_day}")
if missing: errors.append(f"missing days: {missing[:5]}{'...' if len(missing) > 5 else ''} ({len(missing)})")
if extra: errors.append(f"unexpected days: {extra[:5]}")
wrong = [d for d, c in per_day.items() if not lo <= c <= hi]
if wrong: errors.append(f"days with fewer than {lo} or more than {hi} lines: {wrong[:5]} ({len(wrong)})")

# sessions and recordings
sessions = (sd / "SESSIONS.TXT").read_text().splitlines() if (sd / "SESSIONS.TXT").exists() else []
by_day = collections.Counter(s[:10] for s in sessions)
wavs = sorted(sd.glob("*/*.WAV"))
size = sum(w.stat().st_size for w in wavs)
print(f"{len(sessions)} sessions, {len(wavs)} recordings ({size / 1e6:.1f} MB); last session: {sessions[-1] if sessions else None}")
if lo == hi == PER_DAY:  # full windows: one complete session per day
    for s in sessions:
        f = s.split(",")
        if len(f) != 6 or int(f[2]) != WL or int(f[3]) != 0 or int(f[4]) != PER_DAY:
            errors.append(f"session not complete or with lost samples: {s}")
    if any(c != 1 for c in by_day.values()) or len(by_day) != days: errors.append(f"sessions per day: {dict(by_day)}")
    bad = [w.name for w in wavs if w.stat().st_size != 512 + 2 * 32000 * WL]
    if bad or len(wavs) != days: errors.append(f"{len(wavs)} recordings for {days} days, wrong size: {bad[:3]}")
state = (sd / "STATE.TXT").read_text().strip() if (sd / "STATE.TXT").exists() else None
if state != "DONE": errors.append(f"STATE.TXT is {state!r}, expected DONE")
print("\n".join(errors[:10]) if errors else "PASS")
sys.exit(1 if errors else 0)
