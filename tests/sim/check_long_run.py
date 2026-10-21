"""Checks LOG.TXT of a long simulator run (run_scenario.py over many days).
Usage: python check_long_run.py WORKDIR FIRST_DATE DAYS [MIN MAX]      (e.g. 2027-12-30 400)
Every calendar day must appear, in order, with MIN-MAX lines (default 40-50 for the simulator, one per ~6 s; LED
blinks after confident windows take some of the time; a real board gives fewer), all inside 07:00-07:05, times strictly
increasing, well-formed lines and no broken (clamped) logits. Prints the per-day line count and the LOG.TXT size."""
import collections, datetime, sys
from pathlib import Path

work, first, days = Path(sys.argv[1]), datetime.date.fromisoformat(sys.argv[2]), int(sys.argv[3])
lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (40, 50)
log = work / "sd" / "LOG.TXT"
lines = log.read_text().splitlines()
errors, per_day, prev = [], collections.OrderedDict(), None
for i, l in enumerate(lines):
    f = l.split(",")
    try:
        t = datetime.datetime.strptime(f[0], "%Y-%m-%d %H:%M:%S")
        ok = len(f) == 13 and f[3] in "YN" and 0 <= int(f[2]) <= 100 and all(abs(int(x)) < 999999 for x in f[4:])
    except ValueError:
        ok = False
    if not ok: errors.append(f"line {i + 1} malformed: {l!r}"); continue
    if not (datetime.time(7, 0) <= t.time() < datetime.time(7, 5)): errors.append(f"line {i + 1} outside the window: {l[:19]}")
    if prev and t <= prev: errors.append(f"line {i + 1} not after the previous one: {l[:19]}")
    prev = t
    per_day[t.date()] = per_day.get(t.date(), 0) + 1
expected = [first + datetime.timedelta(days=d) for d in range(days)]
missing = [d for d in expected if d not in per_day]
extra = [d for d in per_day if d not in expected]
counts = collections.Counter(per_day.values())
print(f"{len(lines)} lines over {len(per_day)} days, lines/day {dict(counts)}, LOG.TXT {log.stat().st_size} bytes")
print(f"first {lines[0][:19]}, last {lines[-1][:19]}; leap day logged: {datetime.date(2028, 2, 29) in per_day}")
if missing: errors.append(f"missing days: {missing[:5]}{'...' if len(missing) > 5 else ''} ({len(missing)})")
if extra: errors.append(f"unexpected days: {extra[:5]}")
short = [d for d, c in per_day.items() if not lo <= c <= hi]
if short: errors.append(f"days with fewer than {lo} or more than {hi} lines: {short[:5]} ({len(short)})")
print("\n".join(errors[:10]) if errors else "PASS")
sys.exit(1 if errors else 0)
