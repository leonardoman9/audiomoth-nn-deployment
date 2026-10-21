"""Real timing from a LOG.TXT written by the board during the 07:00 session.
Usage: python log_timing.py LOG.TXT
Each line is stamped when a 3 s window has been recorded, just before it is classified, and the next window starts
recording right after the classification. So the spacing between consecutive lines of one session is
3 s of recording + the classification time (+ ~0.3 s for each LED blink after a confident window).
The estimate from QEMU instruction counts was 6-8 s of classification, i.e. a spacing of ~9-11 s."""
import collections, datetime, statistics, sys

sessions = collections.defaultdict(list)
for l in open(sys.argv[1], errors="replace"):
    if l.startswith("REPLAY,") or len(l) < 19: continue
    try: t = datetime.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError: continue
    sessions[t.date()].append(t)
for day, ts in sessions.items():
    gaps = [(b - a).total_seconds() for a, b in zip(ts, ts[1:])]
    if not gaps: print(f"{day}: {len(ts)} line(s), not enough to measure"); continue
    med = statistics.median(gaps)
    print(f"{day}: {len(ts)} windows, spacing median {med:.0f} s (min {min(gaps):.0f}, max {max(gaps):.0f})"
          f" -> classification ~{med - 3:.0f} s per window, {len(ts)} windows in the 5-minute session")
