"""Real timing from SESSIONS.TXT written by the board (one line per session).
Usage: python log_timing.py SESSIONS.TXT
Each line is "<start>,<recording>,<seconds recorded>,<samples lost>,<windows>,<classification seconds>": the whole window
is recorded first, then classified, so classification seconds / windows is the time the chip needs per 3 s window.
The estimate from QEMU instruction counts was 6-8 s per window, i.e. ~10-13 minutes for the 100 windows of a 5-minute
window. Samples lost > 0 means the SD card stalled for more than the 1.5 s recording buffer (try another card)."""
import sys

for l in open(sys.argv[1], errors="replace"):
    f = l.strip().split(",")
    if len(f) != 6: continue
    start, path, rec, lost, windows, secs = f[0], f[1], int(f[2]), int(f[3]), int(f[4]), int(f[5])
    per = secs / windows if windows else float("nan")
    awake = rec + secs
    print(f"{start} {path}: {rec} s recorded, {lost} samples lost ({lost / 32000:.2f} s), {windows} windows classified in"
          f" {secs} s -> {per:.1f} s per window; awake ~{awake / 60:.1f} min for this session")
