"""Runs the bird logger simulator over several simulated days.
Usage: python run_scenario.py SIM_BINARY AUDIO.raw WORKDIR DAYS [CLOCK_TEXT]
Each boot is one process (a power cycle). Prints a summary and leaves LOG.TXT and dump/ in WORKDIR."""
import os, shutil, subprocess, sys
from pathlib import Path

binary, audio, work, days = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4])
clock = sys.argv[5] if len(sys.argv) > 5 else "2026-10-21 06:58:30"
shutil.rmtree(work, ignore_errors=True)
(work / "sd").mkdir(parents=True); (work / "dump").mkdir()
(work / "sd" / "CONFIG.TXT").write_text(clock + "\n")
env = dict(os.environ, SIM_DIR=str(work), SIM_AUDIO=audio)
boots, leds, powerdowns, log = 0, {"green": 0, "red": 0}, [], []
limit_boots = days * 40 + 50
while boots < limit_boots:
    r = subprocess.run([binary], env=env, capture_output=True, text=True)
    boots += 1
    for line in r.stdout.splitlines():
        if "green LED" in line: leds["green"] += 1
        elif "red LED" in line: leds["red"] += 1
        elif "POWER_DOWN" in line: powerdowns.append(int(line.split("wake_after=")[1]))
    if r.returncode != 0:
        print("boot", boots, "exit code", r.returncode, r.stderr[-300:]); break
    # stop once `days` calendar days are logged and the last session is complete (the device went to sleep with
    # nothing pending: STATE.TXT says DONE and the last power-down is a long one, i.e. until the next window)
    logf, statef = work / "sd" / "LOG.TXT", work / "sd" / "STATE.TXT"
    if logf.exists() and statef.exists() and statef.read_text().startswith("DONE") and powerdowns and powerdowns[-1] >= 3600:
        if len({l[:10] for l in logf.read_text().splitlines()}) >= days: break
lines = (work / "sd" / "LOG.TXT").read_text().splitlines() if (work / "sd" / "LOG.TXT").exists() else []
print(f"boots={boots} log lines={len(lines)} LED={leds} sleeps={len(powerdowns)} (first {powerdowns[:3]}, last {powerdowns[-2:]})")
print("sd files:", sorted(str(p.relative_to(work / "sd")) for p in (work / "sd").rglob("*") if p.is_file()))
print("first lines:"); print("\n".join(lines[:4]))
print("last lines:"); print("\n".join(lines[-2:]))
