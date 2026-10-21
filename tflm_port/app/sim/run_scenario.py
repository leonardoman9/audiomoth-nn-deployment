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
    state = (work / "rtc.state").read_text().split()
    simulated_days = (int(state[0]) / 1000 - 1792540800) / 86400 if False else None
    logf = work / "sd" / "LOG.TXT"
    if logf.exists():
        lines = logf.read_text().splitlines()
        first_day = lines[0][:10]
        # stop once we have logged `days` different calendar days
        if len({l[:10] for l in lines}) >= days and (int(state[0]) // 1000) % 86400 > 7 * 3600 + 400: break
lines = (work / "sd" / "LOG.TXT").read_text().splitlines() if (work / "sd" / "LOG.TXT").exists() else []
print(f"boots={boots} log lines={len(lines)} LED={leds} sleeps={len(powerdowns)} (first {powerdowns[:3]}, last {powerdowns[-2:]})")
print("sd files:", sorted(p.name for p in (work / 'sd').iterdir()))
print("first lines:"); print("\n".join(lines[:4]))
print("last lines:"); print("\n".join(lines[-2:]))
