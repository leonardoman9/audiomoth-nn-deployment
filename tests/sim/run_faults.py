"""Fault-injection scenarios for the bird logger simulator (see SIM_FAULT in hal_sim.c).
Usage: python run_faults.py SIM_BINARY AUDIO.raw WORKDIR
Each scenario starts from a fresh SD card with CONFIG.TXT = 06:58:30, runs boots (one process = one power cycle)
until the simulated clock passes 07:06 or a boot limit is hit, and checks what the firmware did."""
import datetime, math, os, shutil, subprocess, sys
from pathlib import Path

binary, audio, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
START = datetime.datetime(2026, 10, 21, 6, 58, 30)
END_S = int((datetime.datetime(2026, 10, 21, 7, 6) - datetime.datetime(1970, 1, 1)).total_seconds())


class Run:
    def __init__(self, name, config=START.strftime("%Y-%m-%d %H:%M:%S") + "\n", audio_file=audio, files=None):
        self.work = root / name
        shutil.rmtree(self.work, ignore_errors=True)
        (self.work / "sd").mkdir(parents=True); (self.work / "dump").mkdir()
        if config is not None: (self.work / "sd" / "CONFIG.TXT").write_bytes(config.encode() if isinstance(config, str) else config)
        for n, data in (files or {}).items(): (self.work / "sd" / n).write_bytes(data)
        self.audio, self.boots, self.red, self.green, self.sleeps, self.events = audio_file, 0, 0, 0, [], []

    def boot(self, fault=""):
        env = dict(os.environ, SIM_DIR=str(self.work), SIM_AUDIO=self.audio, SIM_FAULT=fault)
        r = subprocess.run([binary], env=env, capture_output=True, text=True, timeout=600)
        self.boots += 1
        if r.returncode != 0: raise RuntimeError(f"boot {self.boots} crashed ({r.returncode}): {r.stderr[-400:]}")
        for l in r.stdout.splitlines():
            if "red LED" in l: self.red += 1
            elif "green LED" in l: self.green += 1
            elif "POWER_DOWN" in l: self.sleeps.append(int(l.split("wake_after=")[1]))
            elif "POWER_CUT" in l or "RESET" in l: self.events.append(l.split("SIM: ")[1])

    def clock(self):
        st = self.work / "rtc.state"  # "<ms> <time set>"; the simulator also saves it when the clock was never set
        if not st.exists(): return None
        ms, is_set = st.read_text().split()
        return int(ms) // 1000 if is_set == "1" else None

    def until_end(self, fault="", max_boots=40):
        while self.boots < max_boots and (self.clock() or 0) < END_S: self.boot(fault)

    def log(self):
        f = self.work / "sd" / "LOG.TXT"
        return f.read_text().splitlines() if f.exists() else []

    def file(self, name):
        f = self.work / "sd" / name
        return f.read_bytes() if f.exists() else None


results = []
def check(name, cond, detail):
    results.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + " | " + detail)

def well_formed(lines):
    for l in lines:
        f = l.split(",")
        if l.startswith("REPLAY,"): f = f[1:]
        if len(f) != 13 or f[3] not in "YN" or not (0 <= int(f[2]) <= 100): return False
        if any(abs(int(x)) >= 999999 for x in f[4:]): return False  # clamped value = broken logit
    return True

# 0. reference: no fault
r = Run("ok"); r.until_end()
base = r.log()
check("no fault", len(base) >= 40 and well_formed(base), f"{len(base)} lines, boots {r.boots}")

# 1. no SD card from the start: no clock, retry every 60 s, never crash
r = Run("nosd_start"); [r.boot("nosd") for _ in range(5)]
check("no SD at first boot", r.clock() is None and r.sleeps == [60] * 5 and r.log() == [], f"sleeps {r.sleeps}, red {r.red}")

# 2. SD fails after the clock is set: no session, sleeps to the next window (not a tight loop)
r = Run("nosd_later"); r.boot(); r.until_end("nosd")
check("SD lost after clock set", r.log() == [] and r.boots <= 4 and r.sleeps[-1] > 3600, f"boots {r.boots}, sleeps {r.sleeps}")

# 3. external SRAM does not enable: no session, no crash
r = Run("nosram"); r.until_end("nosram")
check("no external SRAM", r.log() == [] and r.boots <= 4 and r.sleeps[-1] > 3600, f"boots {r.boots}, sleeps {r.sleeps}, red {r.red}")

# 4. every SD write fails / LOG.TXT cannot be opened: windows still classified, red blinks, session ends normally
for fault in ["writefail", "appendfail"]:
    r = Run(fault); r.boot(); r.until_end(fault)
    check(f"SD {fault}", r.log() == [] and r.red >= 2 * 40 and r.sleeps[-1] > 3600, f"boots {r.boots}, red blinks {r.red}")

# 5. classification fails: no lines, session continues to the end of the window
r = Run("pipefail"); r.until_end("pipefail")
check("classification fails", r.log() == [] and r.red >= 3 * 40 and r.sleeps[-1] > 3600, f"boots {r.boots}, red {r.red}")

# 6. watchdog-style reset in the middle of the session: clock survives, logging resumes in the same window
r = Run("reset"); r.boot()  # first boot sets the clock and sleeps until 07:00; the next boot runs the whole session
r.boot("reset=60000"); r.until_end()
lines = r.log()
check("reset mid-session", "RESET" in r.events and len(lines) >= 35 and well_formed(lines),
      f"{len(lines)} lines (no fault: {len(base)}), events {r.events}")

# 7. power cut in the middle of the session: lines written before the cut survive intact; the clock is lost, and
#    CONFIG.TXT is already marked APPLIED, so the device waits (60 s retries) for a new CONFIG.TXT
r = Run("powercut"); r.boot()  # first boot sets the clock and sleeps until 07:00; the next boot runs the whole session
r.boot("powercut=60000"); before = r.log()
for _ in range(3): r.boot()
check("power cut mid-session", "POWER_CUT" in r.events and len(before) >= 5 and well_formed(before) and r.log() == before
      and r.clock() is None and r.sleeps[-3:] == [60] * 3, f"{len(before)} lines kept, then sleeps {r.sleeps[-3:]}")
(r.work / "sd" / "CONFIG.TXT").write_text("2026-10-22 06:59:00\n"); r.boot()
check("new CONFIG.TXT after power cut", r.clock() is not None and (r.file("CONFIG.TXT") or b"").startswith(b"APPLIED"), f"clock {r.clock()}")

# 8. malformed CONFIG.TXT variants: no clock, no crash, file left alone
for i, cfg in enumerate(["", "garbage\n", "2026-02-30 07:00:00\n", "2026-10-21 07:00\n", "\xff" * 39, "2026-10-21 06:58:30" + "9" * 30]):
    r = Run(f"badcfg{i}", config=cfg); [r.boot() for _ in range(2)]
    check(f"bad CONFIG.TXT {cfg[:20]!r}", r.clock() is None and r.sleeps == [60, 60] and r.file("CONFIG.TXT") == cfg.encode(),
          f"sleeps {r.sleeps}")
r = Run("nocfg", config=None); [r.boot() for _ in range(2)]
check("no CONFIG.TXT", r.clock() is None and r.sleeps == [60, 60], f"sleeps {r.sleeps}")

# 9. malformed REPLAY.PCM: rejected (6 red blinks), truncated so it is not retried, normal session still runs
for name, data in [("short", b"RPLY" + b"\0" * 1000), ("badhead", b"XXXX" + b"\0" * 192000 + b"END!"), ("notail", b"RPLY" + b"\0" * 192000 + b"ENDX")]:
    r = Run(f"replay_{name}", files={"REPLAY.PCM": data}); r.until_end()
    lines = r.log()
    check(f"bad REPLAY.PCM ({name})", r.file("REPLAY.PCM") == b"x" and not any(l.startswith("REPLAY") for l in lines) and len(lines) >= 40,
          f"{len(lines)} lines, REPLAY.PCM now {r.file('REPLAY.PCM')!r}")

# 10. silent microphone (all zeros): per-clip normalisation sees zero spread; lines must still be finite and well formed
silent = root / "silence.raw"; silent.write_bytes(b"\0" * 32000 * 2 * 10)
r = Run("silence", audio_file=str(silent)); r.until_end()
lines = r.log()
check("silent microphone", len(lines) >= 40 and well_formed(lines), f"{len(lines)} lines, first: {lines[0] if lines else None}")

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed" + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
