"""Fault-injection scenarios for the logger simulator (see SIM_FAULT in hal_sim.c).
Usage: [MODEL_PKG=dir] python run_faults.py SIM_BINARY AUDIO.raw WORKDIR
Each scenario starts from a fresh SD card with CONFIG.TXT = 90 s before the package's daily window and runs boots (one
process = one power cycle) until the session is over (clock past the window, nothing pending in STATE.TXT, long
sleep) or a boot limit is hit, then checks what the firmware did: log lines, recordings, STATE.TXT, SESSIONS.TXT."""
import datetime, json, os, shutil, subprocess, sys
from pathlib import Path
import numpy as np

binary, audio, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
HERE = Path(__file__).resolve().parent
# classes and window of the model package: MODEL_PKG, default firmware/bird/model
CFG = json.load(open(Path(os.environ.get("MODEL_PKG") or HERE.parents[1] / "firmware/bird/model") / "model_config.json"))
W0, WL = CFG["window_start_s"], CFG["window_length_s"]
HOP = CFG.get("window_hop_samples", 96000)
DAY = datetime.datetime(2026, 10, 21)
EPOCH_DAY = int((DAY - datetime.datetime(1970, 1, 1)).total_seconds())
START = DAY + datetime.timedelta(seconds=W0 - 90)    # clock set 90 s before the window
WINDOW_END = EPOCH_DAY + W0 + WL
FIELDS = 4 + CFG["classes"]
PER_DAY = (WL * 32000 - 96000) // HOP + 1             # windows in a whole recorded window
EXPECTED_TIMES = [DAY + datetime.timedelta(seconds=W0 + i * HOP // 32000) for i in range(PER_DAY)]
def windows_in(seconds):
    """3 s windows in a recording of `seconds` seconds."""
    return 0 if seconds * 32000 < 96000 else (seconds * 32000 - 96000) // HOP + 1
def in_window(t):
    return all(DAY + datetime.timedelta(seconds=W0) <= x < DAY + datetime.timedelta(seconds=W0 + WL) for x in t)


class Run:
    def __init__(self, name, config=START.strftime("%Y-%m-%d %H:%M:%S") + "\n", audio_file=audio, files=None, env=None):
        self.work = root / name
        shutil.rmtree(self.work, ignore_errors=True)
        (self.work / "sd").mkdir(parents=True); (self.work / "dump").mkdir()
        if config is not None: (self.work / "sd" / "CONFIG.TXT").write_bytes(config.encode() if isinstance(config, str) else config)
        for n, data in (files or {}).items():
            (self.work / "sd" / n).parent.mkdir(parents=True, exist_ok=True); (self.work / "sd" / n).write_bytes(data)
        self.audio, self.env = audio_file, env or {}
        self.boots, self.red, self.green, self.sleeps, self.events = 0, 0, 0, [], []

    def boot(self, fault=""):
        env = dict(os.environ, SIM_DIR=str(self.work), SIM_AUDIO=self.audio, SIM_FAULT=fault, **self.env)
        r = subprocess.run([binary], env=env, capture_output=True, text=True, timeout=900)
        self.boots += 1
        if r.returncode != 0: raise RuntimeError(f"{self.work.name}: boot {self.boots} crashed ({r.returncode}): {r.stderr[-600:]}")
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

    def state(self):
        s = self.file("STATE.TXT")
        return s.decode(errors="replace").strip() if s is not None else None

    def until_end(self, fault="", max_boots=40):
        while self.boots < max_boots:
            done = (self.clock() or 0) > WINDOW_END and self.state() in (None, "DONE") and self.sleeps and self.sleeps[-1] >= 3600
            if done: break
            self.boot(fault)

    def log(self):
        f = self.work / "sd" / "LOG.TXT"
        return [l for l in f.read_text().splitlines() if not l.startswith("REPLAY,")] if f.exists() else []

    def replay_lines(self):
        f = self.work / "sd" / "LOG.TXT"
        return [l for l in f.read_text().splitlines() if l.startswith("REPLAY,")] if f.exists() else []

    def file(self, name):
        f = self.work / "sd" / name
        return f.read_bytes() if f.exists() else None

    def sessions(self):
        s = self.file("SESSIONS.TXT")
        return [l.split(",") for l in s.decode().splitlines()] if s else []

    def wavs(self):
        return sorted((self.work / "sd").glob("*/*.WAV"))


results = []
def check(name, cond, detail):
    results.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + " | " + detail, flush=True)

def well_formed(lines):
    for l in lines:
        f = l.split(",")
        if l.startswith("REPLAY,"): f = f[1:]
        if len(f) != FIELDS or f[3] not in "YN" or not (0 <= int(f[2]) <= 100): return False
        if any(abs(int(x)) >= 999999 for x in f[4:]): return False  # clamped value = broken logit
    return True

def times(lines):
    return [datetime.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S") for l in lines]

def wav_ok(path):
    """512-byte header (RIFF, fmt, JUNK, data) consistent with the file size, 32 kHz mono 16 bit."""
    raw = path.read_bytes()
    return (raw[:4] == b"RIFF" and raw[8:16] == b"WAVEfmt " and int.from_bytes(raw[24:28], "little") == 32000
            and raw[504:508] == b"data" and int.from_bytes(raw[508:512], "little") == len(raw) - 512
            and int.from_bytes(raw[4:8], "little") == len(raw) - 8)

def samples_of(path):
    return (path.stat().st_size - 512) // 2

def full_session(r):
    """One complete recording of the whole window, classified window by window, nothing pending."""
    lines, s, w = r.log(), r.sessions(), r.wavs()
    return (len(lines) == PER_DAY and times(lines) == EXPECTED_TIMES and well_formed(lines) and r.state() == "DONE"
            and len(s) == 1 and int(s[0][2]) == WL and int(s[0][4]) == PER_DAY and len(w) == 1 and wav_ok(w[0])
            and samples_of(w[0]) == 32000 * WL)

def resumed(lines, expected=EXPECTED_TIMES):
    """After a cut during the classification: every window once, at most one repeated (a cut between its line and
    the saved progress)."""
    t = times(lines)
    return set(t) == set(expected) and len(t) - len(set(t)) <= 1 and well_formed(lines)

def windows_match_wav(r):
    """Every classified window (dumped by the simulator) is the matching slice of the recording: nothing stale."""
    w = r.wavs()
    if len(w) != 1: return False
    wav = np.frombuffer(w[0].read_bytes()[512:], np.int16)
    dumps = sorted((r.work / "dump").glob("win_*.pcm"))
    return len(dumps) == windows_in(len(wav) // 32000) and all(
        np.array_equal(np.fromfile(d, np.int16), wav[i * HOP:i * HOP + 96000]) for i, d in enumerate(dumps))


# 0. reference: no fault. Every window is the matching slice of the recording.
r = Run("ok"); r.until_end()
check("no fault", full_session(r) and r.red == 0, f"{len(r.log())} lines, sessions {r.sessions()}, boots {r.boots}, sleeps {r.sleeps[:3]}")
check("windows = slices of the recording", windows_match_wav(r), f"{len(list((r.work / 'dump').glob('win_*.pcm')))} windows compared")

# 1. no SD card from the start: no clock, retry every 60 s, never crash
r = Run("nosd_start"); [r.boot("nosd") for _ in range(5)]
check("no SD at first boot", r.clock() is None and r.sleeps == [60] * 5 and r.log() == [], f"sleeps {r.sleeps}, red {r.red}")

# 2. SD fails after the clock is set: no session, sleeps to the next window (not a tight loop)
r = Run("nosd_later"); r.boot(); r.until_end("nosd")
check("SD lost after clock set", r.log() == [] and r.boots <= 4 and r.sleeps[-1] > 3600, f"boots {r.boots}, sleeps {r.sleeps}")

# 3. external SRAM does not enable: no recording, no crash
r = Run("nosram"); r.until_end("nosram")
check("no external SRAM", r.log() == [] and r.wavs() == [] and r.boots <= 4 and r.sleeps[-1] > 3600, f"boots {r.boots}, sleeps {r.sleeps}, red {r.red}")

# 4. every SD write fails: no recording, red blinks, sleeps to the next window
r = Run("writefail"); r.boot(); r.until_end("writefail")
check("SD writes fail", r.log() == [] and r.red >= 2 and r.boots <= 4 and r.sleeps[-1] > 3600, f"boots {r.boots}, red {r.red}, sleeps {r.sleeps}")

# 5. LOG.TXT cannot be opened: the window is still recorded and classified (red blinks), the session completes
r = Run("appendfail"); r.boot(); r.until_end("appendfail")
s = r.sessions()
check("LOG.TXT cannot be written", r.log() == [] and r.red >= 2 * PER_DAY and len(s) == 1 and int(s[0][4]) == PER_DAY
      and r.state() == "DONE" and len(r.wavs()) == 1, f"red {r.red}, sessions {s}")

# 6. classification fails: no lines, every window tried, session completes
r = Run("pipefail"); r.until_end("pipefail")
check("classification fails", r.log() == [] and r.red >= 3 * PER_DAY and r.state() == "DONE", f"boots {r.boots}, red {r.red}")

# 7. SD stalls. One write busy for 400 ms succeeds and the ring absorbs it: nothing lost. One busy for 3 s fails after
#    the driver's 500 ms timeout: the file is reopened, the lost stretch written as silence and counted, and every
#    window is still classified at its real time.
for ms, lossy in [(400, False), (3000, True)]:
    r = Run(f"sdstall{ms}"); r.boot(); r.until_end(f"sdstall={ms}")
    s = r.sessions()
    gaps = int(s[0][3]) if s else -1
    check(f"SD busy for {ms} ms", full_session(r) and ((0 < gaps < 32000) if lossy else gaps == 0) and windows_match_wav(r),
          f"lost samples {gaps} ({gaps / 32000:.2f} s), red {r.red}")
chk = subprocess.run([sys.executable, str(HERE / "check_recording.py"), audio, str(r.work / "sd")], capture_output=True, text=True)
check("SD failure: the rest of the recording is intact", chk.returncode == 0, chk.stdout.strip().splitlines()[0] if chk.stdout else chk.stderr[-200:])

# 8. one write fails in the middle of the recording (half of it reaches the card): same recovery
r = Run("wfail"); r.boot(); r.until_end("wfail=2000")
s = r.sessions()
gaps = int(s[0][3]) if s else -1
check("one SD write fails mid-recording", full_session(r) and 0 < gaps < 32000 and windows_match_wav(r), f"lost samples {gaps}, red {r.red}")

# 9. the card fills up after ~100 s of recording: the recording stops there with a correct header, only the windows
#    that are really in the file are classified (none from stale data), and nothing is left pending
r = Run("diskfull"); r.boot(); r.until_end(f"diskfull={512 + 2 * 32000 * 100 + 1234}")
w = r.wavs()
# (the room also holds STATE.TXT and the folder, so the recording ends a little before 100 s + 1234 bytes)
check("card full", len(w) == 1 and wav_ok(w[0]) and 99.9 * 32000 < samples_of(w[0]) <= 100 * 32000 + 617 and windows_match_wav(r)
      and r.state() == "DONE", f"{samples_of(w[0]) / 32000 if w else 0:.2f} s recorded, {len(list((r.work / 'dump').glob('win_*.pcm')))} windows classified, red {r.red}")

# 10. slow cards: 50 ms per 4 KB write keeps up (64 ms of audio per write); 80 ms does not, every second loses audio
for ms, lossy in [(50, False), (80, True)]:
    r = Run(f"slow{ms}", env={"SIM_SD_WRITE_MS": str(ms)}); r.until_end()
    s = r.sessions()
    gaps = int(s[0][3]) if s else -1
    check(f"SD writes of {ms} ms", full_session(r) and ((gaps > 0) if lossy else (gaps == 0)), f"lost samples {gaps} ({gaps / 32000:.1f} s of {WL} s)")

# 11. reset (watchdog, brown-out) or power cut during the recording: the clock is lost with it, but the next boot,
#     without a clock, classifies the part of the recording that reached the card (up to the last sync) with its
#     original times; then it waits (60 s) for a new CONFIG.TXT
for fault in ["reset", "powercut"]:
    r = Run(f"{fault}_rec"); r.boot()  # sets the clock, sleeps until 2 s before the window
    r.boot(f"{fault}=62000"); r.boot()
    lines, t = r.log(), times(r.log())
    check(f"{fault} during the recording", r.events and r.clock() is None and r.sleeps[-1] == 60 and r.state() == "DONE"
          and windows_in(56) <= len(lines) <= windows_in(60) and t == EXPECTED_TIMES[:len(t)] and len(r.wavs()) == 1
          and wav_ok(r.wavs()[0]), f"{len(lines)} lines from ~60 s recorded, {r.sessions()}, sleeps {r.sleeps[-2:]}")

# 12. reset or power cut during the classification: resumed at the next boot, without a clock
for fault in ["reset", "powercut"]:
    r = Run(f"{fault}_cls"); r.boot()
    r.boot(f"{fault}={(WL + 62) * 1000}"); r.boot()
    check(f"{fault} during the classification", r.events and resumed(r.log()) and r.state() == "DONE" and r.sleeps[-1] == 60,
          f"{len(r.log())} lines for {PER_DAY} windows, sessions {r.sessions()}")

# 13. power cuts at 25 times spread over the recording and the classification, including inside the saving of
#     STATE.TXT: after the next boot every window is logged once (at most one repeated), nothing is pending, the
#     recording's header matches the file
rnd = __import__("random").Random(7)
bad = []
for k, cut in enumerate(sorted(rnd.uniform(3.0, WL + 2 + PER_DAY * 3.4) for _ in range(25))):
    r = Run("sweep"); r.boot()
    r.boot(f"powercut={int(cut * 1000)}"); r.boot()
    w = r.wavs()
    n = samples_of(w[0]) if w else 0  # windows from the exact samples (whole seconds are not enough with a 1.5 s hop)
    recorded = n // 32000
    exp = EXPECTED_TIMES[:0 if n < 96000 else (n - 96000) // HOP + 1]
    ok = r.state() == "DONE" and len(w) == 1 and wav_ok(w[0]) and resumed(r.log(), exp)
    if not ok: bad.append(f"cut at {cut:.3f} s: {len(r.log())} lines, {recorded} s recorded, state {r.state()!r}")
check("power cut sweep (25 times)", not bad, "; ".join(bad[:3]) if bad else "every cut recovered")

# 14. corrupt STATE.TXT, or one pointing to a recording that is not on the card: ignored, normal session
for name, text in [("garbage", b"\xff\x00CLS nonsense"), ("missing", b"CLS 20261020/070000.WAV 1792479600 9600000 5 0 0\n"),
                   ("too_far", b"CLS 20261020/070000.WAV 1792479600 9600000 101 0 0\n"), ("empty", b"")]:
    r = Run(f"state_{name}", files={"STATE.TXT": text}); r.until_end()
    check(f"STATE.TXT {name}", full_session(r), f"{len(r.log())} lines, state {r.state()!r}")

# 15. a window that crashes the board (here: a power cut 1.5 s into it, while it is being classified). MAX_TRIES = 3
#     boots may start it: crashing twice and then working logs every window; crashing three times skips that window
#     at the 4th boot, which goes on with the rest
crash_at = (WL + 2) + 10 * 3.3   # inside the 11th window or so
for crashes, want in [(2, PER_DAY), (3, PER_DAY - 1)]:
    r = Run(f"crash_window{crashes}"); r.boot()
    r.boot(f"powercut={int(crash_at * 1000)}")
    for _ in range(crashes - 1): r.boot("powercut=1500")  # every resume dies 1.5 s in, on the same window
    r.boot()
    t = times(r.log())
    check(f"window crashing {crashes} times", r.state() == "DONE" and len(set(t)) == want and len(t) - len(set(t)) <= 1 and r.sleeps[-1] == 60,
          f"{len(set(t))} of {PER_DAY} windows logged (expected {want}), red {r.red}")

# 15b. a pending session whose recording cannot be opened (here its path is a folder): abandoned after MAX_TRIES boots
#      with a summary line, so it cannot block the next sessions; the same for a pending REC
for kind, text in [("CLS", b"CLS 20261020/070000.WAV 1792479600 9600000 5 0 0\n"), ("REC", b"REC 20261020/070000.WAV 1792479600\n")]:
    r = Run(f"wedge_{kind}"); (r.work / "sd" / "20261020" / "070000.WAV").mkdir(parents=True)
    (r.work / "sd" / "STATE.TXT").write_bytes(text)
    r.until_end(max_boots=12)
    s_ = r.sessions()
    # boots at 06:58:30 and 06:59:58 fail (tries 1, 2); the third boot, in the window, abandons it and records the rest
    check(f"unreadable pending {kind} abandoned", len(s_) == 2 and s_[0][1] == "20261020/070000.WAV" and s_[0][4] == ("5" if kind == "CLS" else "0")  # windows done before
          and int(s_[1][4]) > 0 and r.red >= 7 and r.boots >= 3 and r.state() == "DONE" and len(r.log()) == int(s_[1][4]),
          f"sessions {[x[1:5] for x in s_]}, boots {r.boots}, red {r.red}")

# 15c. creating the file takes longer than the 2 s the board allows (very slow card): the missed seconds become
#      silence (counted), so the file name and every time in LOG.TXT stay the times of the samples
r = Run("late_start", env={"SIM_SD_OPEN_MS": "1500"}); r.until_end()
s_ = r.sessions()
gaps = int(s_[0][3]) if s_ else -1
check("late start", full_session(r) and 2 * 32000 <= gaps <= 4 * 32000, f"lost samples {gaps} ({gaps / 32000:.1f} s), sessions {s_}")

# 16. boots inside the window with little time left (clock just set): 5 s left -> one window; 4 or 3 s -> none
for left in [5, 4, 3]:
    cfg = (DAY + datetime.timedelta(seconds=W0 + WL - left)).strftime("%Y-%m-%d %H:%M:%S") + "\n"
    r = Run(f"left{left}", config=cfg); r.boot()
    want = windows_in(left - 2)
    # recording starts 2 s after boot and needs 3 s; otherwise the board sleeps just past the end of the window
    check(f"{left} s left in the window", len(r.log()) == want and r.sleeps[-1] in ((left + 1,) if want == 0 else (21600,))
          and r.state() in (None, "DONE"),
          f"{len(r.log())} lines (expected {want}), sleeps {r.sleeps}")

# 17. a valid REPLAY.PCM found at a boot inside the window: the replay and the session both run (the network is
#     initialised twice in one boot)
pcm = np.fromfile(audio, np.int16)[:96000].astype("<i2").tobytes()
inside = (DAY + datetime.timedelta(seconds=W0 + 5)).strftime("%Y-%m-%d %H:%M:%S") + "\n"
r = Run("replay_in_window", config=inside, files={"REPLAY.PCM": b"RPLY" + pcm + b"END!"}); r.until_end()
s = r.sessions()
check("replay then session in one boot", len(r.replay_lines()) == 1 and well_formed(r.replay_lines()) and len(s) == 1
      and int(s[0][2]) >= WL - 12 and int(s[0][4]) == windows_in(int(s[0][2])) and len(r.log()) == int(s[0][4]) and r.state() == "DONE",
      f"replay lines {len(r.replay_lines())}, session {s}")

# 18. malformed CONFIG.TXT variants: no clock, no crash, file left alone
for i, cfg in enumerate(["", "garbage\n", "2026-02-30 07:00:00\n", "2026-10-21 07:00\n", "\xff" * 39, "2026-10-21 06:58:30" + "9" * 30]):
    r = Run(f"badcfg{i}", config=cfg); [r.boot() for _ in range(2)]
    check(f"bad CONFIG.TXT {cfg[:20]!r}", r.clock() is None and r.sleeps == [60, 60] and r.file("CONFIG.TXT") == cfg.encode(),
          f"sleeps {r.sleeps}")
r = Run("nocfg", config=None); [r.boot() for _ in range(2)]
check("no CONFIG.TXT", r.clock() is None and r.sleeps == [60, 60], f"sleeps {r.sleeps}")

# 19. malformed REPLAY.PCM: rejected (6 red blinks), truncated so it is not retried, normal session still runs
for name, data in [("short", b"RPLY" + b"\0" * 1000), ("badhead", b"XXXX" + b"\0" * 192000 + b"END!"), ("notail", b"RPLY" + b"\0" * 192000 + b"ENDX")]:
    r = Run(f"replay_{name}", files={"REPLAY.PCM": data}); r.until_end()
    # 6 red blinks once, at the boot that finds it; later boots skip the truncated file silently
    check(f"bad REPLAY.PCM ({name})", r.file("REPLAY.PCM") == b"x" and r.replay_lines() == [] and full_session(r) and r.red == 6,
          f"{len(r.log())} lines, red {r.red}, REPLAY.PCM now {r.file('REPLAY.PCM')!r}")

# 20. silent microphone (all zeros): per-clip normalisation sees zero spread; lines must still be finite and well formed
silent = root / "silence.raw"; silent.write_bytes(b"\0" * 32000 * 2 * 10)
r = Run("silence", audio_file=str(silent)); r.until_end()
check("silent microphone", full_session(r), f"{len(r.log())} lines, first: {r.log()[0] if r.log() else None}")

failed = [n for n, ok, _ in results if not ok]
print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed" + (f"; FAILED: {failed}" if failed else ""))
sys.exit(1 if failed else 0)
