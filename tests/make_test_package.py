"""Makes a model package with a different number of classes, to test that the firmware and the tests do not depend
on the default model's 9 classes. The classifier rows are taken from the default package (cycled, with a different
bias per row); everything else is copied. The weights mean nothing: use it for plumbing tests only.
Usage: python make_test_package.py OUT_DIR N_CLASSES [--window-start HH:MM] [--window-minutes M] [--confidence PCT]"""
import argparse, re, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "firmware/bird/model"
p = argparse.ArgumentParser()
p.add_argument("out"); p.add_argument("classes", type=int)
p.add_argument("--window-start", default="07:00"); p.add_argument("--window-minutes", default="5"); p.add_argument("--confidence", default="50")
a = p.parse_args()
out, n = Path(a.out), a.classes
shutil.rmtree(out, ignore_errors=True)
shutil.copytree(SRC, out, ignore=shutil.ignore_patterns("test", "model_config.json", "model_config.h", "README.md"))

hdr = out / "generated" / "bird_tail_weights_i8.h"
text = hdr.read_text()
def get(name):
    m = re.search(r"static const (\w+) %s\[(\d+)\] = \{(.*?)\};" % name, text, re.S)
    return m.group(1), m.group(3).split(",")
def put(name, ctype, values):
    global text
    text = re.sub(r"static const \w+ %s\[\d+\] = \{.*?\};" % name,
                  lambda _: "static const %s %s[%d] = {%s};" % (ctype, name, len(values), ",".join(values)), text, flags=re.S)
tw, w = get("k_w_f"); ts, sc = get("k_w_f_scale"); tb, b = get("k_b_f")
hidden = len(w) // len(sc)
rows = [w[(i % len(sc)) * hidden:(i % len(sc) + 1) * hidden] for i in range(n)]
put("k_w_f", tw, [v for r in rows for v in r])
put("k_w_f_scale", ts, [sc[i % len(sc)] for i in range(n)])
put("k_b_f", tb, ["%.6ff" % (float(b[i % len(b)].rstrip("f")) + 0.25 * (i // len(b))) for i in range(n)])
hdr.write_text(text)

names = ",".join([f"Species {i:02d}" for i in range(n - 1)] + ["background"])
subprocess.run([sys.executable, str(ROOT / "firmware/bird/gen_model_config.py"), str(out), "--names", names,
                "--window-start", a.window_start, "--window-minutes", a.window_minutes, "--confidence", a.confidence], check=True)
