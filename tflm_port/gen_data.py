"""Generate C headers (models + parity vectors) for the TFLM bare-metal test.

Usage: [BACKBONE_TFLITE=.. STREAMING_TFLITE=..] python gen_data.py WRENNET_ROOT OUT_DIR
Models land in .rodata (counted as firmware flash); vectors land in .testdata
(a separate linker region, not part of the firmware size).
"""
import os
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
kws = root / "birds_distillation_edge/convert-and-quantize/streamable-kws"


def emit(name, raw, section, path):
    data = np.frombuffer(raw, dtype=np.uint8)
    body = ",".join(str(b) for b in data)
    path.write_text(
        f'extern "C" alignas(16) __attribute__((section("{section}"))) const unsigned char {name}[] = {{{body}}};\n')


emit("g_backbone_model", Path(os.environ.get("BACKBONE_TFLITE", kws / "backbone_int8_FINAL.tflite")).read_bytes(), ".rodata", out / "backbone_model.inc")
emit("g_streaming_model", Path(os.environ.get("STREAMING_TFLITE", kws / "streaming_int8_FINAL.tflite")).read_bytes(), ".rodata", out / "streaming_model.inc")
v = np.load(root / "docs/deployment/reference/vectors.npz")
for key, name in [("backbone_inputs", "backbone_in"), ("backbone_outputs", "backbone_out"),
                  ("sequence_features", "seq_feat"), ("sequence_hidden_outputs", "seq_hidden"),
                  ("sequence_logits", "seq_logits")]:
    emit("g_" + name, np.ascontiguousarray(v[key], dtype=np.int8).tobytes(), ".testdata", out / f"{name}.inc")
(out / "counts.h").write_text(
    f"#define N_BACKBONE {len(v['backbone_inputs'])}\n#define N_FRAMES {len(v['sequence_features'])}\n")
