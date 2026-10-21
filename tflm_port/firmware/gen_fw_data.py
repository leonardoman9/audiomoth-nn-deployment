"""Models + self-test vectors for the AudioMoth firmware.
Usage: python gen_fw_data.py WRENNET_ROOT OUT_DIR   (BACKBONE_TFLITE/STREAMING_TFLITE override the models)
"""
import os
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
kws = root / "birds_distillation_edge/convert-and-quantize/streamable-kws"
v = np.load(root / "docs/deployment/reference/vectors.npz")


def emit(name, raw):
    body = ",".join(str(b) for b in np.frombuffer(raw, dtype=np.uint8))
    (out / f"{name}.cc").write_text(
        f'#include <stdint.h>\nextern "C" {{\nalignas(16) extern const unsigned char {name}[] = {{{body}}};\n}}\n')


emit("g_backbone_model", Path(os.environ.get("BACKBONE_TFLITE", kws / "backbone_int8_FINAL.tflite")).read_bytes())
emit("g_streaming_model", Path(os.environ.get("STREAMING_TFLITE", kws / "streaming_int8_FINAL.tflite")).read_bytes())
# The reference sequence starts from a zero state: its first 18 frames are the
# streaming-side features of backbone case 0 (requantized), so they double as the
# expected output of NNReal_Backbone.
for key, name, n in [("backbone_inputs", "g_selftest_mel", 1), ("sequence_features", "g_selftest_features", 18),
                     ("sequence_hidden_outputs", "g_selftest_hidden", 18), ("sequence_logits", "g_selftest_logits", 18)]:
    emit(name, np.ascontiguousarray(v[key][:n], dtype=np.int8).tobytes())
