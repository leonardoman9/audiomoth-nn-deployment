"""Embeds the int8 bird backbone and one self-test clip (with expected outputs) as C arrays.
Run in the TensorFlow env (ai_edge_litert, numpy):
  python gen_bird_fw_data.py MODEL.tflite SPECS.npz TAIL_LIB.dylib OUT_DIR
Expected values come from the TFLite reference interpreter + the host build of bird_tail.c.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

model, specs, lib, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
it = Interpreter(model_path=str(model), experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
spec = np.load(specs)["eval"][0]
sc, zp = i["quantization"]
x = np.clip(np.rint(spec / sc) + zp, -128, 127).astype(np.int8)
it.set_tensor(i["index"], x.reshape(i["shape"])); it.invoke()
y = it.get_tensor(o["index"]).reshape(301, 32).copy()
osc, ozp = o["quantization"]
feats = ((y.astype(np.float32) - ozp) * osc).astype(np.float32)

class Tail(ctypes.Structure):
    _fields_ = [("h", ctypes.c_float * 64), ("ctx", ctypes.c_float * 64), ("mx", ctypes.c_float), ("sm", ctypes.c_float)]
c = ctypes.CDLL(lib)
c.BirdTail_Step.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
c.BirdTail_Finish.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
c.BirdTail_Reset.argtypes = [ctypes.POINTER(Tail)]
t = Tail(); c.BirdTail_Reset(ctypes.byref(t))
for row in feats:
    c.BirdTail_Step(ctypes.byref(t), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
logits = (ctypes.c_float * 9)(); c.BirdTail_Finish(ctypes.byref(t), logits)
print("expected logits", np.round(np.array(logits), 4))


def emit(name, a, ctype, section, fmt):
    body = ",".join(fmt(v) for v in a.reshape(-1))
    return f'extern "C" {ctype} {name}[{a.size}] __attribute__((section("{section}"), aligned(16))) = {{{body}}};\n'


(out / "bird_model.cc").write_text('#include <stdint.h>\n' + emit("g_bird_model", np.frombuffer(model.read_bytes(), np.uint8), "const unsigned char", ".rodata", str))
(out / "bird_vec.cc").write_text('#include <stdint.h>\n' + emit("g_bird_in", x, "const int8_t", ".testdata", str)
    + emit("g_bird_out", y, "const int8_t", ".testdata", str)
    + emit("g_bird_logits", np.array(logits, dtype=np.float32), "const float", ".testdata", lambda v: repr(float(v)) + "f"))
