"""Embeds one raw PCM window and the expected logits (host pipeline and PyTorch) as C arrays.
Usage (TF env): python gen_pipeline_vec.py REF.npz INDEX MODEL.tflite LIBFE LIBTAIL OUT.cc"""
import ctypes, sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
ref, idx, model, libfe, libtail, out = np.load(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
pcm = ref["pcm"][idx]
fe_lib, tail_lib = ctypes.CDLL(libfe), ctypes.CDLL(libtail)
fe_lib.BirdFE_SizeOf.restype = ctypes.c_uint32
fe_lib.BirdFE_Init.argtypes = [ctypes.c_void_p]
fe_lib.BirdFE_Push.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
fe_lib.BirdFE_Finish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]; fe_lib.BirdFE_Finish.restype = ctypes.c_bool
fe = ctypes.create_string_buffer(fe_lib.BirdFE_SizeOf())
fe_lib.BirdFE_Init(fe)
class Tail(ctypes.Structure):
    _fields_ = [("h", ctypes.c_float * 64), ("ctx", ctypes.c_float * 64), ("mx", ctypes.c_float), ("sm", ctypes.c_float)]
tail_lib.BirdTail_Step.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
tail_lib.BirdTail_Finish.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
tail_lib.BirdTail_Reset.argtypes = [ctypes.POINTER(Tail)]
it = Interpreter(model_path=model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
for off in range(0, len(pcm), 512):
    p = np.ascontiguousarray(pcm[off:off + 512]); fe_lib.BirdFE_Push(fe, p.ctypes.data, len(p))
q = np.zeros((301, 64), np.int8); assert fe_lib.BirdFE_Finish(fe, q.ctypes.data, i["quantization"][0], i["quantization"][1])
it.set_tensor(i["index"], q.reshape(i["shape"])); it.invoke()
f = ((it.get_tensor(o["index"]).reshape(301, 32).astype(np.float32) - o["quantization"][1]) * o["quantization"][0]).astype(np.float32)
t = Tail(); tail_lib.BirdTail_Reset(ctypes.byref(t))
for row in f: tail_lib.BirdTail_Step(ctypes.byref(t), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
host = (ctypes.c_float * tail_lib.BirdTail_Classes())(); tail_lib.BirdTail_Finish(ctypes.byref(t), host)
print("window", ref["names"][idx], "| host pipeline logits", np.round(np.array(host), 3), "| torch", np.round(ref["logits"][idx], 3))
def arr(name, a, ctype, fmt):
    return f'extern "C" {ctype} {name}[{a.size}] __attribute__((section(".testdata"), aligned(16))) = {{{",".join(fmt(v) for v in a.reshape(-1))}}};\n'
open(out, "w").write("#include <stdint.h>\n" + arr("g_pcm", pcm, "const int16_t", str)
    + arr("g_host_logits", np.array(host, np.float32), "const float", lambda v: repr(float(v)) + "f")
    + arr("g_torch_logits", ref["logits"][idx].astype(np.float32), "const float", lambda v: repr(float(v)) + "f"))
