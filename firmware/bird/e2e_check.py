"""Device pipeline on the host: C front-end -> int8 TFLite backbone -> C tail, vs the PyTorch logits.
Usage: python e2e_check.py REF.npz MODEL.tflite LIBFE LIBTAIL"""
import ctypes, sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType
ref = np.load(sys.argv[1]); model, libfe, libtail = sys.argv[2], sys.argv[3], sys.argv[4]
fe_lib, tail_lib = ctypes.CDLL(libfe), ctypes.CDLL(libtail)
fe_lib.BirdFE_SizeOf.restype = ctypes.c_uint32
fe_lib.BirdFE_Init.argtypes = [ctypes.c_void_p]
fe_lib.BirdFE_Push.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
fe_lib.BirdFE_Finish.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]; fe_lib.BirdFE_Finish.restype = ctypes.c_bool
fe_lib.BirdFE_Reset.argtypes = [ctypes.c_void_p]
fe = ctypes.create_string_buffer(fe_lib.BirdFE_SizeOf())
fe_lib.BirdFE_Init(fe)
class Tail(ctypes.Structure):
    _fields_ = [("h", ctypes.c_float * 64), ("ctx", ctypes.c_float * 64), ("mx", ctypes.c_float), ("sm", ctypes.c_float)]
tail_lib.BirdTail_Step.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
tail_lib.BirdTail_Finish.argtypes = [ctypes.POINTER(Tail), ctypes.POINTER(ctypes.c_float)]
tail_lib.BirdTail_Reset.argtypes = [ctypes.POINTER(Tail)]
it = Interpreter(model_path=model, experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1); it.allocate_tensors()
i, o = it.get_input_details()[0], it.get_output_details()[0]
isc, izp = i["quantization"]; osc, ozp = o["quantization"]
got = []
for pcm in ref["pcm"]:
    fe_lib.BirdFE_Reset(fe)
    for off in range(0, len(pcm), 512):
        part = np.ascontiguousarray(pcm[off:off + 512]); fe_lib.BirdFE_Push(fe, part.ctypes.data, len(part))
    q = np.zeros((301, 64), np.int8); assert fe_lib.BirdFE_Finish(fe, q.ctypes.data, isc, izp)
    it.set_tensor(i["index"], q.reshape(i["shape"])); it.invoke()
    feats = ((it.get_tensor(o["index"]).reshape(301, 32).astype(np.float32) - ozp) * osc).astype(np.float32)
    t = Tail(); tail_lib.BirdTail_Reset(ctypes.byref(t))
    for row in feats: tail_lib.BirdTail_Step(ctypes.byref(t), row.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    out = (ctypes.c_float * 9)(); tail_lib.BirdTail_Finish(ctypes.byref(t), out); got.append(np.array(out))
got = np.stack(got); lg = ref["logits"]
def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
p_ref, p_got = sm(lg), sm(got)
conf = p_ref.max(1); flip = got.argmax(1) != lg.argmax(1)
print(f"windows={len(lg)} | argmax agree={int((~flip).sum())}/{len(lg)} | logits max|d|={np.abs(got-lg).max():.3f} mean={np.abs(got-lg).mean():.3f} | max |dprob|={np.abs(p_got-p_ref).max():.3f}")
print("flipped windows' reference confidence:", np.round(np.sort(conf[flip]), 2).tolist())
print("windows with reference confidence >= 0.5:", int((conf >= 0.5).sum()), "flipped among them:", int((flip & (conf >= 0.5)).sum()))
