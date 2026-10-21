"""Flash (.text/.rodata) by contributor from a GNU ld map. Usage: map_sizes.py fw.map"""
import collections, re, sys
lines = open(sys.argv[1], errors="ignore").read().split("\n")
sizes = collections.Counter(); inmap = False
pat = re.compile(r"^\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(\S+)$")
for i, l in enumerate(lines):
    if l.startswith("Linker script and memory map"): inmap = True
    if not inmap: continue
    m2 = re.match(r"^ (\.text\S*|\.rodata\S*)\s+(0x[0-9a-f]+)\s+(0x[0-9a-f]+)\s+(\S+)", l)
    m = re.match(r"^ (\.text\S*|\.rodata\S*)\s*$", l)
    if m2: f, s = m2.group(4), int(m2.group(3), 16)
    elif m and i + 1 < len(lines) and pat.match(lines[i + 1]): n = pat.match(lines[i + 1]); f, s = n.group(3), int(n.group(2), 16)
    else: continue
    if "libtensorflow" in f: g = "TFLM + CMSIS-NN"
    elif "bird_model" in f: g = "model"
    elif "bird_tail" in f: g = "C tail"
    elif "nn_bird" in f or "qemu_bird" in f: g = "wrapper"
    elif "/arm-none-eabi/" in f or "libc" in f or "libm" in f or "libgcc" in f or "libstdc" in f: g = "libc/libm/libgcc"
    else: g = f.split("/")[-1]
    sizes[g] += s
for g, s in sizes.most_common(8): print("%8d %s" % (s, g))
print("%8d TOTAL (text+rodata, test main included)" % sum(sizes.values()))
