"""Copies the file-system part of the AudioMoth hardware library (audiomoth.c) verbatim, so the test runs
the library's own wrappers around FatFS. Usage: extract_fs_funcs.py audiomoth.c out.inc"""
import sys
s = open(sys.argv[1]).read()
a = s.index("bool AudioMoth_doesFileExist(char *filename){")
b = s.index("bool AudioMoth_closeFile(void) {")
b = s.index("\n}\n", b) + 3
open(sys.argv[2], "w").write(s[a:b])
