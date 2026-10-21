"""Copies the file-system part of the AudioMoth hardware library (audiomoth.c) verbatim, so the test runs
the library's own wrappers around FatFS. Usage: extract_fs_funcs.py audiomoth.c out.inc   (doesFileExist ... makeDirectory)"""
import sys
s = open(sys.argv[1]).read()
a = s.index("bool AudioMoth_doesFileExist(char *filename){")
b = s.index("bool AudioMoth_makeDirectory(char *folderName) {")  # last of the file functions the app uses
b = s.index("\n}\n", b) + 3
open(sys.argv[2], "w").write(s[a:b])
