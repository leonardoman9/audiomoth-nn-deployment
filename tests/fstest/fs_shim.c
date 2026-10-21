/* The AudioMoth library's file wrappers (copied verbatim by extract_fs_funcs.py) on top of the real FatFS. */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ff.h"

static FATFS fatfs;
static FIL file;
static UINT bw, br;

#include "fs_audiomoth_funcs.inc"

bool Shim_mount(void) { return f_mount(&fatfs, "", 1) == FR_OK; }
void Shim_unmount(void) { f_mount(NULL, "", 0); }
