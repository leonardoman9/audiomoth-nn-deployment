// Host-emulation stand-in for the AudioMoth hardware header (QEMU has no EFM32).
#pragma once
#include <stdbool.h>
static inline bool AudioMoth_enableExternalSRAM(void) { return true; }
