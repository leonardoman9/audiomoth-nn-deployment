#!/bin/sh
# Run out/fw.elf under QEMU (Cortex-M4, mps2-an386); output comes via semihosting.
cd "$(dirname "$0")"
qemu-system-arm -M mps2-an386 -cpu cortex-m4 -display none -serial none -monitor none \
  -semihosting-config enable=on,target=native -kernel out/fw.elf
