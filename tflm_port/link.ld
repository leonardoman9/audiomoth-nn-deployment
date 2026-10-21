ENTRY(Reset_Handler)
/* AudioMoth (EFM32WG380F256): 256 KB Flash, 32 KB RAM. Test vectors live in a
   separate region that is not part of the firmware image. */
MEMORY {
  FLASH (rx)   : ORIGIN = 0x00000000, LENGTH = 256K
  TESTDATA (r) : ORIGIN = 0x00100000, LENGTH = 512K
  RAM (rwx)    : ORIGIN = 0x20000000, LENGTH = 32K
  /* AudioMoth: 256 KB external SRAM at 0x80000000. QEMU mps2 has no memory
     there, so a stand-in region at 0x20100000 is used for the emulation. */
  EXTSRAM (rwx): ORIGIN = 0x20100000, LENGTH = 64K
}
_estack = ORIGIN(RAM) + LENGTH(RAM);
SECTIONS {
  .text : { KEEP(*(.vectors)) *(.text*) *(.rodata*) *(.init) *(.fini) . = ALIGN(4);
            __preinit_array_start = .; KEEP(*(.preinit_array*)) __preinit_array_end = .;
            __init_array_start = .; KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array)) __init_array_end = .; } > FLASH
  .ARM.exidx : { *(.ARM.exidx*) } > FLASH
  .testdata : { *(.testdata) } > TESTDATA
  .extsram (NOLOAD) : { *(.extsram*) } > EXTSRAM
  _sidata = LOADADDR(.data);
  .data : { _sdata = .; *(.data*) . = ALIGN(4); _edata = .; } > RAM AT > FLASH
  .bss (NOLOAD) : { _sbss = .; *(.bss*) *(COMMON) . = ALIGN(8); _ebss = .; } > RAM
  end = _ebss;
}
