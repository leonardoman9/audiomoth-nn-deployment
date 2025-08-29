/****************************************************************************
 * audiomoth_flash_swap.ld
 * Linker script with Flash-based virtual SRAM for tensor arena
 * Based on Gemini's suggestion adapted for internal Flash
 *****************************************************************************/

MEMORY
{
	/* Use entire Flash for code (full 1MB) */
	FLASH_CODE (rx) : ORIGIN = 0x4000, LENGTH = 0xFC000 /* 1008KB for firmware */
	/* Use real external SRAM for tensor arena */
	EXT_SRAM (rwx) : ORIGIN = 0x80000000, LENGTH = 0x40000 /* 256KB external SRAM */
	RAM (rwx) : ORIGIN = 0x20000000, LENGTH = 0x8000 /* 32KB internal RAM */
}

ENTRY(Reset_Handler)

SECTIONS
{
  .text :
  {
    KEEP(*(.vectors))
    __Vectors_End = .;
    __Vectors_Size = __Vectors_End - __Vectors;
    __end__ = .;

    *(.text*)

    KEEP(*(.init))
    KEEP(*(.fini))

    /* .ctors */
    *crtbegin.o(.ctors)
    *crtbegin?.o(.ctors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .ctors)
    *(SORT(.ctors.*))
    *(.ctors)

    /* .dtors */
    *crtbegin.o(.dtors)
    *crtbegin?.o(.dtors)
    *(EXCLUDE_FILE(*crtend?.o *crtend.o) .dtors)
    *(SORT(.dtors.*))
    *(.dtors)

    *(.rodata*)
    KEEP(*(.eh_frame*))

  } > FLASH_CODE

  .ARM.extab :
  {
    *(.ARM.extab* .gnu.linkonce.armextab.*)
  } > FLASH_CODE

  __exidx_start = .;
  
  .ARM.exidx :
  {
    *(.ARM.exidx* .gnu.linkonce.armexidx.*)
  } > FLASH_CODE

  __exidx_end = .;
  __etext = .;

  /* Tensor arena in real external SRAM */
  .tensor_arena (NOLOAD): ALIGN(32)
  {
    . = ALIGN(32);
    __tensor_arena_start__ = .;
    . = . + 0x10000; /* Reserve 64KB for tensor arena */
    __tensor_arena_end__ = .;
  } > EXT_SRAM

  .data : AT (__etext)
  {
    __data_start__ = .;
    *(vtable)
    *(.data*)
    . = ALIGN (4);
    *(.ram)

    . = ALIGN(4);
    
    /* preinit data */
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP(*(.preinit_array))
    PROVIDE_HIDDEN (__preinit_array_end = .);

    . = ALIGN(4);
    
    /* init data */
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP(*(SORT(.init_array.*)))
    KEEP(*(.init_array))
    PROVIDE_HIDDEN (__init_array_end = .);

    . = ALIGN(4);
    
    /* finit data */
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP(*(SORT(.fini_array.*)))
    KEEP(*(.fini_array))
    PROVIDE_HIDDEN (__fini_array_end = .);
    KEEP(*(.jcr*))
    
    . = ALIGN(4);
    
    /* All data end */
    __data_end__ = .;

  } > RAM

  .bss :
  {
    . = ALIGN(4);
    __bss_start__ = .;
    *(.bss*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end__ = .;
  } > RAM

  .heap (COPY):
  {
    __HeapBase = .;
    __end__ = .;
    end = __end__;
    _end = __end__;
    KEEP(*(.heap*))
    __HeapLimit = .;
  } > RAM

  /* .stack_dummy section doesn't contains any symbols. It is only
   * used for linker to calculate size of stack sections, and assign
   * values to stack symbols later */
  
  .stack_dummy (COPY):
  {
    KEEP(*(.stack*))
  } > RAM

  /* Set stack top to end of RAM, and stack limit move down by
   * size of stack_dummy section */
  
  __StackTop = ORIGIN(RAM) + LENGTH(RAM);
  __StackLimit = __StackTop - SIZEOF(.stack_dummy);
  PROVIDE(__stack = __StackTop);

  /* Check if data + heap + stack exceeds RAM limit */
  ASSERT(__StackLimit >= __HeapLimit, "RAM memory overflowed!")

  /* Check if FLASH usage exceeds FLASH size */
  ASSERT( LENGTH(FLASH_CODE) >= (__etext + SIZEOF(.data)), "FLASH memory overflowed!")
}