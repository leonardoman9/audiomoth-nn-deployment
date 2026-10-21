#include <stdint.h>
#include <stddef.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern int main(void);
extern void __libc_init_array(void);

void Reset_Handler(void) {
  *(volatile uint32_t *)0xE000ED88 |= 0xFu << 20;  /* enable FPU (CP10/CP11) */
  __asm volatile("dsb\n isb");
  for (uint32_t *s = &_sidata, *d = &_sdata; d < &_edata;) *d++ = *s++;
  /* Paint everything from .bss end to the top of RAM to measure stack depth. */
  for (uint32_t *p = &_sbss; p < &_ebss;) *p++ = 0;
  for (uint32_t *p = &_ebss; p < (uint32_t *)__builtin_frame_address(0) - 8;) *p++ = 0xDEADBEEF;
  __libc_init_array();
  main();
  for (;;) {}
}
void Default_Handler(void) { for (;;) {} }

__attribute__((section(".vectors"), used)) const void *g_vectors[] = {
    &_estack, Reset_Handler, Default_Handler, Default_Handler, Default_Handler,
    Default_Handler, Default_Handler, 0, 0, 0, 0, Default_Handler, Default_Handler,
    0, Default_Handler, Default_Handler};
