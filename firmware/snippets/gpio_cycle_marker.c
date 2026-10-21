// Minimal C snippet: cycle counter + GPIO marker around a block
// For ARM Cortex‑M (e.g., EFM32), using DWT->CYCCNT for cycle counts.
// Integrate this into your firmware as needed; adjust GPIO macros/APIs.

#include <stdint.h>

// Include CMSIS core header for your MCU to get DWT/DEMCR definitions
// e.g., for Cortex‑M3/M4: #include "core_cm3.h" or "core_cm4.h"
#include "core_cm3.h"

// Replace these with your platform's GPIO controls (example: EFM32 emlib)
// Example from EFM32:
//   #include "em_gpio.h"
//   #define MARK_PORT gpioPortE
//   #define MARK_PIN  10
//   #define GPIO_INIT()      GPIO_PinModeSet(MARK_PORT, MARK_PIN, gpioModePushPull, 0)
//   #define GPIO_MARK_HIGH() GPIO_PinOutSet(MARK_PORT, MARK_PIN)
//   #define GPIO_MARK_LOW()  GPIO_PinOutClear(MARK_PORT, MARK_PIN)

#ifndef GPIO_INIT
#define GPIO_INIT()      do {} while (0)
#endif
#ifndef GPIO_MARK_HIGH
#define GPIO_MARK_HIGH() do {} while (0)
#endif
#ifndef GPIO_MARK_LOW
#define GPIO_MARK_LOW()  do {} while (0)
#endif

static inline void dwt_enable_cycle_counter(void) {
    // Enable TRC
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    // Reset cycle counter
    DWT->CYCCNT = 0;
    // Enable cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline uint32_t dwt_get_cycles(void) {
    return DWT->CYCCNT;
}

// Call once at boot
void perf_init_marker_and_counter(void) {
    GPIO_INIT();
    dwt_enable_cycle_counter();
}

// Example measurement wrapper: toggles GPIO around N invocations and returns cycles
// Convert cycles to time by: seconds = cycles / SystemCoreClock
typedef void (*inference_fn_t)(void *ctx);

uint32_t perf_measure_block_cycles(inference_fn_t fn, void *ctx, int N) {
    if (N <= 0 || fn == 0) return 0;
    // Optional: fence/flush as appropriate for your platform
    volatile uint32_t start_cycles, end_cycles;
    GPIO_MARK_HIGH();
    start_cycles = dwt_get_cycles();
    for (int i = 0; i < N; ++i) {
        fn(ctx);
    }
    end_cycles = dwt_get_cycles();
    GPIO_MARK_LOW();
    return (end_cycles - start_cycles);
}

// Usage sketch:
//   perf_init_marker_and_counter();
//   uint32_t cyc = perf_measure_block_cycles(run_inference_wrapper, ctx, N);
//   float t_s = (float)cyc / (float)SystemCoreClock;
//   float t_inf_avg = t_s / (float)N;

