/**
 * tensor_arena.c
 * Real External SRAM tensor arena for TensorFlow Lite Micro
 * Uses the actual AudioMoth external SRAM instead of Flash virtualization
 */

#include "tensor_arena.h"
#include "audiomoth.h"
#include <string.h>

// External symbols from linker script
extern uint8_t __tensor_arena_start__[];
extern uint8_t __tensor_arena_end__[];

// For debugging purposes
static uint32_t dbg_arena_size = 0;
static uint32_t dbg_buf_addr = 0;
static uint32_t dbg_check_pass = 0;

bool TensorArena_Init(void) {
    
    if (!AudioMoth_enableExternalSRAM()) {
        dbg_check_pass = 0; // FAIL
        return false;
    }

    uint32_t arena_start = (uint32_t)__tensor_arena_start__;
    uint32_t arena_end = (uint32_t)__tensor_arena_end__;
    uint32_t arena_size = arena_end - arena_start;

    // --- DEBUG ---
    dbg_arena_size  = arena_size;
    dbg_buf_addr    = (uint32_t)__tensor_arena_start__;
    // ---

    if (arena_start < AM_EXTERNAL_SRAM_START_ADDRESS || 
        arena_end > (AM_EXTERNAL_SRAM_START_ADDRESS + AM_EXTERNAL_SRAM_SIZE_IN_BYTES)) {
        dbg_check_pass = 0; // FAIL
        return false;
    }

    if (arena_size < TENSOR_ARENA_SIZE) {
        dbg_check_pass = 0; // FAIL
        return false;
    }

    // Clear the entire arena using actual calculated size
    memset(__tensor_arena_start__, 0, arena_size);

    dbg_check_pass = 1;        // OK
    return true;
}

uint8_t* TensorArena_GetBuffer(void) {
    return __tensor_arena_start__;
}

uint32_t TensorArena_GetSize(void) {
    uint32_t arena_start = (uint32_t)__tensor_arena_start__;
    uint32_t arena_end = (uint32_t)__tensor_arena_end__;
    return arena_end - arena_start;
}

void TensorArena_GetStats(TensorArenaStats_t* stats) {
    if (!stats) return;
    
    stats->total_size = TensorArena_GetSize();
    stats->buffer_addr = (uint32_t)__tensor_arena_start__;
    stats->init_success = (dbg_check_pass == 1);

    // Placeholder for future use
    stats->used_size = 0; 
    stats->free_size = stats->total_size;
}