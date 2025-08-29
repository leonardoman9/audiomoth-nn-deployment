/**
 * tensor_arena.c
 * Real External SRAM tensor arena for TensorFlow Lite Micro
 * Uses the actual AudioMoth external SRAM instead of Flash virtualization
 */

#include "tensor_arena.h"
#include "../../inc/audiomoth.h"
#include <string.h>

// Tensor arena in external SRAM (defined by linker)
// This is placed in the .tensor_arena section which maps to EXT_SRAM
__attribute__((section(".tensor_arena"), aligned(32)))
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

// External symbols from linker
extern uint32_t __tensor_arena_start__;
extern uint32_t __tensor_arena_end__;

/**
 * Initialize the tensor arena system
 * Must enable external SRAM before use
 */
bool TensorArena_Init(void) {
    // Enable external SRAM hardware
    if (!AudioMoth_enableExternalSRAM()) {
        return false;  // Failed to enable SRAM (might be v4+ hardware)
    }
    
    // Verify arena is mapped correctly
    uint32_t arena_start = (uint32_t)&__tensor_arena_start__;
    uint32_t arena_end = (uint32_t)&__tensor_arena_end__;
    uint32_t arena_size = arena_end - arena_start;
    
    // Check if arena is in external SRAM range
    if (arena_start < AM_EXTERNAL_SRAM_START_ADDRESS || 
        arena_end > (AM_EXTERNAL_SRAM_START_ADDRESS + AM_EXTERNAL_SRAM_SIZE_IN_BYTES)) {
        return false;  // Arena not properly mapped to external SRAM
    }
    
    if (arena_size < TENSOR_ARENA_SIZE) {
        return false;  // Arena too small
    }
    
    // Clear the arena
    memset(tensor_arena, 0, TENSOR_ARENA_SIZE);
    
    return true;
}

/**
 * Get the tensor arena pointer for TensorFlow Lite Micro
 */
uint8_t* TensorArena_GetBuffer(void) {
    return tensor_arena;
}

/**
 * Get the tensor arena size
 */
uint32_t TensorArena_GetSize(void) {
    return TENSOR_ARENA_SIZE;
}

/**
 * Get arena statistics for debugging
 */
void TensorArena_GetStats(TensorArenaStats_t* stats) {
    if (!stats) return;
    
    uint32_t arena_start = (uint32_t)&__tensor_arena_start__;
    uint32_t arena_end = (uint32_t)&__tensor_arena_end__;
    
    stats->arena_start_addr = arena_start;
    stats->arena_end_addr = arena_end;
    stats->arena_size = arena_end - arena_start;
    stats->buffer_addr = (uint32_t)tensor_arena;
    stats->buffer_size = TENSOR_ARENA_SIZE;
    stats->is_external_sram = (arena_start >= AM_EXTERNAL_SRAM_START_ADDRESS && 
                              arena_end <= (AM_EXTERNAL_SRAM_START_ADDRESS + AM_EXTERNAL_SRAM_SIZE_IN_BYTES));
}

/**
 * Cleanup (disable external SRAM)
 */
void TensorArena_Cleanup(void) {
    AudioMoth_disableExternalSRAM();
}