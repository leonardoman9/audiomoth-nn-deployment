/**
 * tensor_arena.h
 * Real External SRAM tensor arena for TensorFlow Lite Micro
 * Replaces the Flash-based virtual arena with actual external SRAM
 */

#ifndef TENSOR_ARENA_H
#define TENSOR_ARENA_H

#include <stdint.h>
#include <stdbool.h>

// Configuration - using 64KB of the 256KB external SRAM
#define TENSOR_ARENA_SIZE      (64 * 1024)    // 64KB in external SRAM

// Statistics for monitoring
typedef struct {
    uint32_t arena_start_addr;      // Linker symbol address
    uint32_t arena_end_addr;        // Linker symbol address  
    uint32_t arena_size;            // Size from linker
    uint32_t buffer_addr;           // Actual buffer address
    uint32_t buffer_size;           // Buffer size
    bool is_external_sram;          // True if properly mapped to external SRAM
} TensorArenaStats_t;

// API Functions
bool TensorArena_Init(void);
uint8_t* TensorArena_GetBuffer(void);
uint32_t TensorArena_GetSize(void);
void TensorArena_GetStats(TensorArenaStats_t* stats);
void TensorArena_Cleanup(void);

#endif // TENSOR_ARENA_H