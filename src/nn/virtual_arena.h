/**
 * virtual_arena.h
 * Flash-Resident Tensor Store - Uses Flash as read-only storage for TFLM tensors
 * Policy: ZERO scritture Flash a runtime - solo read + RAM cache LRU
 * Implements Gemini's suggestion adapted for internal Flash
 */

#ifndef VIRTUAL_ARENA_H
#define VIRTUAL_ARENA_H

#include <stdint.h>
#include <stdbool.h>

// Configuration
#define VIRTUAL_ARENA_SIZE      (64 * 1024)    // 64KB in Flash (the "External SRAM")
#define RAM_CACHE_SIZE          (8 * 1024)     // 8KB actual RAM for GRU-64 tensors
#define MAX_VIRTUAL_TENSORS     32              // Reduced to save .bss
#define INVALID_TENSOR_ID       0xFFFFFFFF

// Tensor metadata
typedef struct {
    uint32_t id;                // Unique tensor ID
    uint32_t size;              // Size in bytes (aligned to 8)
    uint32_t flash_offset;      // Offset in Flash arena
    uint8_t* ram_addr;          // RAM address (if loaded)
    bool in_ram;                // Currently in RAM?
    bool is_const;              // true = peso read-only, false = attivazione
    bool pinned;                // Prevent eviction?
    uint32_t last_access;       // For LRU tracking
    char name[32];              // Debug name
} VirtualTensor_t;

// Statistics for monitoring
typedef struct {
    uint32_t total_tensors;
    uint32_t tensors_in_ram;
    uint32_t ram_used;
    uint32_t ram_total;
    uint32_t flash_used;
    uint32_t flash_total;
    uint32_t swap_count;
    uint32_t cache_hits;
    uint32_t cache_misses;
} VirtualArenaStats_t;

// API Functions
bool VirtualArena_Init(void);
uint32_t VirtualArena_AllocTensor(uint32_t size, const char* name, bool is_const);
void* VirtualArena_GetTensor(uint32_t tensor_id);
void VirtualArena_PinTensor(uint32_t tensor_id);
void VirtualArena_UnpinTensor(uint32_t tensor_id);
void VirtualArena_GetStats(VirtualArenaStats_t* stats);

// Helper macros for TFLM integration  
#define VIRTUAL_TENSOR_PTR(id) VirtualArena_GetTensor(id)
// Note: No WRITE macro - Flash is read-only!

#endif // VIRTUAL_ARENA_H