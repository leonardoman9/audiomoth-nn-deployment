/**
 * virtual_arena.c
 * Flash-Resident Tensor Store - Uses Flash as read-only storage for TFLM tensors
 * Policy: ZERO scritture Flash a runtime - solo read + RAM cache LRU
 * Based on Gemini's suggestion adapted for internal Flash memory
 */

#include "virtual_arena.h"
#include "../../inc/audiomoth.h"
#include "../../emlib/inc/em_msc.h"
#include <string.h>
#include <stdlib.h>

// Flash base address from linker (points to .tensor_arena section)
// The .tensor_arena section is defined in the linker script and reserved in Flash

// Small RAM buffer for active tensors (actual working memory)
// Allocate dynamically to avoid .bss bloat
static uint8_t* ram_cache = NULL;

// Tensor tracking table
static VirtualTensor_t tensor_table[MAX_VIRTUAL_TENSORS];
static uint32_t num_tensors = 0;
static uint32_t access_counter = 0;

// Current RAM usage
static uint32_t ram_used = 0;

extern uint32_t __tensor_arena_start__;
extern uint32_t __tensor_arena_end__;

/**
 * Initialize the virtual arena system
 */
bool VirtualArena_Init(void) {
    // Initialize MSC for Flash operations
    MSC_Init();
    
    // Clear tensor table
    memset(tensor_table, 0, sizeof(tensor_table));
    num_tensors = 0;
    ram_used = 0;
    access_counter = 0;
    
    // Allocate RAM cache from heap to avoid .bss
    if (ram_cache == NULL) {
        ram_cache = (uint8_t*)malloc(RAM_CACHE_SIZE);
        if (ram_cache == NULL) {
            return false;  // Failed to allocate
        }
    }
    memset(ram_cache, 0, RAM_CACHE_SIZE);
    
    // Verify Flash arena is accessible
    uint32_t flash_start = (uint32_t)&__tensor_arena_start__;
    uint32_t flash_end = (uint32_t)&__tensor_arena_end__;
    uint32_t flash_size = flash_end - flash_start;
    
    // Debug: If symbols not properly set, use hardcoded addresses
    if (flash_size == 0) {
        flash_start = 0x30000;  // Start of Flash arena
        flash_end = 0x40000;    // End of Flash arena (64KB)
        flash_size = 0x10000;   // 64KB
    }
    
    if (flash_size < VIRTUAL_ARENA_SIZE) {
        return false;  // Not enough Flash space
    }
    
    return true;
}

/**
 * Allocate a virtual tensor (in Flash, not RAM)
 */
uint32_t VirtualArena_AllocTensor(uint32_t size, const char* name, bool is_const) {
    if (num_tensors >= MAX_VIRTUAL_TENSORS) {
        return INVALID_TENSOR_ID;
    }
    
    // Find space in virtual arena (Flash)
    static uint32_t flash_offset = 0;
    
    if (flash_offset + size > VIRTUAL_ARENA_SIZE) {
        return INVALID_TENSOR_ID;  // Out of virtual space
    }
    
    // Create tensor entry
    VirtualTensor_t* tensor = &tensor_table[num_tensors];
    tensor->id = num_tensors;
    tensor->size = (size + 7) & ~7;  // Align to 8 bytes
    tensor->flash_offset = flash_offset;
    tensor->ram_addr = NULL;
    tensor->in_ram = false;
    tensor->last_access = 0;
    tensor->is_const = is_const;     // Mark if read-only weight or mutable activation
    tensor->pinned = false;
    
    if (name) {
        strncpy(tensor->name, name, 31);
        tensor->name[31] = '\0';
    }
    
    flash_offset += size;
    num_tensors++;
    
    return tensor->id;
}

/**
 * Find least recently used unpinned tensor in RAM
 */
static uint32_t find_lru_tensor(void) {
    uint32_t lru_id = INVALID_TENSOR_ID;
    uint32_t min_access = 0xFFFFFFFF;
    
    for (uint32_t i = 0; i < num_tensors; i++) {
        VirtualTensor_t* t = &tensor_table[i];
        if (t->in_ram && !t->pinned && t->last_access < min_access) {
            min_access = t->last_access;
            lru_id = i;
        }
    }
    
    return lru_id;
}

/**
 * Evict a tensor from RAM to Flash
 */
static bool evict_tensor(uint32_t tensor_id) {
    if (tensor_id >= num_tensors) {
        return false;
    }
    
    VirtualTensor_t* tensor = &tensor_table[tensor_id];
    
    if (!tensor->in_ram || tensor->pinned) {
        return false;
    }
    
    // ⚠️ POLICY: NO FLASH WRITES A RUNTIME!
    // Flash è read-only per pesi/const. Tensori modificati restano in RAM.
    // Se evicted, i dati modificati vengono persi (OK per attivazioni temporanee)
    if (!tensor->is_const) {
        // Log warning: tensore modificato viene perso
        // In una implementazione reale, questo tensore dovrebbe essere
        // marcato come "must stay in RAM" o essere un peso read-only
    }
    
    // Free RAM
    ram_used -= tensor->size;
    tensor->ram_addr = NULL;
    tensor->in_ram = false;
    
    return true;
}

/**
 * Load tensor from Flash to RAM
 */
static bool load_tensor(uint32_t tensor_id) {
    if (tensor_id >= num_tensors) {
        return false;
    }
    
    VirtualTensor_t* tensor = &tensor_table[tensor_id];
    
    if (tensor->in_ram) {
        return true;  // Already in RAM
    }
    
    // Check if we need to evict something
    while (ram_used + tensor->size > RAM_CACHE_SIZE) {
        uint32_t lru_id = find_lru_tensor();
        if (lru_id == INVALID_TENSOR_ID) {
            return false;  // Can't evict anything
        }
        evict_tensor(lru_id);
    }
    
    // Allocate RAM
    // Simple allocation - just append (real implementation needs proper allocator)
    tensor->ram_addr = &ram_cache[ram_used];
    
    // Copy from Flash to RAM
    uint32_t flash_addr = (uint32_t)&__tensor_arena_start__ + tensor->flash_offset;
    memcpy(tensor->ram_addr, (void*)flash_addr, tensor->size);
    
    tensor->in_ram = true;
    ram_used += tensor->size;
    
    return true;
}

/**
 * Get tensor data pointer (loads to RAM if needed)
 */
void* VirtualArena_GetTensor(uint32_t tensor_id) {
    if (tensor_id >= num_tensors) {
        return NULL;
    }
    
    VirtualTensor_t* tensor = &tensor_table[tensor_id];
    
    // Update access time
    tensor->last_access = access_counter++;
    
    // Ensure in RAM
    if (!tensor->in_ram) {
        if (!load_tensor(tensor_id)) {
            return NULL;  // Failed to load
        }
    }
    
    return tensor->ram_addr;
}

// VirtualArena_MarkDirty() RIMOSSA
// Policy: NO SCRITTURE FLASH A RUNTIME
// Flash è read-only per pesi/modelli

/**
 * Pin tensor in RAM (prevent eviction)
 */
void VirtualArena_PinTensor(uint32_t tensor_id) {
    if (tensor_id < num_tensors) {
        tensor_table[tensor_id].pinned = true;
    }
}

/**
 * Unpin tensor (allow eviction)
 */
void VirtualArena_UnpinTensor(uint32_t tensor_id) {
    if (tensor_id < num_tensors) {
        tensor_table[tensor_id].pinned = false;
    }
}

/**
 * Get statistics for debugging
 */
void VirtualArena_GetStats(VirtualArenaStats_t* stats) {
    if (!stats) return;
    
    stats->total_tensors = num_tensors;
    stats->tensors_in_ram = 0;
    stats->ram_used = ram_used;
    stats->ram_total = RAM_CACHE_SIZE;
    stats->flash_used = 0;
    stats->flash_total = VIRTUAL_ARENA_SIZE;
    stats->swap_count = 0;  // TODO: Track this
    
    for (uint32_t i = 0; i < num_tensors; i++) {
        if (tensor_table[i].in_ram) {
            stats->tensors_in_ram++;
        }
        stats->flash_used += tensor_table[i].size;
    }
}