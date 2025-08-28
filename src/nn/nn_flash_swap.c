#include "../../inc/nn/nn_model.h"
#include "../../emlib/inc/em_msc.h"
#include <string.h>

// Flash swap memory addresses (starting after firmware)
#define FLASH_SWAP_START    0x18000     // Start after firmware code
#define FLASH_SWAP_SIZE     (64 * 1024)  // 64KB for tensor storage
#define NN_FLASH_PAGE_SIZE  2048         // EFM32WG page size (avoid conflict)

// Tensor swap slots in Flash
#define TENSOR_SLOT_SIZE    (4 * 1024)   // 4KB per slot
#define MAX_TENSOR_SLOTS    16           // 16 slots = 64KB total

typedef struct {
    uint32_t flash_addr;
    uint32_t size;
    bool in_use;
} TensorSlot_t;

static TensorSlot_t g_tensor_slots[MAX_TENSOR_SLOTS];
static bool g_flash_swap_initialized = false;

// Initialize Flash swap system
bool NN_FlashSwap_Init(void) {
    if (g_flash_swap_initialized) {
        return true;
    }
    
    // Initialize MSC for Flash operations
    MSC_Init();
    
    // Setup tensor slots
    for (int i = 0; i < MAX_TENSOR_SLOTS; i++) {
        g_tensor_slots[i].flash_addr = FLASH_SWAP_START + (i * TENSOR_SLOT_SIZE);
        g_tensor_slots[i].size = TENSOR_SLOT_SIZE;
        g_tensor_slots[i].in_use = false;
    }
    
    g_flash_swap_initialized = true;
    return true;
}

// Allocate a tensor slot in Flash
int NN_FlashSwap_AllocateSlot(uint32_t size) {
    if (!g_flash_swap_initialized || size > TENSOR_SLOT_SIZE) {
        return -1;
    }
    
    for (int i = 0; i < MAX_TENSOR_SLOTS; i++) {
        if (!g_tensor_slots[i].in_use) {
            g_tensor_slots[i].in_use = true;
            return i;
        }
    }
    
    return -1; // No free slots
}

// Store tensor data to Flash
bool NN_FlashSwap_StoreTensor(int slot_id, const void* data, uint32_t size) {
    if (slot_id < 0 || slot_id >= MAX_TENSOR_SLOTS || !g_tensor_slots[slot_id].in_use) {
        return false;
    }
    
    if (size > TENSOR_SLOT_SIZE) {
        return false;
    }
    
    uint32_t flash_addr = g_tensor_slots[slot_id].flash_addr;
    
    // Erase Flash pages first
    uint32_t pages_to_erase = (size + NN_FLASH_PAGE_SIZE - 1) / NN_FLASH_PAGE_SIZE;
    for (uint32_t i = 0; i < pages_to_erase; i++) {
        MSC_ErasePage((uint32_t*)(flash_addr + i * NN_FLASH_PAGE_SIZE));
    }
    
    // Write data to Flash
    MSC_Status_TypeDef status = MSC_WriteWord((uint32_t*)flash_addr, data, size);
    return (status == mscReturnOk);
}

// Load tensor data from Flash
bool NN_FlashSwap_LoadTensor(int slot_id, void* data, uint32_t size) {
    if (slot_id < 0 || slot_id >= MAX_TENSOR_SLOTS || !g_tensor_slots[slot_id].in_use) {
        return false;
    }
    
    if (size > TENSOR_SLOT_SIZE) {
        return false;
    }
    
    uint32_t flash_addr = g_tensor_slots[slot_id].flash_addr;
    memcpy(data, (void*)flash_addr, size);
    return true;
}

// Free a tensor slot
void NN_FlashSwap_FreeSlot(int slot_id) {
    if (slot_id >= 0 && slot_id < MAX_TENSOR_SLOTS) {
        g_tensor_slots[slot_id].in_use = false;
    }
}

// Enhanced TFLM arena that uses Flash swap
typedef struct {
    uint8_t* ram_buffer;      // Small RAM buffer (4KB)
    uint32_t ram_size;
    int current_slot;         // Currently loaded tensor slot
    bool dirty;               // RAM buffer needs to be written back
} SwapArena_t;

static SwapArena_t g_swap_arena;
static uint8_t g_ram_buffer[1024];      // 1KB RAM buffer (minimal for .bss limit)

bool NN_SwapArena_Init(void) {
    if (!NN_FlashSwap_Init()) {
        return false;
    }
    
    g_swap_arena.ram_buffer = g_ram_buffer;
    g_swap_arena.ram_size = sizeof(g_ram_buffer);
    g_swap_arena.current_slot = -1;
    g_swap_arena.dirty = false;
    
    return true;
}

// Get RAM buffer for current tensor (swap in from Flash if needed)
uint8_t* NN_SwapArena_GetBuffer(int tensor_id, uint32_t size) {
    if (size > g_swap_arena.ram_size) {
        return NULL;  // Tensor too large
    }
    
    // If different tensor, save current and load new
    if (g_swap_arena.current_slot != tensor_id) {
        // Save current tensor to Flash if dirty
        if (g_swap_arena.dirty && g_swap_arena.current_slot >= 0) {
            NN_FlashSwap_StoreTensor(g_swap_arena.current_slot, 
                                   g_swap_arena.ram_buffer, 
                                   g_swap_arena.ram_size);
            g_swap_arena.dirty = false;
        }
        
        // Load new tensor from Flash
        if (tensor_id >= 0) {
            NN_FlashSwap_LoadTensor(tensor_id, g_swap_arena.ram_buffer, size);
        }
        
        g_swap_arena.current_slot = tensor_id;
    }
    
    return g_swap_arena.ram_buffer;
}

// Mark buffer as dirty (needs to be written back to Flash)
void NN_SwapArena_MarkDirty(void) {
    g_swap_arena.dirty = true;
}