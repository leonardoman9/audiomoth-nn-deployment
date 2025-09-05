#ifndef TENSOR_ARENA_H
#define TENSOR_ARENA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Match the linker script's EXT_SRAM size
#define TENSOR_ARENA_SIZE      (64 * 1024)     // 64KB ultra-ridotti per Direct Inference

typedef struct {
    uint32_t total_size;
    uint32_t used_size;
    uint32_t free_size;
    uint32_t buffer_addr;
    bool init_success;
} TensorArenaStats_t;

// Lifecycle
bool TensorArena_Init(void);

// Getters
uint8_t* TensorArena_GetBuffer(void);
uint32_t TensorArena_GetSize(void);
void TensorArena_GetStats(TensorArenaStats_t* stats);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_ARENA_H