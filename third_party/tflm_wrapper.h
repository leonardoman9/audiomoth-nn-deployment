#ifndef TFLM_WRAPPER_H
#define TFLM_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Includiamo common.h per la definizione di TfLiteTensor
#include "tensorflow/lite/core/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

// Puntatore opaco alla nostra struttura di implementazione C++.
// Il codice C sa solo che esiste, ma non può vederne il contenuto.
typedef struct TFLM_Interpreter_s TFLM_Interpreter_t;

/**
 * @brief Crea e inizializza un interprete TFLM per un dato modello.
 * 
 * @param model_data Puntatore ai dati del modello FlatBuffer.
 * @param model_size Dimensione dei dati del modello.
 * @param arena Puntatore all'area di memoria per i tensori.
 * @param arena_size Dimensione dell'arena.
 * @return Un handle all'interprete creato, o NULL in caso di fallimento.
 */
TFLM_Interpreter_t* tflm_create_interpreter(
    const unsigned char* model_data,
    size_t model_size,
    uint8_t* arena,
    size_t arena_size);

/**
 * @brief Distrugge un interprete e libera le risorse associate.
 */
void tflm_destroy_interpreter(TFLM_Interpreter_t* interpreter);

/**
 * @brief Esegue l'inferenza del modello.
 */
bool tflm_invoke(TFLM_Interpreter_t* interpreter);

/**
 * @brief Ottiene un puntatore al tensore di input.
 */
TfLiteTensor* tflm_get_input_tensor(TFLM_Interpreter_t* interpreter, int index);

/**
 * @brief Ottiene un puntatore al tensore di output.
 */
TfLiteTensor* tflm_get_output_tensor(TFLM_Interpreter_t* interpreter, int index);

/**
 * @brief Restituisce i byte utilizzati nell'arena dei tensori.
 */
uint32_t tflm_get_arena_used_bytes(TFLM_Interpreter_t* interpreter);

#ifdef __cplusplus
}
#endif

#endif // TFLM_WRAPPER_H