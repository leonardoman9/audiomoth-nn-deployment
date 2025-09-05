#ifndef TFLITE_PARSER_MINIMAL_H
#define TFLITE_PARSER_MINIMAL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ================================================================
// TFLITE PARSER MINIMALE PER AUDIOMOTH
// Parse solo quello che serve dai file .tflite
// ================================================================

// TFLite magic number e versioni
#define TFLITE_MAGIC 0x334C4654  // "TFL3" in little-endian
#define TFLITE_SCHEMA_VERSION 3

// Tipi tensor supportati (subset)
typedef enum {
    TFLITE_TYPE_FLOAT32 = 1,
    TFLITE_TYPE_INT8 = 9,
    TFLITE_TYPE_INT32 = 5
} TFLiteDataType;

// Opcodes operatori (subset che usiamo)
typedef enum {
    TFLITE_OP_ADD = 0,
    TFLITE_OP_CONV_2D = 3,
    TFLITE_OP_FULLY_CONNECTED = 9,
    TFLITE_OP_LOGISTIC = 14,        // Sigmoid
    TFLITE_OP_RELU = 17,
    TFLITE_OP_RESHAPE = 22,
    TFLITE_OP_SOFTMAX = 25,
    TFLITE_OP_TANH = 28,
    TFLITE_OP_MUL = 18,
    TFLITE_OP_MEAN = 40,
    TFLITE_OP_RELU6 = 18,
    TFLITE_OP_QUANTIZE = 74,
    TFLITE_OP_DEQUANTIZE = 6
} TFLiteBuiltinOperator;

// ================================================================
// STRUTTURE DATI
// ================================================================

// Header del file TFLite
typedef struct {
    uint32_t magic;           // 0x54464C33
    uint32_t version;         // Schema version  
    uint32_t num_subgraphs;   // Numero di subgraph (di solito 1)
} TFLiteHeader;

// Quantization info
typedef struct {
    float scale;              // Quantization scale
    int32_t zero_point;       // Quantization zero point
} TFLiteQuantization;

// Tensor info
typedef struct {
    uint32_t* shape;          // [batch, height, width, channels]
    uint32_t shape_size;      // Numero dimensioni
    TFLiteDataType type;      // Tipo dati (INT8, FLOAT32, etc.)
    void* data;               // Pointer ai dati (weights o buffer)
    TFLiteQuantization quant; // Parametri quantizzazione
    bool is_variable;         // Se è una variabile o costante
    uint32_t buffer_id;       // ID del buffer contenente i dati
} TFLiteTensor;

// Operator info
typedef struct {
    uint32_t opcode_index;    // Index nell'opcode table
    uint32_t* inputs;         // Array di tensor input IDs
    uint32_t num_inputs;      // Numero input
    uint32_t* outputs;        // Array di tensor output IDs  
    uint32_t num_outputs;     // Numero output
    void* builtin_options;    // Parametri specifici operazione (opzionale)
    uint32_t builtin_options_size; // Dimensione parametri
} TFLiteOperator;

// Buffer che contiene i dati (weights, bias, etc.)
typedef struct {
    uint8_t* data;            // Dati raw
    uint32_t size;            // Dimensione buffer
} TFLiteBuffer;

// Subgraph (di solito ne abbiamo solo uno)
typedef struct {
    TFLiteTensor* tensors;    // Array di tensori
    uint32_t num_tensors;     // Numero tensori
    TFLiteOperator* operators; // Array di operatori
    uint32_t num_operators;   // Numero operatori
    uint32_t* inputs;         // Input tensor IDs del modello
    uint32_t num_inputs;      // Numero input del modello
    uint32_t* outputs;        // Output tensor IDs del modello
    uint32_t num_outputs;     // Numero output del modello
    char* name;               // Nome subgraph (opzionale)
} TFLiteSubgraph;

// Modello completo
typedef struct {
    TFLiteHeader header;      // Header informazioni
    TFLiteSubgraph* subgraphs; // Array subgraphs (di solito 1)
    uint32_t num_subgraphs;   // Numero subgraphs
    TFLiteBuffer* buffers;    // Array buffer dati
    uint32_t num_buffers;     // Numero buffer
    uint32_t* opcodes;        // Array opcodes utilizzati
    uint32_t num_opcodes;     // Numero opcodes
} TFLiteModel;

// ================================================================
// PARSING FUNCTIONS
// ================================================================

/**
 * @brief Parse completo del modello TFLite
 * @param model_data Dati raw del file .tflite
 * @param model_size Dimensione dati
 * @return Pointer al modello parsato o NULL se errore
 */
TFLiteModel* parse_tflite_model(const uint8_t* model_data, size_t model_size);

/**
 * @brief Libera memoria del modello parsato
 * @param model Modello da liberare
 */
void free_tflite_model(TFLiteModel* model);

/**
 * @brief Ottieni pointer ai dati di un tensor
 * @param model Modello parsato
 * @param subgraph_id ID subgraph (di solito 0)
 * @param tensor_id ID tensor
 * @return Pointer ai dati o NULL se errore
 */
void* get_tensor_data(TFLiteModel* model, uint32_t subgraph_id, uint32_t tensor_id);

/**
 * @brief Ottieni informazioni quantizzazione tensor
 * @param model Modello parsato
 * @param subgraph_id ID subgraph
 * @param tensor_id ID tensor
 * @param quant Output quantization info
 * @return true se successo
 */
bool get_tensor_quantization(TFLiteModel* model, uint32_t subgraph_id, 
                            uint32_t tensor_id, TFLiteQuantization* quant);

/**
 * @brief Ottieni shape di un tensor
 * @param model Modello parsato
 * @param subgraph_id ID subgraph
 * @param tensor_id ID tensor
 * @param shape Output array dimensioni
 * @param max_dims Dimensione massima array shape
 * @return Numero dimensioni o 0 se errore
 */
uint32_t get_tensor_shape(TFLiteModel* model, uint32_t subgraph_id, 
                         uint32_t tensor_id, uint32_t* shape, uint32_t max_dims);

/**
 * @brief Ottieni opcode di un operatore
 * @param model Modello parsato
 * @param subgraph_id ID subgraph
 * @param operator_id ID operatore
 * @return Opcode o 0xFFFFFFFF se errore
 */
TFLiteBuiltinOperator get_operator_opcode(TFLiteModel* model, uint32_t subgraph_id, 
                                          uint32_t operator_id);

/**
 * @brief Debug: stampa informazioni modello
 * @param model Modello da stampare
 */
void print_model_info(TFLiteModel* model);

/**
 * @brief Parse singolo subgraph (for debugging)
 * @param data Model data
 * @param subgraph_offset Offset del subgraph
 * @param subgraph Output subgraph
 * @return true se successo
 */
bool parse_subgraph(const uint8_t* data, uint32_t subgraph_offset, TFLiteSubgraph* subgraph);

// Custom parser for non-standard streaming model format
bool parse_streaming_model_custom(const uint8_t* data, uint32_t len, TFLiteModel* model);

#ifdef __cplusplus
}
#endif

#endif // TFLITE_PARSER_MINIMAL_H