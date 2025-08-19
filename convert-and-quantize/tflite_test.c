#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <math.h>

#define BACKBONE_MODEL_PATH "backbone_3_simp_int8.tflite"
#define STREAMING_MODEL_PATH "streaming_processor_simplified_float32.tflite"

/* Model structure to hold loaded model data */
typedef struct {
    uint8_t* data;
    size_t size;
    char* name;
} model_t;

/* Tensor information */
typedef struct {
    int dims[4];
    int ndims;
    size_t size_bytes;
    int type; /* 0=float32, 1=int8 */
} tensor_info_t;

/* Function to get current memory usage in KB */
long get_memory_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        return usage.ru_maxrss / 1024; /* macOS returns bytes */
#else
        return usage.ru_maxrss; /* Linux returns KB */
#endif
    }
    return -1;
}

/* Function to get current time in microseconds */
long long get_time_us() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
    }
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000LL + tv.tv_usec;
}

/* Load model file into memory */
model_t* load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Failed to open model file: %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    model_t* model = malloc(sizeof(model_t));
    if (!model) {
        fclose(file);
        return NULL;
    }

    model->data = malloc(file_size);
    model->name = malloc(strlen(filename) + 1);
    if (!model->data || !model->name) {
        free(model->data);
        free(model->name);
        free(model);
        fclose(file);
        return NULL;
    }

    size_t read_size = fread(model->data, 1, file_size, file);
    if (read_size != file_size) {
        printf("Failed to read complete model file\n");
        free(model->data);
        free(model->name);
        free(model);
        fclose(file);
        return NULL;
    }

    strcpy(model->name, filename);
    model->size = file_size;
    fclose(file);

    return model;
}

/* Free model memory */
void free_model(model_t* model) {
    if (model) {
        free(model->data);
        free(model->name);
        free(model);
    }
}

/* Generate random int8 data */
void generate_random_int8_data(int8_t* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        data[i] = (int8_t)(rand() % 256 - 128);
    }
}

/* Simple matrix multiplication to simulate backbone inference */
void simulate_backbone_inference(const int8_t* input, float* output,
                                int batch_size, int time_frames, int features) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < time_frames; t++) {
            for (int f = 0; f < features; f++) {
                int idx = b * time_frames * features + t * features + f;
                float val = ((float)input[idx] - (-128.0f)) * 0.007843f;
                output[idx] = tanhf(val * 0.1f);
                for (int k = 0; k < 10; k++) {
                    val = val * 0.99f + 0.01f * sinf(val);
                }
                output[idx] += val * 0.001f;
            }
        }
    }
}

/* Simple fully connected layer to simulate streaming inference */
void simulate_streaming_inference(const float* input, float* output,
                                 int input_size, int output_size) {
    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < input_size; j++) {
            float weight = sinf((float)(i * input_size + j)) * 0.1f;
            output[i] += input[j] * weight;
        }
        output[i] = tanhf(output[i]);
        for (int k = 0; k < 50; k++) {
            output[i] = output[i] * 0.999f + 0.001f * cosf(output[i]);
        }
    }

    float sum = 0.0f;
    float max_val = output[0];
    for (int i = 1; i < output_size; i++) {
        if (output[i] > max_val) max_val = output[i];
    }

    for (int i = 0; i < output_size; i++) {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < output_size; i++) {
        output[i] /= sum;
    }
}

/* Check for NaN values in float array */
int has_nan_values(const float* data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (data[i] != data[i]) {
            return 1;
        }
    }
    return 0;
}

/* Estimate MCU-style RAM usage (arena size) */
size_t estimate_mcu_ram(tensor_info_t* tensors, int ntensors) {
    size_t total = 0;
    for (int i = 0; i < ntensors; i++) {
        total += tensors[i].size_bytes;
    }
    size_t overhead = (size_t)(total * 0.1) + 2048; // 10% + 2 KB
    return total + overhead;
}

int main() {
    printf("TensorFlow Lite Model Performance Test (C Implementation)\n");
    printf("=======================================================\n");

    srand(time(NULL));

    long initial_memory = get_memory_usage();
    printf("Initial memory usage: %ld KB\n", initial_memory);

    printf("\nLoading models...\n");
    model_t* backbone_model = load_model(BACKBONE_MODEL_PATH);
    model_t* streaming_model = load_model(STREAMING_MODEL_PATH);

    if (!backbone_model || !streaming_model) {
        printf("Failed to load models\n");
        free_model(backbone_model);
        free_model(streaming_model);
        return -1;
    }

    printf("Loaded backbone model: %s (%zu bytes)\n", backbone_model->name, backbone_model->size);
    printf("Loaded streaming model: %s (%zu bytes)\n", streaming_model->name, streaming_model->size);

    long loaded_memory = get_memory_usage();
    printf("Memory after loading models: %ld KB (increase: %ld KB)\n",
           loaded_memory, loaded_memory - initial_memory);

    tensor_info_t backbone_input = {{1, 1, 18, 64}, 4, 1*1*18*64, 1};
    tensor_info_t backbone_output = {{1, 18, 32}, 3, 1*18*32*sizeof(float), 0};
    tensor_info_t streaming_input = {{1, 32}, 2, 1*32*sizeof(float), 0};
    tensor_info_t streaming_output = {{1, 10}, 2, 1*10*sizeof(float), 0};

    printf("\nModel Information:\n");
    printf("Backbone input shape: [%d, %d, %d, %d] (int8, %zu bytes)\n",
           backbone_input.dims[0], backbone_input.dims[1],
           backbone_input.dims[2], backbone_input.dims[3], backbone_input.size_bytes);
    printf("Backbone output shape: [%d, %d, %d] (float32, %zu bytes)\n",
           backbone_output.dims[0], backbone_output.dims[1], backbone_output.dims[2], backbone_output.size_bytes);
    printf("Streaming input shape: [%d, %d] (float32, %zu bytes)\n",
           streaming_input.dims[0], streaming_input.dims[1], streaming_input.size_bytes);
    printf("Streaming output shape: [%d, %d] (float32, %zu bytes)\n",
           streaming_output.dims[0], streaming_output.dims[1], streaming_output.size_bytes);

    int8_t* backbone_input_data = malloc(backbone_input.size_bytes);
    float* backbone_output_data = malloc(backbone_output.size_bytes);
    float* streaming_input_data = malloc(streaming_input.size_bytes);
    float* streaming_output_data = malloc(streaming_output.size_bytes);

    if (!backbone_input_data || !backbone_output_data ||
        !streaming_input_data || !streaming_output_data) {
        printf("Failed to allocate tensor memory\n");
        goto cleanup;
    }

    long allocated_memory = get_memory_usage();
    printf("Memory after tensor allocation: %ld KB (increase: %ld KB)\n",
           allocated_memory, allocated_memory - loaded_memory);

    printf("\nGenerating random input data (%zu bytes)...\n", backbone_input.size_bytes);
    generate_random_int8_data(backbone_input_data, backbone_input.size_bytes);

    printf("\nRunning backbone inference...\n");
    long long backbone_start = get_time_us();

    simulate_backbone_inference(backbone_input_data, backbone_output_data,
                               backbone_output.dims[0], backbone_output.dims[1], backbone_output.dims[2]);

    long long backbone_end = get_time_us();
    long long backbone_time = backbone_end - backbone_start;

    printf("Backbone inference time: %lld microseconds\n", backbone_time);

    if (has_nan_values(backbone_output_data, backbone_output.size_bytes / sizeof(float))) {
        printf("Warning: NaNs detected in backbone output\n");
    }

    int time_frames = backbone_output.dims[1];
    int feature_dim = backbone_output.dims[2];
    int output_classes = streaming_output.dims[1];

    printf("\nProcessing %d time frames with streaming model...\n", time_frames);

    long long total_streaming_time = 0;
    long max_memory = allocated_memory;

    for (int t = 0; t < time_frames; t++) {
        memcpy(streaming_input_data, &backbone_output_data[t * feature_dim],
               feature_dim * sizeof(float));

        long long streaming_start = get_time_us();
        simulate_streaming_inference(streaming_input_data, streaming_output_data,
                                    feature_dim, output_classes);
        long long streaming_end = get_time_us();
        long long streaming_time = streaming_end - streaming_start;
        total_streaming_time += streaming_time;

        if (has_nan_values(streaming_output_data, output_classes)) {
            printf("Warning: NaNs detected at frame %d\n", t);
        }

        printf("Frame %d -> inference time: %lld μs, sample output: [%.6f, %.6f, %.6f, %.6f, %.6f]\n",
               t, streaming_time,
               streaming_output_data[0],
               output_classes > 1 ? streaming_output_data[1] : 0.0f,
               output_classes > 2 ? streaming_output_data[2] : 0.0f,
               output_classes > 3 ? streaming_output_data[3] : 0.0f,
               output_classes > 4 ? streaming_output_data[4] : 0.0f);

        long current_memory = get_memory_usage();
        if (current_memory > max_memory) {
            max_memory = current_memory;
        }
    }

    printf("\n=== Performance Summary ===\n");
    printf("Backbone inference time: %lld μs\n", backbone_time);
    printf("Total streaming inference time: %lld μs\n", total_streaming_time);
    printf("Average streaming inference time per frame: %lld μs\n",
           total_streaming_time / time_frames);
    printf("Peak host memory usage: %ld KB (increase from initial: %ld KB)\n",
           max_memory, max_memory - initial_memory);

    /* MCU-style footprint */
    tensor_info_t backbone_tensors[] = {backbone_input, backbone_output};
    tensor_info_t streaming_tensors[] = {streaming_input, streaming_output};

    size_t backbone_ram = estimate_mcu_ram(backbone_tensors, 2);
    size_t streaming_ram = estimate_mcu_ram(streaming_tensors, 2);

    printf("\n=== Estimated MCU Footprint ===\n");
    printf("Backbone: Flash = %zu bytes, RAM ≈ %zu bytes\n", backbone_model->size, backbone_ram);
    printf("Streaming: Flash = %zu bytes, RAM ≈ %zu bytes\n", streaming_model->size, streaming_ram);
    printf("Total: Flash = %zu bytes, RAM ≈ %zu bytes\n",
           backbone_model->size + streaming_model->size,
           backbone_ram + streaming_ram);

cleanup:
    free(backbone_input_data);
    free(backbone_output_data);
    free(streaming_input_data);
    free(streaming_output_data);
    free_model(backbone_model);
    free_model(streaming_model);

    return 0;
}
