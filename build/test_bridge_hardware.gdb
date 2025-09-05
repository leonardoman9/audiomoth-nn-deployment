# =================================================================
# GDB SCRIPT PER TEST HARDWARE TFLITE-CMSIS BRIDGE
# =================================================================

# Load firmware file
file audiomoth_smart.axf

# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load firmware
monitor reset 1
load
monitor reset 0

# Set print pretty on
set print pretty on

# Enable semihosting for printf output
monitor semihosting enable

echo \n
echo "🚀 ============================================\n"
echo "   AUDIOMOTH NEURAL NETWORK BRIDGE TEST\n" 
echo "============================================\n"
echo \n

# Break at main to start
break main
command 1
    echo "📍 MAIN STARTED\n"
    echo "Stepping to NN system...\n"
    continue
end

# Break at NN_Init
break NN_Init  
command 2
    echo "\n📍 NN_INIT STARTED\n"
    echo "Current NN state: "
    p g_nn_state
    echo "Tensor arena size: "
    p TENSOR_ARENA_SIZE
    continue
end

# Break at initialize_interpreters (now initialize_interpreters)
break initialize_interpreters
command 3
    echo "\n📍 BRIDGE CREATION STARTED\n"
    echo "Creating TFLite-CMSIS bridge...\n"
    continue
end

# Break at backbone bridge creation
break tflite_cmsis_create
command 4
    echo "\n📍 TFLITE-CMSIS BRIDGE CREATION\n"
    echo "Model name: "
    p model_name
    echo "Model size: "
    p model_size
    echo "Arena size: "
    p arena_size
    continue
end

# Break at backbone inference
break run_backbone_inference  
command 5
    echo "\n📍 BACKBONE INFERENCE STARTED\n"
    echo "Input tensor pointer: "
    p input_buffer
    echo "Starting CMSIS-NN inference...\n"
    continue
end

# Break at tflite_cmsis_invoke
break tflite_cmsis_invoke
command 6
    echo "\n📍 CMSIS-NN INFERENCE EXECUTION\n"
    echo "Bridge model: "
    p bridge->model_name
    echo "Input shape: "
    p bridge->input_shape[0]
    p bridge->input_shape[1] 
    p bridge->input_shape[2]
    p bridge->input_shape[3]
    continue
end

# Break at end of main loop to check results
break main.c:140
command 7
    echo "\n📍 INFERENCE COMPLETED\n"
    echo "Performance results:\n"
    p total_time_ms
    p avg_time_us
    p time_100_inferences_ms
    echo "Benchmark completed: "
    p benchmark_completed
    continue
end

# Start execution
echo "🚀 Starting AudioMoth Neural Network Bridge Test...\n"
echo "Press 'c' to continue through breakpoints\n"
echo "Press 'q' to quit\n"
echo "\n"

run