# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load SMART firmware
monitor reset 1
load
monitor reset 0

# Set print pretty on
set print pretty on

# Test to see if we're using ENHANCED simulation or real TFLM
break tflm_invoke
command 1
    echo \n
    echo "=====================================================================\n"
    echo "📍 STOP: Inside tflm_invoke - ENHANCED SIMULATION\n"
    echo \n
    echo "This should now run ENHANCED math simulation with real computations!\n"
    echo "Call stack (top 3 levels):\n"
    bt 3
    echo \n
    echo "--> Type 'step' to go deeper, or 'continue' to let it run\n"
    echo "=====================================================================\n"
end

# Check tensor allocation with enhanced simulation
break tflm_allocate_tensors  
command 2
    echo \n
    echo "🎯 ENHANCED TENSOR ALLOCATION!\n"
    echo "This is using the new enhanced simulation that does real math.\n"
    echo "Arena details:\n"
    echo \n
    echo "--> Type 'continue' to proceed with enhanced simulation.\n"
end

# Also check the model parsing 
break tflite_parse_model
command 3
    echo \n
    echo "📋 Model parsing checkpoint\n"
    echo "Model data pointer and size:\n"
    p model_data
    p model_size
    echo \n
end

# Check if we enter the enhanced simulation functions
break simulate_conv2d_layer
command 4
    echo \n
    echo "🔥 CONV2D SIMULATION ACTIVE!\n"
    echo "This proves we're using enhanced neural network simulation!\n"
    echo \n
end

break simulate_dense_layer  
command 5
    echo \n
    echo "🔥 DENSE LAYER SIMULATION ACTIVE!\n"
    echo "Real neural network computation happening!\n"
    echo \n
end

break simulate_softmax
command 6
    echo \n
    echo "🔥 SOFTMAX SIMULATION ACTIVE!\n"
    echo "Final layer of neural network simulation!\n"
    echo \n
end

# Info message
echo \n🔍 Starting ENHANCED SIMULATION Test...\n
echo This will verify we're using enhanced neural network simulation.\n

# Run the program
continue