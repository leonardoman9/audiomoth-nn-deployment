# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load SMART firmware
monitor reset 1
load
monitor reset 0

# Set print pretty on
set print pretty on

# Break right at the start of NN_Init to see what happens
break NN_Init
command 1
    echo \n
    echo "📍 NN_Init started\n"
    echo "Current state:\n"
    p g_nn_state
    echo \n
end

# Break in initialize_models to see progress
break initialize_models
command 2
    echo \n
    echo "📍 initialize_models started\n"
    echo "Arena stats:\n"
    p total_arena_size
    echo \n
end

# Break in the main performance benchmark to see if it ever gets there
break performanceBenchmark
command 3
    echo \n
    echo "📍 performanceBenchmark started - this means NN_Init succeeded!\n"
    echo \n
end

# Break on our enhanced simulation functions to see if they get called
break tflm_allocate_tensors
command 4
    echo \n
    echo "🎯 ENHANCED TENSOR ALLOCATION called!\n"
    echo \n
end

break tflm_invoke
command 5
    echo \n
    echo "🔥 ENHANCED SIMULATION INVOKE called!\n"
    echo \n
end

# Info message
echo \n🔍 Quick Debug Test - NO AudioMoth_delay breakpoints this time...\n

# Run the program
continue