# Load firmware file
file audiomoth_smart.axf

# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load SMART firmware
monitor reset 1
load
monitor reset 0

# Set print pretty on
set print pretty on

# Enable semihosting for printf output
monitor semihosting enable

# Break right at the start of NN_Init to see what happens
break NN_Init
command 1
    echo \n
    echo "📍 NN_Init started\n"
    echo "Current state:\n"
    p g_nn_state
    echo \n
end

# Break in initialize_interpreters to see bridge creation
break initialize_interpreters
command 2
    echo \n
    echo "📍 initialize_models started\n"
    echo "Arena stats:\n"
    p total_arena_size
    echo \n
end

break performanceBenchmark
command 3
    echo \n
    echo "📍 performanceBenchmark started - this means NN_Init succeeded!\n"
    echo \n
end

# Break on our CMSIS wrapper functions to see if they get called
break tflm_create_interpreter
command 4
    echo \n
    echo "🎯 CMSIS WRAPPER: Creating interpreter...\n"
    echo "Model data size: %d\n", $arg1
    echo "Arena size: %zu\n", $arg3
    echo \n
end

break tflm_invoke
command 5
    echo \n
    echo "🔥 CMSIS WRAPPER: Invoking neural network!\n"
    echo "Interpreter: 0x%x\n", $arg0
    echo \n
end

break run_backbone_inference
command 6
    echo \n
    echo "🧠 BACKBONE INFERENCE: Running CMSIS-NN backbone model!\n"
    echo \n
end

break run_streaming_inference
command 7
    echo \n
    echo "🌊 STREAMING INFERENCE: Running CMSIS-NN streaming model!\n"
    echo \n
end

break cmsis_dense_s8
command 8
    echo \n
    echo "⚡ CMSIS-NN: Dense layer executing!\n"
    echo "Input size: %d, Output size: %d\n", $arg3, $arg4
    echo \n
end

break cmsis_softmax_s8
command 9
    echo \n
    echo "🎯 CMSIS-NN: Softmax activation executing!\n"
    echo "Vector size: %d\n", $arg2
    echo \n
end

# Break at the end to see results
break run_backbone_inference
command 10
    echo \n
    echo "📊 BACKBONE RESULTS:\n"
    echo "Output tensor (first 5 values):\n"
    x/5xb interp->output_data
    echo \n
    finish
end

break run_streaming_inference  
command 11
    echo \n
    echo "📊 STREAMING RESULTS:\n"
    echo "Output tensor (first 5 values):\n"
    x/5xb interp->output_data
    echo \n
    finish
end

# Info message
echo \n🚀 CMSIS-NN FLASH TEST - Real neural network inference on AudioMoth!\n
echo Expected flow: NN_Init → tflm_create_interpreter → tflm_invoke → CMSIS operations\n

# Add a breakpoint at a function that runs AFTER inference
break main
command 99
    echo "\n===============================================\n"
    echo "📊 TIMING VARIABLES ANALYSIS (Runtime)\n" 
    echo "===============================================\n"
    echo "🕐 Timing Variables from main.c:\n"
    echo "--------------------------------\n"
    
    # Direct memory read from .bss section (where global vars are)
    echo "📍 Memory scan for timing variables:\n"
    echo "BSS section starts at 0x20000D1C (536871020 decimal)\n"
    x/20wd 0x20000D1C
    echo "\n"
    
    echo "🔍 Looking for patterns in timing data...\n"
    # total_time_ms, avg_time_us, time_100_inferences_ms, benchmark_completed
    # Should be around lines 21-24 in main.c 
    x/4wd 0x20000D1C+80  
    x/4wd 0x20000D1C+96
    x/4wd 0x20000D1C+112
    
    echo "\n===============================================\n"
    finish
end

# Run the program
continue

echo "\n🎯 Program is running. LED should show pattern:\n"
echo "  - Fast blink (5Hz): Initialization phase\n"
echo "  - Slow blink (1Hz): Running inference\n"  
echo "  - Solid ON: Inference complete\n"
echo "  - Watch LED on device for status\n"
echo "\n"