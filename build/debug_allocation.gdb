# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load SMART firmware
monitor reset 1
load
monitor reset 0

# Set print pretty on
set print pretty on

# Break in tflm_allocate_tensors to see what happens inside
break tflm_allocate_tensors
command 1
    echo \n
    echo "🎯 ENTERING tflm_allocate_tensors\n"
    echo "Interpreter pointer:\n"
    p interpreter
    echo "SmartInterpreterImpl details:\n"
    p *(SmartInterpreterImpl*)interpreter
    echo \n
end

# Break where it might fail - check if models are parsed correctly
break tflite_parse_model
command 2
    echo \n
    echo "📋 tflite_parse_model called\n"
    echo "Model data and size:\n"
    p model_data
    p model_size
    echo "Parse result will determine success...\n"
    echo \n
end

# Break on the printf statements in allocate_tensors to see what happens
break ../third_party/tflm_smart_wrapper.c:169
command 3
    echo \n
    echo "🔧 Enhanced simulation - calculating memory\n"
    echo "Input size and output size:\n"
    p impl->input_size
    p impl->output_size
    echo \n
end

break ../third_party/tflm_smart_wrapper.c:177
command 4
    echo \n
    echo "🚨 MEMORY ALLOCATION ERROR!\n"
    echo "Required vs Available memory:\n"
    p total_required
    p impl->arena_size
    echo "This is why allocation fails!\n"
    echo \n
end

break ../third_party/tflm_smart_wrapper.c:190
command 5
    echo \n
    echo "✅ MEMORY ALLOCATION SUCCESS!\n"
    echo "Arena used:\n"
    p impl->arena_used
    echo \n
end

# Info message
echo \n🔍 Detailed Allocation Debug - let's see where it fails...\n

# Run the program
continue