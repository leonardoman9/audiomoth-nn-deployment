target remote localhost:2331
load
set print pretty on
set confirm off

# Break after the model is parsed inside the wrapper
# This breakpoint will be hit twice: once for the backbone, once for streaming.
break 'third_party/tflm_smart_wrapper.c':126

# Break right after the model is parsed to inspect it
break tflite_parse_model
command 1
    echo \n
    echo "--- PARSING MODEL --- \n"
    echo "Model data at: "
    p/x model_data
    echo "Model size: "
    p model_size
    echo "\n"
    finish
    echo "\n--- PARSING COMPLETE --- \n"
    echo "Parsed model contents:\n"
    p *model
    echo "\n"
end

# Break at the start of the benchmark to see the final state
break performanceBenchmark
command 2
    echo "\n--- REACHED BENCHMARK ---\n"
    echo "Final NN State: "
    p g_nn_state
    echo "Backbone Interpreter State:\n"
    p *(SmartInterpreterImpl*)g_backbone_interpreter
    echo "Streaming Interpreter State:\n"
    p *(SmartInterpreterImpl*)g_streaming_interpreter
    echo "\n"
end


# Info message
echo \n🔍 Verifying Real TFLite Parser...\n

# Run the program
continue