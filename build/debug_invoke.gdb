target remote localhost:2331
load
set print pretty on
set confirm off

# Set a breakpoint inside the new wrapper's invoke function
break 'third_party/tflm_smart_wrapper.c':125

run