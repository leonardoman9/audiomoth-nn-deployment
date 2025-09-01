echo 🧠 SMART FIRMWARE FINAL TEST\n
echo ===============================\n

# Connect and flash
target remote localhost:2331
monitor reset 1
load
monitor reset 0

# Key breakpoints for smart test
break performanceBenchmark
break NN_ProcessAudio
break tflm_invoke

echo \n🎯 Breakpoints set:\n
echo - performanceBenchmark (start of test)\n  
echo - NN_ProcessAudio (neural pipeline)\n
echo - tflm_invoke (REAL inference!)\n

echo \n🚀 Starting execution...\n
continue