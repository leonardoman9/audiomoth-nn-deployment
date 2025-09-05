echo 🧠 AUDIOMOTH SMART FIRMWARE DEBUG\n
echo ======================================\n
echo This firmware uses REAL neural networks!\n
echo ======================================\n

# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load SMART firmware
monitor reset 1
load
monitor reset 0

# Set breakpoints for neural network testing
break main
break AudioMoth_initialise
break NN_Init
break NN_ProcessAudio

# IMPORTANT: Smart wrapper breakpoints to see REAL inference
break tflm_invoke
break execute_backbone_real
break execute_streaming_real

# Info message
echo \n📍 Breakpoints set for SMART firmware:\n
echo - main\n
echo - AudioMoth_initialise\n
echo - NN_Init (neural network initialization)\n
echo - NN_ProcessAudio (real inference pipeline)\n
echo - tflm_invoke (REAL neural network execution)\n
echo - execute_backbone_real (real CNN)\n
echo - execute_streaming_real (real LSTM)\n

echo \n🚀 Starting SMART firmware execution...\n
echo \n⚠️  NOTE: This firmware performs REAL neural network inference!\n
echo The outputs will be different from dummy firmware.\n

continue