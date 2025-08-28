echo 🎯 AUDIOMOTH BASIC FIRMWARE DEBUG\n
echo ====================================\n

# Connect to J-Link GDB Server
target remote localhost:2331

# Reset and load firmware
monitor reset 1
load
monitor reset 0

# Set breakpoints for basic functionality
break main
break AudioMoth_initialise

# Info message
echo \n📍 Breakpoints set:\n
echo - main\n
echo - AudioMoth_initialise\n

echo \n🚀 Starting execution...\n
continue