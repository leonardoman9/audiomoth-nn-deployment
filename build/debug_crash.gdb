# Connect to J-Link GDB Server
target remote localhost:2331

# Halt the MCU immediately after reset, BEFORE running any code
monitor reset 1
load

# Set a breakpoint at the very first instruction of our code
break Reset_Handler

echo "\n--- GDB Crash Detection ---"
echo "Will stop at Reset_Handler, the earliest point possible."
echo "From there, we can step forward to find the crash."

# Run the program
continue