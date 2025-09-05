# =================================================================
# SIMPLE GDB SCRIPT PER TEST HARDWARE
# =================================================================

# Load firmware file (no need to specify, already loaded by gdb command line)

# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load firmware
monitor reset 1
load
monitor reset 0

# Set print pretty on
set print pretty on

# Enable semihosting for printf output
monitor semihosting enable

echo \n
echo "🚀 ============================================\n"
echo "   AUDIOMOTH NEURAL NETWORK TEST (SIMPLE)\n" 
echo "============================================\n"
echo \n

# Break at main 
break main
command 1
    echo "📍 MAIN STARTED - Firmware is running!\n"
    echo "Status:\n"
    info registers pc sp lr
    echo "\n🚀 Continuing execution for benchmark...\n"
    continue
end

# Let it run and then manually interrupt after ~10 seconds
echo "\n⏱️ AFTER STARTING, WAIT ~10 SECONDS THEN:\n"
echo "1. Press CTRL+C to interrupt\n"  
echo "2. Run: x/32w 0x20000000\n"
echo "3. Look for timing values!\n"

# Alternative: break at a memory address if line numbers don't work
# You can find addresses with: arm-none-eabi-objdump -d audiomoth_smart.axf | grep main

echo "🚀 Starting simple hardware test...\n"
echo "Press 'c' to continue\n"
echo "\n"

run