# CMSIS-NN Detailed Debug Script
# Connect to J-Link GDB Server  
target remote localhost:2331

# Reset and load SMART firmware
monitor reset 1
load
monitor reset 0

# Set print pretty on
set print pretty on

echo \n🚀 CMSIS-NN DETAILED DEBUG SESSION\n
echo Looking for function execution flow and variable values...\n\n

# FASE 1: BREAKPOINT SULLE FUNZIONI PRINCIPALI (se existono simboli)
echo Setting up function breakpoints (if symbols exist)...\n

# Try to break on main functions
break main
command 1
    echo \n📍 MAIN STARTED\n
    echo "Continuing execution to trace NN flow...\n"
    continue
end

# FASE 2: BREAKPOINT SPECIFICI SULLE FUNZIONI CMSIS
# Indirizzi trovati dal file .lst

# Break on tflm_create_interpreter (address: 0x00005aac)
break *0x5aac
command 4
    echo \n🧠 CMSIS INTERPRETER CREATION!\n
    echo "Address: 0x5aac (tflm_create_interpreter)\n"
    echo "Registers: r0=0x%x r1=0x%x r2=0x%x r3=0x%x\n", $r0, $r1, $r2, $r3
    continue
end

# Break on tflm_invoke (address: 0x000065fc)  
break *0x65fc
command 5
    echo \n🚀 CMSIS INVOKE CALLED!\n
    echo "Address: 0x65fc (tflm_invoke)\n"
    echo "This means neural network inference is starting!\n"
    continue
end

# FASE 3: BREAKPOINT AGGIUNTIVI PER FLOW ANALYSIS

# FASE 4: STRATEGIC MEMORY MONITORING
# Monitor tensor arena (external SRAM)
watch *(int*)0x80000000
command 2
    echo \n🧠 TENSOR ARENA WRITE DETECTED!\n
    echo "Memory: 0x80000000 = 0x%x\n", *(int*)0x80000000
    echo "This suggests tensor allocation/usage!\n"
    bt 3
    echo \n
    continue
end

# FASE 5: MONITOR LED CONTROL (GPIO)
# AudioMoth LED control - monitor GPIO writes
# Green LED e Red LED changes indicate progress

# FASE 6: MONITOR FUNCTION CALLS VIA STACK ANALYSIS  
# Break periodically and analyze stack

# Remove the frequent checkpoint to avoid spam
# Focus on specific CMSIS function calls

echo \nStarting SMART debugging...\n
echo Strategy: Monitor key memory regions and execution checkpoints\n
echo "- Tensor Arena: 0x80000000 (external SRAM)\n"
echo "- Execution flow via periodic breaks\n" 
echo "- LED behavior observation\n\n

# Start execution with intelligent monitoring
continue

# If we get here, the program finished
echo \n📊 PROGRAM EXECUTION COMPLETED\n
echo Checking final memory state...\n

# Check tensor arena final state
echo \nTensor Arena final state:\n
x/16xb 0x80000000

echo \nSession complete.\n