# === TEST CON INDIRIZZI DIRETTI DELLE VARIABILI ===

target remote localhost:2331
monitor reset
load

echo \n=== 🔍 AUDIOMOTH NN DIRECT MEMORY TEST ===\n

# STEP 1: Vai al main e lascia correre
break main
continue
echo \n✅ MAIN REACHED! Now running...\n

# STEP 2: Continua per permettere il benchmark
continue &
echo \n⏱️  WAITING 15 SECONDS FOR BENCHMARK...\n

# Attendi 15 secondi poi interrompi
# (In GDB non possiamo fare sleep, quindi aspetto manualmente)

echo \n📍 PRESS CTRL+C AFTER ~15 SECONDS TO CHECK RESULTS\n
echo \n💡 THEN RUN THESE COMMANDS:\n
echo \ninterrupt\n
echo \nx/w 0x20000000  # Check start of SRAM\n
echo \ninfo registers pc sp lr\n
echo \nbt\n
echo \necho === SEARCH FOR VOLATILE VARIABLES ===\n
echo \nfind 0x20000000, 0x20008000, 0xDEADBEEF  # Look for markers\n
echo \necho === MANUAL MEMORY SCAN ===\n
echo \nx/32w 0x20000000  # First 32 words of SRAM\n
echo \nx/32w 0x20007F00  # Near stack area\n

echo \n🎯 MANUAL COMMANDS TO RUN:\n
echo \n1. interrupt\n
echo \n2. info registers\n
echo \n3. x/32w 0x20000000\n
echo \n4. Look for timing values!\n

