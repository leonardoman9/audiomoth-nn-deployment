# === TEST FIRMWARE CON LOOP FINITI ===

target remote localhost:2331
monitor reset
load

echo \n=== 🚀 AUDIOMOTH NN TEST (FIXED LOOPS) ===\n

# STEP 1: Vai direttamente al main
echo \n📍 SETTING BREAKPOINT AT MAIN:\n
break main
continue

echo \n✅ MAIN RAGGIUNTO! Status:\n
info registers pc sp lr
echo \nNN variables status:\n
print nnInitialized
print total_time_ms
print avg_time_us
print benchmark_completed

# STEP 2: Lascia correre per ~10 secondi per il benchmark + 5 sec di loop
echo \n⏱️  RUNNING BENCHMARK (15 seconds timeout):\n
continue

# STEP 3: Dopo 15 secondi, interrompi e leggi i risultati
echo \n📊 FINAL RESULTS:\n
interrupt
echo \nTiming results:\n
print total_time_ms
print avg_time_us
print benchmark_completed
echo \nNN status:\n
print nnInitialized

echo \n🎯 LOCATION:\n
info registers pc
disassemble $pc,$pc+20

echo \n✅ TEST COMPLETATO!\n

