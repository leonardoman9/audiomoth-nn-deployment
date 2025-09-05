# Script semplice per controllare lo stato NN
target remote localhost:2331

printf "=== 🧠 NN STATUS CHECK ===\n"
printf "Total time (ms): %d\n", *(int*)0x200010f8
printf "Average time (us): %d\n", *(int*)0x200010fc
printf "Benchmark completed: %d\n", *(int*)0x20001104
printf "NN State: %d\n", *(int*)0x20001928
printf "Current PC: 0x%x\n", $pc

# Continua per 10 secondi
printf "\n⏳ Continuing for 10 seconds...\n"
continue &
shell sleep 10
interrupt

printf "\n=== 🔄 AFTER 10 SECONDS ===\n"
printf "Total time (ms): %d\n", *(int*)0x200010f8
printf "Average time (us): %d\n", *(int*)0x200010fc
printf "Benchmark completed: %d\n", *(int*)0x20001104
printf "NN State: %d\n", *(int*)0x20001928
printf "Current PC: 0x%x\n", $pc