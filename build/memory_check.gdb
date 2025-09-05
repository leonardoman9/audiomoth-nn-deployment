# Script per controllare direttamente la memoria
target remote localhost:2331

printf "=== 🧠 NEURAL NETWORK STATUS (MEMORY) ===\n"
printf "Total time (ms): %d\n", *(int*)0x200010f8
printf "Average time (us): %d\n", *(int*)0x200010fc
printf "Benchmark completed: %d\n", *(int*)0x20001104
printf "NN State: %d\n", *(int*)0x20001928
printf "Current PC: 0x%x\n", $pc

# Aspetta 5 secondi
printf "\n⏳ Waiting 5 seconds for benchmark to complete...\n"
continue &
shell sleep 5
interrupt

printf "\n=== 🔄 AFTER 5 SECONDS ===\n"
printf "Total time (ms): %d\n", *(int*)0x200010f8
printf "Average time (us): %d\n", *(int*)0x200010fc
printf "Benchmark completed: %d\n", *(int*)0x20001104
printf "NN State: %d\n", *(int*)0x20001928
printf "Current PC: 0x%x\n", $pc

# Se ancora non completato, aspetta altri 10 secondi
set $total = *(int*)0x200010f8
if $total == 0
    printf "\n⏳ Still running, waiting 10 more seconds...\n"
    continue &
    shell sleep 10
    interrupt
    
    printf "\n=== 🏁 FINAL CHECK ===\n"
    printf "Total time (ms): %d\n", *(int*)0x200010f8
    printf "Average time (us): %d\n", *(int*)0x200010fc
    printf "Benchmark completed: %d\n", *(int*)0x20001104
    printf "NN State: %d\n", *(int*)0x20001928
    printf "Current PC: 0x%x\n", $pc
end

quit