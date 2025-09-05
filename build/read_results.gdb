# Script ottimizzato per debug NN benchmark
target remote localhost:2331

printf "=== 🧠 NEURAL NETWORK BENCHMARK DEBUG ===\n"

# Variabili chiave
printf "📊 Initial NN State:\n"
printf "  Total time (ms): %d\n", *(int*)0x200010f8
printf "  Average time (us): %d\n", *(int*)0x200010fc
printf "  Benchmark completed: %d\n", *(int*)0x20001104
printf "  NN State: %d\n", *(int*)0x20001928

# Posizione corrente
printf "\n📍 Current Position: 0x%x\n", $pc
x/5i $pc

# ASPETTA CHE IL BENCHMARK COMPLETI (60 secondi)
printf "\n⏳ Waiting for NN benchmark (busy loops + 5 inferences)...\n"
printf "   Expected time: ~30-60 seconds due to busy wait loops\n"

set $wait_count = 0
while $wait_count < 12
    printf "   Checking iteration %d/12...\n", $wait_count + 1
    continue &
    shell sleep 5
    interrupt
    
    # Controlla se è completato
    set $completed = *(int*)0x20001104
    set $total_time = *(int*)0x200010f8
    printf "     Status: completed=%d, total_time=%d ms\n", $completed, $total_time
    
    if $completed > 0 || $total_time > 0
        printf "🎉 BENCHMARK COMPLETED!\n"
        loop_break
    end
    
    set $wait_count = $wait_count + 1
end

printf "\n=== 🏁 FINAL RESULTS ===\n"
printf "Total time (ms): %d\n", *(int*)0x200010f8
printf "Average time (us): %d\n", *(int*)0x200010fc
printf "Benchmark completed: %d\n", *(int*)0x20001104
printf "NN State: %d\n", *(int*)0x20001928
printf "Final PC: 0x%x\n", $pc

# Continua esecuzione per vedere LED finali
printf "\n⏳ Continuing for final LED sequence...\n"
continue