# Debug script to verify CMSIS-NN benchmark results
target remote localhost:2331

# Reset and load
monitor reset 1
load  
monitor reset 0

echo \n🚀 CMSIS-NN BENCHMARK VERIFICATION\n
echo Monitoring global variables to prove execution...\n\n

# Variable addresses from .lst file
set $total_time_addr = 0x20001f34
set $avg_time_addr = 0x20001f38  
set $time_100_addr = 0x20001f3c
set $benchmark_flag_addr = 0x20001f40

echo Variable addresses:\n
echo "  total_time_ms:        0x%08x\n", $total_time_addr
echo "  avg_time_us:          0x%08x\n", $avg_time_addr
echo "  time_100_inferences:  0x%08x\n", $time_100_addr
echo "  benchmark_completed:  0x%08x\n", $benchmark_flag_addr
echo \n

# Initial values check
echo ⏰ INITIAL VALUES (should be zero):\n
printf "  total_time_ms = %u\n", {uint32_t}$total_time_addr
printf "  avg_time_us = %u\n", {uint32_t}$avg_time_addr
printf "  time_100_inferences_ms = %u\n", {uint32_t}$time_100_addr
printf "  benchmark_completed = %u\n", {uint32_t}$benchmark_flag_addr
echo \n

# Break when benchmark_completed becomes 1
watch *((uint32_t*)$benchmark_flag_addr)
command 1
    echo \n🎉 BENCHMARK COMPLETED FLAG SET!\n
    printf "  benchmark_completed = %u\n", {uint32_t}$benchmark_flag_addr
    echo \n📊 FINAL BENCHMARK RESULTS:\n
    printf "  total_time_ms = %u ms\n", {uint32_t}$total_time_addr
    printf "  avg_time_us = %u μs\n", {uint32_t}$avg_time_addr
    printf "  time_100_inferences_ms = %u ms\n", {uint32_t}$time_100_addr
    echo \n
    if {uint32_t}$total_time_addr > 0
        echo ✅ SUCCESS: Non-zero timing values prove CMSIS-NN execution!
    else
        echo ❌ FAILURE: Zero values suggest no execution
    end
    echo \n
    continue
end

echo Starting execution and waiting for benchmark completion...\n
echo (This may take a few seconds for 10K inferences)\n\n

# Start execution
continue

# If we reach here, check final state manually
echo \n📊 MANUAL CHECK - Final variable values:\n
printf "  total_time_ms = %u ms\n", {uint32_t}$total_time_addr
printf "  avg_time_us = %u μs\n", {uint32_t}$avg_time_addr
printf "  time_100_inferences_ms = %u ms\n", {uint32_t}$time_100_addr
printf "  benchmark_completed = %u\n", {uint32_t}$benchmark_flag_addr

echo \nSession complete.\n