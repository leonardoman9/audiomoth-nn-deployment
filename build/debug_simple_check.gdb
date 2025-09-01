# Simple check of benchmark variables
target remote localhost:2331

# Reset and load
monitor reset 1
load
monitor reset 0

echo \n🔍 SIMPLE BENCHMARK CHECK\n

# Let program run for a while, then interrupt and check variables
echo Starting program for 10 seconds, then checking variables...\n
continue &
shell sleep 10
interrupt

echo \n📊 CHECKING BENCHMARK VARIABLES:\n
printf "total_time_ms = %u\n", {uint32_t}0x20001f34
printf "avg_time_us = %u\n", {uint32_t}0x20001f38
printf "time_100_inferences_ms = %u\n", {uint32_t}0x20001f3c
printf "benchmark_completed = %u\n", {uint32_t}0x20001f40

echo \nDone.\n
quit