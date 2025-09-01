# Direct memory check without symbols
target remote localhost:2331

monitor reset 1
load
monitor reset 0

echo \n🔍 DIRECT MEMORY CHECK\n
echo Letting program run for 10 seconds, then checking memory...\n

continue &
shell sleep 10
interrupt

echo \n📊 CHECKING BENCHMARK VARIABLES (memory only):\n
# Skip symbol-based commands, go straight to memory
echo "total_time_ms at 0x20001f34:"
x/1wd 0x20001f34
echo "avg_time_us at 0x20001f38:"
x/1wd 0x20001f38  
echo "time_100_inferences_ms at 0x20001f3c:"
x/1wd 0x20001f3c
echo "benchmark_completed at 0x20001f40:"
x/1wd 0x20001f40

quit