# Simple manual variable check
target remote localhost:2331

# Reset and load
monitor reset 1
load
monitor reset 0

# Start program and let it run for a few seconds
echo \n🔍 Starting program and checking variables after execution...\n
continue &

# Wait manually then interrupt
shell sleep 5
interrupt

echo \n📊 CHECKING BENCHMARK VARIABLES (manual):\n
# Check variables using direct memory addresses
set $addr1 = 0x20001f34
set $addr2 = 0x20001f38
set $addr3 = 0x20001f3c
set $addr4 = 0x20001f40

printf "Memory check at benchmark variable locations:\n"
printf "0x20001f34 (total_time_ms) = %u\n", *(uint32_t*)$addr1
printf "0x20001f38 (avg_time_us) = %u\n", *(uint32_t*)$addr2
printf "0x20001f3c (time_100_inferences_ms) = %u\n", *(uint32_t*)$addr3
printf "0x20001f40 (benchmark_completed) = %u\n", *(uint32_t*)$addr4

echo \n
if *(uint32_t*)$addr1 > 0
    echo ✅ SUCCESS: Non-zero values detected - CMSIS-NN executed!
else
    echo ❌ All values are zero - need more investigation
end

quit