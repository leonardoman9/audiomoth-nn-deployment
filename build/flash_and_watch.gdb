# Flash firmware and observe LED pattern
target remote localhost:2331

# Reset and load
monitor reset 1
load
monitor reset 0

echo \n🚀 FIRMWARE FLASHED SUCCESSFULLY!\n
echo \n💡 WATCH THE LEDS FOR CMSIS-NN SUCCESS PATTERN:\n
echo "Expected sequence:\n"
echo "1. Normal AudioMoth startup (red LED)\n"
echo "2. CMSIS-NN benchmark execution (may take a few seconds)\n"
echo "3. SUCCESS PATTERN:\n"
echo "   - 10 GREEN flashes\n"
echo "   - Short pause\n"
echo "   - 10 RED flashes\n"
echo "4. Normal AudioMoth operation (delay loop)\n\n"
echo "If you see the GREEN + RED pattern, CMSIS-NN worked perfectly!\n\n"

# Start execution
continue

# Keep running (user can interrupt with Ctrl+C when done observing)