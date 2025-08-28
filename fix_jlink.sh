#!/bin/bash
echo "🔧 FIXING J-LINK CONNECTION..."

# Kill any existing processes
sudo pkill -f JLinkGDBServer
sudo pkill -9 -f arm-none-eabi-gdb

# Free port 2331
sudo lsof -ti:2331 | xargs sudo kill -9 2>/dev/null

echo "✅ Processes killed, port freed"
echo "🔌 Reconnect USB cable and run:"
echo "   JLinkGDBServer -select usb -if swd -device EFM32WG380F256 -speed 10000 -port 2331"