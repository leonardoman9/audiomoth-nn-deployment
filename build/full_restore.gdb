echo 🔧 FULL CHIP ERASE E RESTORE AUDIOMOTH\n
echo ==========================================\n

# Connect to J-Link GDB Server
target remote localhost:2331

# Reset and halt
monitor reset 1

# Full chip erase (this erases EVERYTHING including bootloader)
monitor flash device = EFM32WG380F256
monitor flash erase

echo \n⚠️  CHIP COMPLETAMENTE CANCELLATO\n

# Load original firmware starting from 0x0000 (includes bootloader)
restore /Users/leonardomannini/Downloads/AudioMoth-Firmware-Basic-1.11.0.bin binary 0x0000

echo \n✅ FIRMWARE + BOOTLOADER RIPRISTINATI\n

# Reset and start
monitor reset 0
monitor go

echo \nOra disconnetti J-Link e testa:\n
echo 1. USB mode - dovrebbe essere visto da AudioMoth Flash App\n
echo 2. DEFAULT mode con batterie - dovrebbe funzionare\n

detach
quit