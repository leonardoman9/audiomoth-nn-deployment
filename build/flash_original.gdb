echo 🔧 FLASH FIRMWARE ORIGINALE AUDIOMOTH\n
echo ==========================================\n

# Connect to J-Link GDB Server
target remote localhost:2331

# Reset and erase
monitor reset 1
monitor flash device = EFM32WG380F256
monitor flash breakpoints = 1
monitor flash download = 1

# Load original firmware (change path if needed)
restore /Users/leonardomannini/Downloads/AudioMoth-Firmware-Basic-1.11.0.bin binary 0x4000

# Reset and start
monitor reset 0
monitor go

echo \n✅ Firmware originale flashato!\n
echo Ora disconnetti J-Link e testa con AudioMoth Flash App\n

detach
quit