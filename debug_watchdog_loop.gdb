# === DEBUGGING WATCHDOG LOOP INFINITO ===

file build/audiomoth_smart.axf
target remote localhost:2331
monitor reset
load

echo \n=== 🚨 WATCHDOG LOOP DEBUGGING ===\n

# STEP 1: Interrompi il loop e analizza
echo \n⏹️  INTERRUZIONE FORZATA:\n
interrupt
info registers pc lr sp
echo \nPC corrente: 
p/x $pc
echo \nLR (return address): 
p/x $lr

# STEP 2: Disassembla la zona del loop
echo \n🔍 DISASSEMBLY DEL LOOP:\n
x/20i 0x0000D0CC
echo \n
x/20i 0x00006328

# STEP 3: Cerca simboli vicini per capire la funzione
echo \n🎯 SIMBOLI VICINI:\n
info symbol 0x0000D0D0
info symbol 0x00006328
info symbol 0x0000D0CC

# STEP 4: Controlla se è un problema di watchdog timer
echo \n⏰ ANALISI WATCHDOG:\n
# Indirizzo registro watchdog EFM32 (ipotetico)
x/4x 0x40088000
x/4x 0x400C8000

# STEP 5: Forza un breakpoint su AudioMoth_feedWatchdog
echo \n🐕 BREAKPOINT SU WATCHDOG:\n
break AudioMoth_feedWatchdog
continue

# STEP 6: Se non funziona, cerca pattern assembly
echo \n🔧 PATTERN ASSEMBLY:\n
# E7FC = branch unconditional (infinite loop)
find 0x4000, 0x30000, 0xE7FC
find 0x4000, 0x30000, 0x4801

echo \n🔚 ANALISI COMPLETA\n
