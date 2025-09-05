# === ANALISI MEMORIA E STACK DEBUGGING ===

target remote localhost:2331
monitor reset
load

echo \n=== 🔍 MEMORY ANALYSIS DEBUG ===\n

# STEP 1: Analizza gli indirizzi del loop
echo \n📍 ANALIZZANDO INDIRIZZI DEL LOOP:\n
x/10i 0x0000D0D0
echo \n
x/10i 0x00006328
echo \n

# STEP 2: Controlla stack pointer e memoria
echo \n📊 STACK E MEMORIA:\n
info registers sp
x/16x $sp
echo \nStack pointer: 
p/x $sp
echo \nMemoria disponibile: 
x/4x 0x20000000
x/4x 0x20008000

# STEP 3: Controlla se il main() è mai chiamato
echo \n🎯 VERIFICANDO SE MAIN È CHIAMATO:\n
break main
continue

# STEP 4: Se non raggiunge main, controlla startup
echo \n⚠️  SE NON RAGGIUNGE MAIN, CONTROLLA STARTUP:\n
break Reset_Handler
continue

# STEP 5: Analizza memoria heap/arena
echo \n💾 MEMORIA HEAP/ARENA:\n
info symbol 0x20000000
info symbol 0x20008000

echo \n🔚 FINE ANALISI - PREMI CTRL+C PER USCIRE\n

