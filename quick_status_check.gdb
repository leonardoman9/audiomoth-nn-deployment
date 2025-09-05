# ===============================================
# ⚡ QUICK STATUS CHECK - Controllo rapido stato
# ===============================================

target remote localhost:2331
monitor halt

printf "\n=== ⚡ QUICK STATUS CHECK ===\n"

# Registri principali
printf "📍 REGISTRI:\n"
printf "  PC (Program Counter): 0x%08X\n", $pc
printf "  SP (Stack Pointer):   0x%08X\n", $sp
printf "  LR (Link Register):   0x%08X\n", $lr

# Backtrace per vedere dove siamo
printf "\n📚 CALL STACK:\n"
bt 5

# Memoria - indirizzi diretti delle variabili globali
printf "\n🧠 VARIABILI NN (indirizzi diretti):\n"
printf "  total_time_ms:       "
x/1uw 0x20001b8c
printf "  avg_time_us:         "
x/1uw 0x20001b90
printf "  benchmark_completed: "
x/1uw 0x20001b94
printf "  g_nn_state:          "
x/1uw 0x20001b98

# Stack top per vedere se c'è stack overflow
printf "\n💾 MEMORIA:\n"
printf "  Stack top (10 words):\n"
x/10wx $sp

# Heap status (se possibile)
printf "  Heap pointer (_sbrk):\n"
if $_memeq($pc, 0x0, 4) == 0
  # Solo se il sistema è in stato valido
  printf "    Sistema attivo\n"
else
  printf "    Sistema in stato critico\n"
end

printf "\n🔍 STATO ATTUALE:\n"
if $pc >= 0x5000 && $pc <= 0x6000
  printf "  ❌ Bloccato in funzione di sistema/delay\n"
elif $pc >= 0xd000 && $pc <= 0xe000  
  printf "  ✅ Nel codice principale\n"
elif $pc >= 0x10000
  printf "  🧠 Nel codice NN\n"
else
  printf "  ⚠️  Posizione sconosciuta\n"
end

printf "\n🎮 COMANDI:\n"
printf "  c      = continue execution\n"
printf "  s      = single step\n"
printf "  n      = next line\n"
printf "  interrupt = stop execution\n"
printf "=============================\n"