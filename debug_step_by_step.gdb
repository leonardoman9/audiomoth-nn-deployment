# ===============================================
# 🔍 AUDIOMOTH NN DEBUGGING - STEP BY STEP
# ===============================================

# Connessione al target
target remote localhost:2331
monitor reset
monitor halt

# Carica il firmware
load

# ===============================================
# 🎯 BREAKPOINTS STRATEGICI
# ===============================================

# 1. Inizio main
break main
commands 1
  printf "🚀 BREAKPOINT 1: MAIN STARTED\n"
  printf "   PC: 0x%08X\n", $pc
  printf "   Stack pointer: 0x%08X\n", $sp
end

# 2. Inizializzazione AudioMoth
break AudioMoth_initialise
commands 2
  printf "🔧 BREAKPOINT 2: AudioMoth_initialise()\n"
  printf "   About to initialize AudioMoth hardware\n"
end

# 3. Inizializzazione NN
break NN_Init
commands 3
  printf "🧠 BREAKPOINT 3: NN_Init()\n"
  printf "   About to initialize Neural Network\n"
  printf "   Available heap/stack before NN init\n"
end

# 4. Creazione backbone bridge
break tflite_cmsis_create
commands 4
  printf "🔗 BREAKPOINT 4: tflite_cmsis_create()\n"
  printf "   Creating TFLite-CMSIS bridge\n"
  printf "   Model type: %d\n", $r0
end

# 5. Inizializzazione bridge
break tflite_cmsis_init
commands 5
  printf "⚙️ BREAKPOINT 5: tflite_cmsis_init()\n"
  printf "   Initializing bridge with arena\n"
  printf "   Arena pointer: 0x%08X\n", $r1
  printf "   Arena size: %d\n", $r2
end

# 6. Performance benchmark start
break performanceBenchmark
commands 6
  printf "📊 BREAKPOINT 6: performanceBenchmark()\n"
  printf "   Starting performance test\n"
  printf "   N iterations: 5\n"
end

# 7. Backbone inference
break run_backbone_inference
commands 7
  printf "🎯 BREAKPOINT 7: run_backbone_inference()\n"
  printf "   Running backbone model inference\n"
end

# 8. Streaming inference
break run_streaming_inference
commands 8
  printf "🔄 BREAKPOINT 8: run_streaming_inference()\n"
  printf "   Running streaming model inference\n"
end

# 9. AudioMoth_feedWatchdog (per capire perché si blocca)
break AudioMoth_feedWatchdog
commands 9
  printf "🐕 BREAKPOINT 9: AudioMoth_feedWatchdog()\n"
  printf "   Watchdog feed - checking for infinite loop\n"
  printf "   Called from: "
  bt 2
end

# 10. Errori di memoria potenziali
break _sbrk
commands 10
  printf "💾 BREAKPOINT 10: _sbrk() - Memory allocation\n"
  printf "   Requested size: %d bytes\n", $r0
end

break HardFault_Handler if defined
commands 11
  printf "💥 BREAKPOINT 11: HARD FAULT!\n"
  printf "   PC: 0x%08X\n", $pc
  printf "   LR: 0x%08X\n", $lr
  printf "   SP: 0x%08X\n", $sp
end

# ===============================================
# 🚀 AVVIO ESECUZIONE
# ===============================================

printf "\n=== 🔍 AUDIOMOTH NN STEP-BY-STEP DEBUGGING ===\n"
printf "Breakpoints attivi:\n"
printf "  1. main() - Inizio programma\n"
printf "  2. AudioMoth_initialise() - Init hardware\n"
printf "  3. NN_Init() - Init neural network\n"
printf "  4. tflite_cmsis_create() - Create bridge\n"
printf "  5. tflite_cmsis_init() - Init bridge\n"
printf "  6. performanceBenchmark() - Start benchmark\n"
printf "  7. run_backbone_inference() - Backbone inference\n"
printf "  8. run_streaming_inference() - Streaming inference\n"
printf "  9. AudioMoth_feedWatchdog() - Watchdog (loop check)\n"
printf "  10. _sbrk() - Memory allocation\n"
printf "  11. HardFault_Handler() - System fault\n"
printf "\n"
printf "🎮 COMANDI UTILI:\n"
printf "  c         = continue\n"
printf "  n         = next (step over)\n"
printf "  s         = step (step into)\n"
printf "  finish    = step out\n"
printf "  bt        = backtrace\n"
printf "  info reg  = show registers\n"
printf "  x/10wx $sp = show stack\n"
printf "\n"
printf "⏯️  Premere 'c' per iniziare...\n"

# Non avviamo automaticamente - lasciamo il controllo all'utente
# continue