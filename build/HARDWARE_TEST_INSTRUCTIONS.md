# 🔌 HARDWARE TEST INSTRUCTIONS - AudioMoth Neural Network

## 📋 PREREQUISITI
- AudioMoth device connesso via J-Link
- J-Link GDB Server in esecuzione su porta 2331
- Terminal/console per output

## 🚀 STEP-BY-STEP DEPLOYMENT

### 1. AVVIA J-LINK GDB SERVER
```bash
# In terminal separato
JLinkGDBServer -device EFM32WG990F256 -speed 4000 -if SWD -port 2331
```

### 2. FLASH E TEST FIRMWARE
```bash
# Nel build directory
arm-none-eabi-gdb -x test_bridge_hardware.gdb
```

### 3. COMANDI GDB DURANTE IL TEST
- `c` - Continua esecuzione
- `n` - Next step
- `s` - Step into
- `info registers` - Stato registri
- `p variable_name` - Print variabile
- `q` - Quit

## 📊 OUTPUT ATTESO

### Sequenza normale:
1. ✅ MAIN STARTED
2. ✅ NN_INIT STARTED
3. ✅ BRIDGE CREATION STARTED  
4. ✅ TFLITE-CMSIS BRIDGE CREATION
5. ✅ BACKBONE INFERENCE STARTED
6. ✅ CMSIS-NN INFERENCE EXECUTION
7. ✅ INFERENCE COMPLETED

### Performance results attesi:
- `total_time_ms`: 20-100ms
- `avg_time_us`: 4000-20000us  
- `benchmark_completed`: 1

## 🐛 TROUBLESHOOTING

### Se connection failed:
- Verifica J-Link connessione
- Controlla porta 2331
- Reset AudioMoth device

### Se load failed:
- Verifica firmware size (81KB)
- Controlla memoria flash disponibile
- Reset e riprova

### Se inference failed:
- Controlla tensor arena allocation
- Verifica CMSIS-NN function calls
- Analizza bridge initialization

## 📈 SUCCESS METRICS
- ✅ Firmware loads (81KB)
- ✅ Bridge initializes
- ✅ Backbone inference runs
- ✅ Performance < 50ms per inference
- ✅ No crashes or hard faults

