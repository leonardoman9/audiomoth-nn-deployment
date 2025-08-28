echo 🎯 AUDIOMOTH FLASH SWAP DEBUG
echo =============================

# Connect to J-Link GDB Server
target remote localhost:2331

# Reset and load firmware
monitor reset 1
load
monitor reset 0

# Breakpoints for LED sequence analysis
break main
break NN_Init

# Breakpoint PRIMA di ogni sequenza LED
break nn_model.c:71
break nn_model.c:76
break nn_model.c:83
break nn_model.c:85
break nn_model.c:96

# Breakpoints per initialize_models  
break nn_model.c:315
break nn_model.c:324
break nn_model.c:337
break nn_model.c:358

# Breakpoints DENTRO i loop LED per contare
break nn_model.c:87
break nn_model.c:98
break nn_model.c:327
break nn_model.c:340
break nn_model.c:361

echo 
echo 📍 Breakpoints LED Debug:
echo - Line 71: Prima LED rosso iniziale
echo - Line 76: Prima Verde+Rosso (Flash Swap mode)
echo - Line 83: Prima controllo NN_SwapArena_Init
echo - Line 85: DENTRO 10 LED rossi (se fallisce)
echo - Line 96: Prima Verde-Rosso alternati (se OK)
echo - Line 315: Prima initialize_models
echo - Line 324: Prima backbone creation
echo - Line 337: Prima backbone interpreter
echo - Line 358: Prima backbone tensors
echo 

echo 🚀 Starting execution...
continue