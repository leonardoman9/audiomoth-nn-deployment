#!/bin/bash
# 🔍 SCRIPT VERIFICA BUILD AUDIOMOTH NN
# Verifica automatica dei parametri critici post-compilazione

echo "🎯 AUDIOMOTH NN BUILD VERIFICATION"
echo "=================================="

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# File da verificare
AXF_FILE="audiomoth.axf"
MAP_FILE="audiomoth.map"

if [ ! -f "$AXF_FILE" ]; then
    echo -e "${RED}❌ File $AXF_FILE non trovato!${NC}"
    echo "Esegui 'make' prima di questo script"
    exit 1
fi

echo "📁 Analizzando: $AXF_FILE"
echo ""

# 1. VERIFICA .BSS SOTTO LIMITE CRITICO
echo "🔍 1. VERIFICA .BSS SECTION"
echo "----------------------------"

BSS_SIZE=$(arm-none-eabi-size -A "$AXF_FILE" | grep "\.bss" | awk '{print $2}')
BSS_LIMIT=4096

echo "   .bss size: ${BSS_SIZE} bytes"
echo "   Limite critico: ${BSS_LIMIT} bytes"

if [ "$BSS_SIZE" -gt "$BSS_LIMIT" ]; then
    echo -e "${RED}❌ .bss=${BSS_SIZE}B > ${BSS_LIMIT}B limit!${NC}"
    echo -e "${YELLOW}   ⚠️  Sistema instabile - ridurre allocazioni statiche${NC}"
    BSS_OK=false
else
    echo -e "${GREEN}✅ .bss=${BSS_SIZE}B < ${BSS_LIMIT}B OK${NC}"
    BSS_OK=true
fi

echo ""

# 2. VERIFICA SIMBOLI VIRTUAL ARENA
echo "🔍 2. VERIFICA VIRTUAL ARENA SYMBOLS"
echo "-------------------------------------"

ARENA_START=$(arm-none-eabi-nm "$AXF_FILE" | grep __tensor_arena_start__ | cut -d' ' -f1)
ARENA_END=$(arm-none-eabi-nm "$AXF_FILE" | grep __tensor_arena_end__ | cut -d' ' -f1)

if [ -z "$ARENA_START" ] || [ -z "$ARENA_END" ]; then
    echo -e "${RED}❌ Simboli Virtual Arena non trovati!${NC}"
    echo "   Verifica linker script audiomoth_flash_swap.ld"
    ARENA_OK=false
elif [ "$ARENA_START" = "$ARENA_END" ]; then
    echo -e "${RED}❌ Virtual Arena size = 0!${NC}"
    echo "   Start: 0x$ARENA_START"
    echo "   End:   0x$ARENA_END"
    echo "   Verifica sezione .tensor_arena nel linker script"
    ARENA_OK=false
else
    ARENA_SIZE=$((0x$ARENA_END - 0x$ARENA_START))
    echo -e "${GREEN}✅ Virtual Arena: ${ARENA_SIZE}B${NC}"
    echo "   Start: 0x$ARENA_START"
    echo "   End:   0x$ARENA_END"
    ARENA_OK=true
fi

echo ""

# 3. ANALISI MEMORY USAGE
echo "🔍 3. MEMORY USAGE ANALYSIS"
echo "----------------------------"

arm-none-eabi-size -A "$AXF_FILE" | head -10

echo ""

# 4. VERIFICA TOTAL RAM USAGE
echo "🔍 4. TOTAL RAM CALCULATION"
echo "---------------------------"

DATA_SIZE=$(arm-none-eabi-size -A "$AXF_FILE" | grep "\.data" | awk '{print $2}')
TOTAL_STATIC=$((BSS_SIZE + DATA_SIZE))
RAM_TOTAL=32768
RAM_FREE=$((RAM_TOTAL - TOTAL_STATIC))

echo "   .data: ${DATA_SIZE}B"
echo "   .bss:  ${BSS_SIZE}B"
echo "   -------------------------"
echo "   Static total: ${TOTAL_STATIC}B"
echo "   RAM total:    ${RAM_TOTAL}B"
echo "   RAM free:     ${RAM_FREE}B (per heap/stack)"

if [ "$TOTAL_STATIC" -gt 8192 ]; then
    echo -e "${YELLOW}   ⚠️  Static usage > 8KB - verifica heap space${NC}"
fi

echo ""

# 5. VERIFICA CONFIGURAZIONE NN
echo "🔍 5. NN CONFIGURATION CHECK"
echo "-----------------------------"

if [ -f "../inc/nn/nn_config.h" ]; then
    GRU_DIM=$(grep "NN_GRU_HIDDEN_DIM" ../inc/nn/nn_config.h | head -1 | awk '{print $3}')
    CLASSES=$(grep "NN_NUM_CLASSES" ../inc/nn/nn_config.h | head -1 | awk '{print $3}')
    INPUT_H=$(grep "NN_INPUT_HEIGHT" ../inc/nn/nn_config.h | head -1 | awk '{print $3}')
    INPUT_W=$(grep "NN_INPUT_WIDTH" ../inc/nn/nn_config.h | head -1 | awk '{print $3}')
    
    echo "   GRU Hidden Dim: $GRU_DIM"
    echo "   Classes: $CLASSES"
    echo "   Input: [${INPUT_W}×${INPUT_H}]"
    
    if [ "$GRU_DIM" = "64" ] && [ "$BSS_OK" = "true" ]; then
        echo -e "${GREEN}✅ GRU-64 model with stable .bss!${NC}"
    elif [ "$GRU_DIM" = "64" ] && [ "$BSS_OK" = "false" ]; then
        echo -e "${RED}❌ GRU-64 model but .bss too large!${NC}"
    fi
else
    echo -e "${YELLOW}   ⚠️  nn_config.h non trovato${NC}"
fi

echo ""

# 6. VERDETTO FINALE
echo "🎯 VERDETTO FINALE"
echo "==================="

if [ "$BSS_OK" = "true" ] && [ "$ARENA_OK" = "true" ]; then
    echo -e "${GREEN}🎉 BUILD OK! Pronto per flash e test${NC}"
    echo ""
    echo "📋 PROSSIMI PASSI:"
    echo "   1. Flash: arm-none-eabi-gdb audiomoth.axf -x debug_basic.gdb"
    echo "   2. Monitor LED sequence per verifica funzionamento"
    echo "   3. Benchmark performance con DWT cycle counter"
    exit 0
else
    echo -e "${RED}❌ BUILD PROBLEMATICO - Risolvere errori prima del flash${NC}"
    echo ""
    echo "🔧 SUGGERIMENTI:"
    if [ "$BSS_OK" = "false" ]; then
        echo "   - Convertire static arrays in malloc()"
        echo "   - Ridurre RAM_CACHE_SIZE in virtual_arena.h"
        echo "   - Verificare MAX_VIRTUAL_TENSORS"
    fi
    if [ "$ARENA_OK" = "false" ]; then
        echo "   - Verificare linker script audiomoth_flash_swap.ld"
        echo "   - Controllare sezione .tensor_arena"
    fi
    exit 1
fi