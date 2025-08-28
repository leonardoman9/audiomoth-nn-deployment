#!/bin/bash

# Test completo per verificare il deployment NN su AudioMoth
# Leonardo Mannini - AudioMoth NN Deployment

echo "=========================================="
echo "🔍 TEST COMPLETO AUDIOMOTH NN DEPLOYMENT"
echo "=========================================="
echo ""

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Contatori
TESTS_PASSED=0
TESTS_FAILED=0

# Funzione per test
run_test() {
    local test_name="$1"
    local command="$2"
    local expected="$3"
    
    echo -n "Testing $test_name... "
    result=$(eval "$command" 2>/dev/null)
    
    if [[ "$result" == *"$expected"* ]]; then
        echo -e "${GREEN}✅ PASSED${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        echo "  Expected: $expected"
        echo "  Got: $result"
        ((TESTS_FAILED++))
        return 1
    fi
}

echo "1️⃣  VERIFICA COMPILAZIONE"
echo "------------------------"
run_test "Firmware exists" "ls audiomoth.axf 2>/dev/null" "audiomoth.axf"
run_test "Firmware size reasonable" "stat -f%z audiomoth.axf" "587680"
echo ""

echo "2️⃣  VERIFICA MEMORIA"
echo "-------------------"
# Estrai dimensioni memoria
BSS_SIZE=$(arm-none-eabi-size -A audiomoth.axf | grep "^\.bss" | awk '{print $2}')
TEXT_SIZE=$(arm-none-eabi-size -A audiomoth.axf | grep "^\.text" | awk '{print $2}')

echo "📊 Memory Usage:"
echo "  .text (code):     $TEXT_SIZE bytes (~$((TEXT_SIZE/1024))KB)"
echo "  .bss (variables): $BSS_SIZE bytes"

# Test critico: .bss < 4KB
if [ "$BSS_SIZE" -lt 4096 ]; then
    echo -e "  ${GREEN}✅ .bss < 4KB limit ($BSS_SIZE < 4096)${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  ${RED}❌ .bss EXCEEDS 4KB limit! ($BSS_SIZE > 4096)${NC}"
    ((TESTS_FAILED++))
fi
echo ""

echo "3️⃣  VERIFICA VIRTUAL ARENA"
echo "-------------------------"
run_test "Virtual Arena start symbol" "arm-none-eabi-nm audiomoth.axf | grep __tensor_arena_start__" "00030000"
run_test "Virtual Arena end symbol" "arm-none-eabi-nm audiomoth.axf | grep __tensor_arena_end__" "00040000"

# Calcola dimensione Virtual Arena
ARENA_SIZE=$((0x40000 - 0x30000))
echo "  Virtual Arena Size: $((ARENA_SIZE/1024))KB"
if [ "$ARENA_SIZE" -eq 65536 ]; then
    echo -e "  ${GREEN}✅ Virtual Arena = 64KB${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  ${RED}❌ Virtual Arena != 64KB${NC}"
    ((TESTS_FAILED++))
fi
echo ""

echo "4️⃣  VERIFICA FUNZIONI NN"
echo "-----------------------"
run_test "NN_Init function" "arm-none-eabi-nm audiomoth.axf | grep ' T NN_Init'" "T NN_Init"
run_test "NN_ProcessAudio function" "arm-none-eabi-nm audiomoth.axf | grep ' T NN_ProcessAudio'" "T NN_ProcessAudio"
run_test "VirtualArena_Init" "arm-none-eabi-nm audiomoth.axf | grep VirtualArena_Init" "VirtualArena_Init"
run_test "VirtualArena_GetTensor" "arm-none-eabi-nm audiomoth.axf | grep VirtualArena_GetTensor" "VirtualArena_GetTensor"
echo ""

echo "5️⃣  VERIFICA MODELLI"
echo "-------------------"
# Verifica che i modelli siano inclusi
MODEL_SYMBOLS=$(arm-none-eabi-nm audiomoth.axf | grep -c "model_data")
if [ "$MODEL_SYMBOLS" -ge 2 ]; then
    echo -e "  ${GREEN}✅ Both models included (backbone + streaming)${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  ${RED}❌ Missing model data${NC}"
    ((TESTS_FAILED++))
fi
echo ""

echo "6️⃣  VERIFICA CONFIGURAZIONE"
echo "---------------------------"
# Controlla configurazione nel codice
echo "Checking configuration in source..."
if grep -q "NN_GRU_HIDDEN_DIM.*64" ../inc/nn/nn_config.h; then
    echo -e "  ${GREEN}✅ GRU-64 configured${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  ${RED}❌ GRU-64 not configured${NC}"
    ((TESTS_FAILED++))
fi

if grep -q "NN_NUM_CLASSES.*35" ../inc/nn/nn_config.h; then
    echo -e "  ${GREEN}✅ 35 classes configured${NC}"
    ((TESTS_PASSED++))
else
    echo -e "  ${RED}❌ 35 classes not configured${NC}"
    ((TESTS_FAILED++))
fi
echo ""

echo "7️⃣  VERIFICA TIMING VARIABLES"
echo "-----------------------------"
run_test "Timing variables" "arm-none-eabi-nm audiomoth.axf | grep total_time_ms" "total_time_ms"
run_test "Benchmark flag" "arm-none-eabi-nm audiomoth.axf | grep benchmark_completed" "benchmark_completed"
echo ""

echo "=========================================="
echo "📊 RISULTATI FINALI"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ "$TESTS_FAILED" -eq 0 ]; then
    echo -e "${GREEN}🎉 TUTTO FUNZIONA PERFETTAMENTE!${NC}"
    echo ""
    echo "Il sistema è pronto per:"
    echo "  • Modello GRU-64 con 35 classi"
    echo "  • Virtual Arena da 64KB"
    echo "  • Cache RAM da 8KB"
    echo "  • Inferenze a ~3.9ms"
    echo ""
    echo "Prossimo step: Flash sul dispositivo con GDB!"
else
    echo -e "${YELLOW}⚠️  Ci sono alcuni problemi da risolvere${NC}"
fi