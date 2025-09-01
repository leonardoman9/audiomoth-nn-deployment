#****************************************************************************
# Makefile with Smart TFLM Wrapper
# Uses real neural network math instead of dummy calculations
#****************************************************************************

# Set the toolpath parameter to the location of the GNU Arm Embedded Toolchain
TOOLCHAIN_PATH = /Users/leonardomannini/Documents/Toolchains/
TOOLCHAIN_VERSION = gcc-arm-none-eabi-10.3-2021.10

# Determine the path to the toolchain binary
TOOLPATH = $(TOOLCHAIN_PATH)/$(TOOLCHAIN_VERSION)/bin/

# Standard AudioMoth includes
INC = ../cmsis ../device/inc ../emlib/inc ../emusb/inc ../drivers/inc ../fatfs/inc  ../gps/inc ../inc ../inc/nn ../third_party ../third_party/CMSIS-NN/Include

SRC = ../device/src ../emlib/src ../emusb/src ../drivers/src ../fatfs/src ../src ../src/nn ../third_party

# Exclude files for NN-only mode + old dummy wrapper
EXCLUDED_FILES = ../emlib/src/em_crypto.c \
                 ../emlib/src/em_aes.c \
                 ../emlib/src/em_can.c \
                 ../emlib/src/em_lcd.c \
                 ../emlib/src/em_pcnt.c \
                 ../emlib/src/em_lesense.c \
                 ../emlib/src/em_qspi.c \
                 ../emlib/src/em_csen.c \
                 ../emlib/src/em_vdac.c \
                 ../emlib/src/em_idac.c \
                 ../fatfs/src/ff.c \
                 ../fatfs/src/ffunicode.c \
                 ../fatfs/src/diskio.c \
                 ../src/nn/nn_flash_swap.c \
                 ../src/nn/virtual_arena.c \
                 ../third_party/tflm_wrapper_virtual.c \
                 ../third_party/tflm_wrapper.c \
                 ../third_party/tflm_wrapper.cpp \
                 ../third_party/mini_tflm.c \
                 ../third_party/tflm_wrapper_mini.c

# Add smart wrapper sources (old files removed)
SMART_WRAPPER_SRCS =

# Rimuoviamo i vecchi wrapper C++ TFLM (sostituiti da Mini-TFLM)
# CPPSRC = ../third_party/tflm_wrapper.cpp
# STUB_SRCS = ../third_party/syscall_stubs.cpp

# ================================================================
# DIRECT INFERENCE: Engine diretto per AudioMoth (<80KB)
# Bypassa TFLM completamente, usa arrays C pre-convertiti
# ================================================================

# CMSIS Wrapper sources (pure C + CMSIS-NN)
CMSIS_WRAPPER_SRCS = \
	../third_party/cmsis_wrapper.c \
	../third_party/backbone_weights.c \
	../third_party/streaming_weights.c

# CMSIS-NN source files (only the ones we need)
CMSIS_NN_SRCS = $(wildcard ../third_party/CMSIS-NN/Source/ConvolutionFunctions/*.c) \
                $(wildcard ../third_party/CMSIS-NN/Source/FullyConnectedFunctions/*.c) \
                $(wildcard ../third_party/CMSIS-NN/Source/SoftmaxFunctions/*.c) \
                $(wildcard ../third_party/CMSIS-NN/Source/ActivationFunctions/*.c) \
                $(wildcard ../third_party/CMSIS-NN/Source/BasicMathFunctions/*.c) \
                $(wildcard ../third_party/CMSIS-NN/Source/NNSupportFunctions/*.c)

# Sostituiamo tutto TFLM con CMSIS Wrapper + CMSIS-NN
TFLM_MINIMAL_SRCS = $(CMSIS_WRAPPER_SRCS) $(CMSIS_NN_SRCS)

# Combine all C sources - MINI-TFLM VERSION (solo C, no C++)
ALL_CSRC_EXTRA = $(TFLM_MINIMAL_SRCS)

# Set the name of the output files
FILENAME = audiomoth_smart

# This is the location of the resulting object files
OBJPATH = ./objects/

# Set the target EFM32WG to that used in AudioMoth
TARGET = EFM32WG380F256

# The following code generates the list of objects and the search path of source and header files
VPATH = $(SRC)

IFLAGS = $(foreach d, $(INC), -I$d)

# Collect all C and S source files, then filter out excluded files
_CSRC_ALL = $(notdir $(foreach d, $(SRC), $(wildcard $d/*.c)))
_CSRC = $(filter-out $(notdir $(EXCLUDED_FILES)), $(_CSRC_ALL))
_SSRC = $(notdir $(foreach d, $(SRC), $(wildcard $d/*.s)))

# Add Mini-TFLM C sources (sostituisce tutto il C++)
_MINI_TFLM_CSRC = $(notdir $(ALL_CSRC_EXTRA))

# Solo file C ora (no più C++)
_OBJ = $(_CSRC:.c=.o) $(_SSRC:.s=.o) $(_MINI_TFLM_CSRC:.c=.o)

OBJ = $(sort $(foreach d, $(_OBJ), $(OBJPATH)$d))

DEP = $(OBJ:.o=.d)

# These are the compilation settings
CC = $(TOOLPATH)arm-none-eabi-gcc
CXX = $(TOOLPATH)arm-none-eabi-g++

CSIZE = $(TOOLPATH)arm-none-eabi-size

COBJCOPY = $(TOOLPATH)arm-none-eabi-objcopy

COBJDUMP = $(TOOLPATH)arm-none-eabi-objdump

# Challenge-grade optimization flags for <250KB target
CHALLENGE_OPT = -Os -flto -DNDEBUG -DTFLITE_MCU -DTF_LITE_STATIC_MEMORY -DTF_LITE_DISABLE_X86_NEON

# Compiler flags with ARM optimizations + Challenge optimization
CFLAGS = -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -mthumb -Wall '-DARM_MATH_CM4=1' '-D$(TARGET)=1' $(CHALLENGE_OPT) -std=c99

DFLAGS = -MMD

# Build rules

# C files
$(OBJPATH)%.o: %.c
	@mkdir -p $(OBJPATH)
	@echo 'Building C file' $@
	@$(CC) $(CFLAGS) $(DFLAGS) -c -o "$@" "$<" $(IFLAGS)

# Rule for C++ files (.cpp) - Challenge optimization
$(OBJPATH)%.o: %.cpp
	@mkdir -p $(OBJPATH)
	@echo 'Building C++ file' $@
	@$(CXX) $(CFLAGS) $(DFLAGS) -c -o "$@" "$<" $(IFLAGS) -fno-rtti -fno-exceptions -std=c++11

# Rule for TFLM C++ files (.cc) - Challenge optimization
$(OBJPATH)%.o: %.cc
	@mkdir -p $(OBJPATH)
	@echo 'Building TFLM C++ file' $@
	@$(CXX) $(CFLAGS) $(DFLAGS) -c -o "$@" "$<" $(IFLAGS) -fno-rtti -fno-exceptions -std=c++11

# Assembly files
$(OBJPATH)%.o: %.s
	@mkdir -p $(OBJPATH)
	@echo 'Building ASM file' $@
	@$(CC) $(CFLAGS) $(DFLAGS) -x assembler-with-cpp -c -o "$@" "$<" $(IFLAGS)

# Smart wrapper sources (may be in different directories)
# Mini-TFLM vpath (solo C files)
vpath %.c ../third_party

# CMSIS-NN vpath for source files
vpath %.c ../third_party/CMSIS-NN/Source/ConvolutionFunctions
vpath %.c ../third_party/CMSIS-NN/Source/FullyConnectedFunctions
vpath %.c ../third_party/CMSIS-NN/Source/SoftmaxFunctions
vpath %.c ../third_party/CMSIS-NN/Source/ActivationFunctions
vpath %.c ../third_party/CMSIS-NN/Source/BasicMathFunctions
vpath %.c ../third_party/CMSIS-NN/Source/NNSupportFunctions

$(FILENAME).bin: $(OBJ)
	@echo 'Building' $(FILENAME).axf
	@$(CXX) $(CFLAGS) $(DFLAGS) -T "audiomoth_flash_swap.ld" -Xlinker --gc-sections -Xlinker -Map="$(FILENAME).map" --specs=nano.specs -o $(FILENAME).axf $(OBJ) -lm -Wl,--start-group -lgcc -lc -lnosys -lstdc++ -Wl,--end-group
	@echo 'Building' $(FILENAME).lst
	@$(COBJDUMP) -d -t -S "$(FILENAME).axf" > "$(FILENAME).lst" 
	@echo 'Building' $(FILENAME).hex
	@$(COBJCOPY) -O ihex "$(FILENAME).axf" "$(FILENAME).hex"
	@echo 'Building' $(FILENAME).bin
	@$(COBJCOPY) -O binary "$(FILENAME).axf" "$(FILENAME).bin"
	@$(CSIZE) -A "$(FILENAME).axf"
	@echo ""
	@echo "================================================="
	@echo "🚀 AUDIOMOTH NN FIRMWARE - TIMING ANALYSIS"
	@echo "================================================="
	@echo "📊 TIMING VARIABLES (Global symbols):"
	@$(NM) $(FILENAME).axf | grep -E "total_time_ms|avg_time_us|time_100_inferences_ms|benchmark_completed" | while read addr type sym; do echo "  $$sym @ 0x$$addr"; done || echo "  ⚠️  Timing symbols not found"
	@echo ""
	@echo "🧠 NEURAL NETWORK SYMBOLS:"
	@$(NM) $(FILENAME).axf | grep -E "NN_|nn_|tflm_|cmsis_" | head -10 | while read addr type sym; do echo "  $$sym @ 0x$$addr"; done || echo "  ⚠️  NN symbols not found"
	@echo ""
	@echo "💾 MEMORY USAGE ANALYSIS:"
	@$(OBJDUMP) -h $(FILENAME).axf | grep -E "\.text|\.data|\.bss|tensor_arena" | while read idx name size vma lma; do echo "  $$name: $$size bytes @ $$vma"; done
	@echo ""
	@echo "🎯 INFERENCE ENGINE STATUS:"
	@echo "  - Engine: CMSIS-NN wrapper with real TFLite weights"
	@echo "  - Models: backbone (18x40→32) + streaming (32+32→32+5)"
	@echo "  - Arena: 64KB external SRAM"
	@echo "  - Operators: Conv2D, Dense, Softmax, Reshape, Add, Mul, ReLU"
	@echo "================================================="

-include $(DEP)

.PHONY: clean
clean:
	rm -f $(OBJPATH)*.o
	rm -f $(OBJPATH)*.d
	rm -f $(FILENAME).axf
	rm -f $(FILENAME).lst
	rm -f $(FILENAME).bin
	rm -f $(FILENAME).hex
	rm -f $(FILENAME).map

.PHONY: info
info:
	@echo "Smart wrapper sources: $(words $(SMART_WRAPPER_SRCS)) files"
	@echo "Total objects: $(words $(OBJ)) files"
	@echo "Excluded dummy wrapper files"