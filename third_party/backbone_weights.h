#ifndef BACKBONE_WEIGHTS_H
#define BACKBONE_WEIGHTS_H

#include <stdint.h>
#include <stddef.h>

// ================================================================
// BACKBONE MODEL STRUCTURE (REAL FROM TFLITE)
// ================================================================

// Input 0: serving_default_spectrogram:0
#define BACKBONE_INPUT_0_TOTAL 720

// Output 0: StatefulPartitionedCall:0
#define BACKBONE_OUTPUT_0_SIZE 18

// Weight arrays (extern declarations)
extern const int8_t backbone_serving_default_spectrogram_0[720];
extern const float backbone_arith_constant[6];
extern const float backbone_arith_constant1[4];
extern const float backbone_arith_constant2[3];
extern const float backbone_arith_constant4[4];
extern const float backbone_arith_constant5[3];
extern const float backbone_arith_constant6[4];
extern const float backbone_arith_constant7[3];
extern const float backbone_arith_constant8[6];
extern const float backbone_arith_constant9[4];
extern const float backbone_arith_constant10[4];
extern const float backbone_arith_constant11[3];
extern const float backbone_arith_constant12[4];
extern const float backbone_arith_constant13[6];
extern const float backbone_arith_constant14[4];
extern const float backbone_arith_constant15[3];
extern const float backbone_arith_constant16[6];
extern const float backbone_arith_constant17[4];
extern const float backbone_arith_constant18[4];
extern const float backbone_arith_constant19[3];
extern const float backbone_arith_constant20[4];
extern const int8_t backbone_tfl_pseudo_qconst[32];
extern const int8_t backbone_tfl_pseudo_qconst1[1024];
extern const int8_t backbone_tfl_pseudo_qconst2[160];
extern const float backbone_tfl_pseudo_qconst3[5];
extern const int8_t backbone_tfl_pseudo_qconst4[160];
extern const int8_t backbone_tfl_pseudo_qconst5[96];
extern const int8_t backbone_tfl_pseudo_qconst6[512];
extern const int8_t backbone_tfl_pseudo_qconst7[32];
extern const int8_t backbone_tfl_pseudo_qconst8[32];
extern const int8_t backbone_tfl_pseudo_qconst9[80];
extern const float backbone_tfl_pseudo_qconst10[32];
extern const int8_t backbone_tfl_pseudo_qconst11[512];
extern const int8_t backbone_tfl_pseudo_qconst12[256];
extern const int8_t backbone_tfl_pseudo_qconst13[32];
extern const int8_t backbone_tfl_pseudo_qconst14[32];
extern const int8_t backbone_tfl_pseudo_qconst15[48];
extern const int8_t backbone_tfl_pseudo_qconst16[256];
extern const int8_t backbone_tfl_pseudo_qconst17[32];
extern const int8_t backbone_tfl_pseudo_qconst18[32];
extern const int8_t backbone_tfl_pseudo_qconst19[48];
extern const int8_t backbone_tfl_pseudo_qconst20[256];
extern const int8_t backbone_tfl_pseudo_qconst21[32];
extern const int8_t backbone_tfl_pseudo_qconst22[32];
extern const int8_t backbone_tfl_pseudo_qconst23[48];
extern const int8_t backbone_tfl_pseudo_qconst24[16];
extern const int8_t backbone_tfl_pseudo_qconst25[256];
extern const int8_t backbone_tfl_pseudo_qconst26[32];
extern const float backbone_tfl_pseudo_qconst27[2];
extern const float backbone_tfl_pseudo_qconst28[2];
extern const float backbone_tfl_pseudo_qconst29[2];
extern const float backbone_tfl_pseudo_qconst30[2];
extern const float backbone_tfl_pseudo_qconst31[2];
extern const int8_t backbone_tfl_pseudo_qconst32[32];
extern const float backbone_tfl_pseudo_qconst33[16];
extern const float backbone_tfl_pseudo_qconst34[16];
extern const float backbone_tfl_pseudo_qconst35[16];
extern const float backbone_tfl_pseudo_qconst36[16];
extern const float backbone_tfl_pseudo_qconst37[16];
extern const float backbone_tfl_pseudo_qconst38[16];
extern const float backbone_tfl_pseudo_qconst39[16];
extern const float backbone_tfl_pseudo_qconst40[16];
extern const float backbone_tfl_pseudo_qconst41[16];
extern const float backbone_tfl_pseudo_qconst42[16];
extern const float backbone_tfl_pseudo_qconst43[16];
extern const float backbone_tfl_pseudo_qconst44[16];
extern const float backbone_tfl_pseudo_qconst45[16];
extern const float backbone_tfl_pseudo_qconst46[16];
extern const int8_t backbone_tfl_pseudo_qconst47[80];
extern const float backbone_tfl_pseudo_qconst48[16];
extern const int8_t backbone_tfl_pseudo_qconst49[512];
extern const float backbone_tfl_pseudo_qconst50[32];
extern const float backbone_tfl_pseudo_qconst51[32];
extern const float backbone_tfl_pseudo_qconst52[32];
extern const float backbone_tfl_pseudo_qconst53[32];
extern const float backbone_tfl_pseudo_qconst54[32];
extern const int8_t backbone_tfl_pseudo_qconst55[1280];
extern const int8_t backbone_tfl_pseudo_qconst56[240];
extern const float backbone_tfl_pseudo_qconst57[6];
extern const int8_t backbone_tfl_pseudo_qconst58[240];
extern const float backbone_tfl_pseudo_qconst59[40];
extern const float backbone_tfl_pseudo_qconst60[40];
extern const int8_t backbone_tfl_pseudo_qconst61[280];
extern const int8_t backbone_StatefulPartitionedCall_0[576];

#endif // BACKBONE_WEIGHTS_H
