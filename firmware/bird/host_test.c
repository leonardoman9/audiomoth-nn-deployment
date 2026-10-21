// Host check of bird_tail.c against the PyTorch reference (feats.bin / logits.bin).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "bird_tail.h"
#include "tail_dims.h"

int main(int argc, char** argv) {
  static float feats[BIRD_TAIL_FRAMES * BIRD_FEATURES];
  float ref[BIRD_CLASSES], got[BIRD_CLASSES];
  FILE* f = fopen(argv[1], "rb");
  if (!f || fread(feats, 4, BIRD_TAIL_FRAMES * BIRD_FEATURES, f) != BIRD_TAIL_FRAMES * BIRD_FEATURES) return 2;
  fclose(f);
  f = fopen(argv[2], "rb");
  if (!f || fread(ref, 4, BIRD_CLASSES, f) != BIRD_CLASSES) return 3;
  fclose(f);
  BirdTail t;
  BirdTail_Reset(&t);
  for (int i = 0; i < BIRD_TAIL_FRAMES; ++i) BirdTail_Step(&t, feats + i * BIRD_FEATURES);
  BirdTail_Finish(&t, got);
  float worst = 0; int a = 0, b = 0;
  for (int i = 0; i < BIRD_CLASSES; ++i) {
    worst = fmaxf(worst, fabsf(got[i] - ref[i]));
    if (got[i] > got[a]) a = i;
    if (ref[i] > ref[b]) b = i;
  }
  printf("max |C - PyTorch| = %.3g, argmax C=%d PyTorch=%d\n", worst, a, b);
  return 0;
}
