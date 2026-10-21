#include "bird_tail.h"

#include <math.h>
#include <string.h>

#ifdef BIRD_TAIL_FLOAT_WEIGHTS
#include "bird_tail_weights_f32.h"
#define MATVEC(out, rows, cols, w, scale, x) MatVecF(out, rows, cols, w, x)
#else
#include "bird_tail_weights_i8.h"
#define MATVEC(out, rows, cols, w, scale, x) MatVecQ(out, rows, cols, w, scale, x)
#endif

#define GATES (3 * BIRD_HIDDEN)

#ifdef BIRD_TAIL_FLOAT_WEIGHTS
static void MatVecF(float* out, int rows, int cols, const float* w, const float* x) {
  for (int r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc += w[r * cols + c] * x[c];
    out[r] = acc;
  }
}
#else
// Row-wise symmetric int8 weights: out = scale[r] * sum_c w[r][c] * x[c].
static void MatVecQ(float* out, int rows, int cols, const int8_t* w, const float* scale, const float* x) {
  for (int r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (int c = 0; c < cols; ++c) acc += (float)w[r * cols + c] * x[c];
    out[r] = acc * scale[r];
  }
}
#endif

static float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void BirdTail_Reset(BirdTail* t) {
  memset(t, 0, sizeof(*t));
  t->max_score = -INFINITY;
}

void BirdTail_Step(BirdTail* t, const float* feature) {
  float gi[GATES], gh[GATES];
  MATVEC(gi, GATES, BIRD_FEATURES, k_w_ih, k_w_ih_scale, feature);
  MATVEC(gh, GATES, BIRD_HIDDEN, k_w_hh, k_w_hh_scale, t->h);
  // PyTorch gate order: reset, update, new.
  for (int i = 0; i < BIRD_HIDDEN; ++i) {
    const float r = Sigmoid(gi[i] + k_b_ih[i] + gh[i] + k_b_hh[i]);
    const float z = Sigmoid(gi[BIRD_HIDDEN + i] + k_b_ih[BIRD_HIDDEN + i] + gh[BIRD_HIDDEN + i] + k_b_hh[BIRD_HIDDEN + i]);
    const float n = tanhf(gi[2 * BIRD_HIDDEN + i] + k_b_ih[2 * BIRD_HIDDEN + i] +
                          r * (gh[2 * BIRD_HIDDEN + i] + k_b_hh[2 * BIRD_HIDDEN + i]));
    t->h[i] = (1.0f - z) * n + z * t->h[i];
  }
  // Projection and attention score for this frame.
  float p[BIRD_HIDDEN], score;
  MATVEC(p, BIRD_HIDDEN, BIRD_HIDDEN, k_w_p, k_w_p_scale, t->h);
  for (int i = 0; i < BIRD_HIDDEN; ++i) p[i] += k_b_p[i];
  MATVEC(&score, 1, BIRD_HIDDEN, k_w_a, k_w_a_scale, p);
  score += k_b_a[0];
  // Online softmax over time: keeps O(hidden) memory instead of T x hidden.
  if (score > t->max_score) {
    const float rescale = expf(t->max_score - score);  // 0 on the first frame (-inf)
    for (int i = 0; i < BIRD_HIDDEN; ++i) t->ctx[i] *= rescale;
    t->sum_exp *= rescale;
    t->max_score = score;
  }
  const float w = expf(score - t->max_score);
  for (int i = 0; i < BIRD_HIDDEN; ++i) t->ctx[i] += w * p[i];
  t->sum_exp += w;
}

void BirdTail_Finish(const BirdTail* t, float* logits) {
  float ctx[BIRD_HIDDEN];
  const float inv = 1.0f / t->sum_exp;
  for (int i = 0; i < BIRD_HIDDEN; ++i) ctx[i] = t->ctx[i] * inv;
  MATVEC(logits, BIRD_CLASSES, BIRD_HIDDEN, k_w_f, k_w_f_scale, ctx);
  for (int i = 0; i < BIRD_CLASSES; ++i) logits[i] += k_b_f[i];
}
