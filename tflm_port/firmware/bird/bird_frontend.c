#include "bird_frontend.h"

#include <math.h>
#include <string.h>

#include "bird_frontend_tables.h"

/* FFT working set: internal RAM, 6 KB. */
static float s_re[BFE_NFFT / 2], s_im[BFE_NFFT / 2], s_mag[BFE_BINS];

void BirdFE_Reset(BirdFrontEnd* fe) {
  fe->samples = 0;
  fe->frames = 0;
}

void BirdFE_Init(BirdFrontEnd* fe) { BirdFE_Reset(fe); }

/* Sample of the zero-based clip at index i, with torch's reflect padding outside [0, n). */
static inline int16_t PaddedSample(const BirdFrontEnd* fe, int32_t i, int32_t n_total) {
  if (i < 0) i = -i;
  if (i >= n_total) i = 2 * (n_total - 1) - i;
  return fe->ring[(uint32_t)i & (BFE_RING - 1)];
}

/* Periodic Hann window from the cosine table (kFeCos has 512 entries: index 512 is the centre, w = 1). */
static inline float HannAt(int i) {
  if (i == BFE_NFFT / 2) return 1.0f;
  return 0.5f - 0.5f * kFeCos[i < BFE_NFFT / 2 ? i : BFE_NFFT - i];
}

/* In-place complex FFT of 512 points on s_re/s_im, input already in bit-reversed order. */
static void Fft512(void) {
  for (int len = 2; len <= BFE_NFFT / 2; len <<= 1) {
    const int half = len >> 1, stride = BFE_NFFT / len;  /* W_len^k = W_1024^(k*stride) */
    for (int base = 0; base < BFE_NFFT / 2; base += len) {
      for (int k = 0; k < half; ++k) {
        const float wr = kFeCos[k * stride], wi = kFeSin[k * stride];
        const int a = base + k, b = a + half;
        const float tr = s_re[b] * wr - s_im[b] * wi, ti = s_re[b] * wi + s_im[b] * wr;
        s_re[b] = s_re[a] - tr; s_im[b] = s_im[a] - ti;
        s_re[a] += tr; s_im[a] += ti;
      }
    }
  }
}

static void ComputeFrame(BirdFrontEnd* fe, uint32_t t, int32_t n_total) {
  const int32_t start = (int32_t)(t * BFE_HOP) - BFE_NFFT / 2;
  /* z[n] = w[2n] x[2n] + j w[2n+1] x[2n+1], stored directly in bit-reversed order. */
  for (int n = 0; n < BFE_NFFT / 2; ++n) {
    const int i0 = 2 * n, i1 = 2 * n + 1;
    const float w0 = HannAt(i0), w1 = HannAt(i1);
    const int j = kFeBitRev512[n];
    s_re[j] = (float)PaddedSample(fe, start + i0, n_total) * (1.0f / 32768.0f) * w0;
    s_im[j] = (float)PaddedSample(fe, start + i1, n_total) * (1.0f / 32768.0f) * w1;
  }
  Fft512();
  /* Split: X[k] = E[k] + W_1024^k O[k], E = (Z[k]+conj Z[N-k])/2, O = (Z[k]-conj Z[N-k])/(2j). */
  for (int k = 0; k <= BFE_NFFT / 2; ++k) {
    const int ka = k & (BFE_NFFT / 2 - 1), kb = (BFE_NFFT / 2 - k) & (BFE_NFFT / 2 - 1);
    const float zr = s_re[ka], zi = s_im[ka], cr = s_re[kb], ci = -s_im[kb];  /* conj Z[N-k] */
    const float er = 0.5f * (zr + cr), ei = 0.5f * (zi + ci);
    const float dr = 0.5f * (zr - cr), di = 0.5f * (zi - ci);
    const float orr = di, oi = -dr;                                            /* (d)/(j) = -j d */
    float xr, xi;
    if (k < BFE_NFFT / 2) {
      const float wr = kFeCos[k], wi = kFeSin[k];
      xr = er + wr * orr - wi * oi;
      xi = ei + wr * oi + wi * orr;
    } else {                                                                    /* W^512 = -1 */
      xr = er - orr;
      xi = ei - oi;
    }
    s_mag[k] = sqrtf(xr * xr + xi * xi);
  }
  for (int m = 0; m < BFE_MELS; ++m) {
    const float* w = kFeFbWeight + kFeFbOffset[m];
    const float* x = s_mag + kFeFbStart[m];
    float acc = 0.0f;
    for (int j = 0; j < kFeFbLen[m]; ++j) acc += w[j] * x[j];
    float p = acc * acc;
    if (p < 1e-10f) p = 1e-10f;
    fe->spec_db[t][m] = (int16_t)lrintf(10.0f * log10f(p) * 256.0f);
  }
}

void BirdFE_Push(BirdFrontEnd* fe, const int16_t* pcm, uint32_t n) {
  /* Blocks of at most 256 samples keep the ring from wrapping over samples a pending frame still needs. */
  while (n > 0 && fe->samples < BFE_CLIP_SAMPLES) {
    uint32_t chunk = n < 256 ? n : 256;
    if (fe->samples + chunk > BFE_CLIP_SAMPLES) chunk = BFE_CLIP_SAMPLES - fe->samples;
    for (uint32_t i = 0; i < chunk; ++i) fe->ring[fe->samples++ & (BFE_RING - 1)] = pcm[i];
    pcm += chunk;
    n -= chunk;
    /* Frame t needs samples up to t*hop + 511 (no end padding): compute it as soon as present. */
    while (fe->frames < BFE_FRAMES) {
      const uint32_t last_needed = fe->frames * BFE_HOP + BFE_NFFT / 2 - 1;
      if (last_needed >= fe->samples) break;
      if (fe->frames == 0 && fe->samples < BFE_NFFT / 2 + 1) break; /* start reflection needs x[1..512] */
      ComputeFrame(fe, fe->frames, (int32_t)BFE_CLIP_SAMPLES);
      ++fe->frames;
    }
  }
}

bool BirdFE_Complete(const BirdFrontEnd* fe) { return fe->samples >= BFE_CLIP_SAMPLES; }

static bool FinishDb(BirdFrontEnd* fe, float* mean, float* inv_std, int16_t* floor_db) {
  if (fe->samples < BFE_CLIP_SAMPLES) return false;
  /* The last frames need samples beyond the end: reflect them. */
  while (fe->frames < BFE_FRAMES) {
    ComputeFrame(fe, fe->frames, (int32_t)BFE_CLIP_SAMPLES);
    ++fe->frames;
  }
  int16_t max_db = INT16_MIN;
  for (int t = 0; t < BFE_FRAMES; ++t)
    for (int m = 0; m < BFE_MELS; ++m) if (fe->spec_db[t][m] > max_db) max_db = fe->spec_db[t][m];
  *floor_db = (int16_t)(max_db - 80 * 256); /* top_db = 80 */
  /* Mean of the (clamped) dB values: the sum of the int16 codes is exact in 64 bits. */
  int64_t sum = 0;
  for (int t = 0; t < BFE_FRAMES; ++t)
    for (int m = 0; m < BFE_MELS; ++m) {
      int16_t v = fe->spec_db[t][m];
      if (v < *floor_db) { v = *floor_db; fe->spec_db[t][m] = v; }
      sum += v;
    }
  const float count = (float)BFE_FRAMES * (float)BFE_MELS;
  const float mu = (float)sum / (256.0f * count);
  /* Second pass for the variance (no catastrophic cancellation, no double precision needed). */
  float acc = 0.0f;
  for (int t = 0; t < BFE_FRAMES; ++t)
    for (int m = 0; m < BFE_MELS; ++m) {
      const float d = (float)fe->spec_db[t][m] / 256.0f - mu;
      acc += d * d;
    }
  const float var = acc / (count - 1.0f); /* unbiased, as torch.std */
  *mean = mu;
  *inv_std = 1.0f / (sqrtf(var) + 1e-5f);
  return true;
}

bool BirdFE_FinishFloat(BirdFrontEnd* fe, float* out) {
  float mean, inv_std; int16_t floor_db;
  if (!FinishDb(fe, &mean, &inv_std, &floor_db)) return false;
  for (int t = 0; t < BFE_FRAMES; ++t)
    for (int m = 0; m < BFE_MELS; ++m) out[t * BFE_MELS + m] = ((float)fe->spec_db[t][m] / 256.0f - mean) * inv_std;
  return true;
}

bool BirdFE_Finish(BirdFrontEnd* fe, int8_t* out, float scale, int zero_point) {
  float mean, inv_std; int16_t floor_db;
  if (!FinishDb(fe, &mean, &inv_std, &floor_db)) return false;
  const float inv_scale = 1.0f / scale;
  for (int t = 0; t < BFE_FRAMES; ++t)
    for (int m = 0; m < BFE_MELS; ++m) {
      const float v = ((float)fe->spec_db[t][m] / 256.0f - mean) * inv_std;
      long q = lrintf(v * inv_scale) + zero_point;
      out[t * BFE_MELS + m] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
  return true;
}

uint32_t BirdFE_SizeOf(void) { return (uint32_t)sizeof(BirdFrontEnd); }
