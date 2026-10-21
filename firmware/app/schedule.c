#include "schedule.h"

#include <math.h>

/* In the model's output index order. For the default bird model, why this order and not the one in the training log:
 * firmware/bird/model/README.md. */
static const char* const kClassNames[SCHED_CLASSES] = {MODEL_CLASS_NAMES};

const char* Sched_ClassName(int index) { return index >= 0 && index < SCHED_CLASSES ? kClassNames[index] : "?"; }

/* Howard Hinnant's civil <-> days algorithms (days since 1970-01-01). */
static int32_t DaysFromCivil(int32_t y, int32_t m, int32_t d) {
  y -= m <= 2;
  const int32_t era = (y >= 0 ? y : y - 399) / 400;
  const int32_t yoe = y - era * 400;
  const int32_t doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
  const int32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + doe - 719468;
}

static void CivilFromDays(int32_t z, int32_t* y, int32_t* m, int32_t* d) {
  z += 719468;
  const int32_t era = (z >= 0 ? z : z - 146096) / 146097;
  const int32_t doe = z - era * 146097;
  const int32_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int32_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const int32_t mp = (5 * doy + 2) / 153;
  *d = doy - (153 * mp + 2) / 5 + 1;
  *m = mp + (mp < 10 ? 3 : -9);
  *y = yoe + era * 400 + (*m <= 2);
}

static bool ReadNumber(const char* s, uint32_t length, uint32_t* pos, uint32_t digits, int32_t* value) {
  if (*pos + digits > length) return false;
  int32_t v = 0;
  for (uint32_t i = 0; i < digits; ++i) {
    const char c = s[*pos + i];
    if (c < '0' || c > '9') return false;
    v = v * 10 + (c - '0');
  }
  *pos += digits;
  *value = v;
  return true;
}

static bool Expect(const char* s, uint32_t length, uint32_t* pos, char c) {
  if (*pos >= length || s[*pos] != c) return false;
  ++*pos;
  return true;
}

bool Sched_ParseConfig(const char* text, uint32_t length, uint32_t* seconds) {
  uint32_t pos = 0;
  while (pos < length && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\r' || text[pos] == '\n')) ++pos;
  int32_t Y, M, D, h, m, s;
  if (!ReadNumber(text, length, &pos, 4, &Y) || !Expect(text, length, &pos, '-') ||
      !ReadNumber(text, length, &pos, 2, &M) || !Expect(text, length, &pos, '-') ||
      !ReadNumber(text, length, &pos, 2, &D) || !Expect(text, length, &pos, ' ') ||
      !ReadNumber(text, length, &pos, 2, &h) || !Expect(text, length, &pos, ':') ||
      !ReadNumber(text, length, &pos, 2, &m) || !Expect(text, length, &pos, ':') ||
      !ReadNumber(text, length, &pos, 2, &s))
    return false;
  /* Only whitespace may follow: "06:58:309" is a typo, not 06:58:30. */
  if (pos < length && text[pos] != ' ' && text[pos] != '\t' && text[pos] != '\r' && text[pos] != '\n' && text[pos] != '\0') return false;
  if (Y < 2020 || Y > 2100 || M < 1 || M > 12 || D < 1 || D > 31 || h > 23 || m > 59 || s > 59) return false;
  const int32_t days = DaysFromCivil(Y, M, D);
  int32_t cy, cm, cd;
  CivilFromDays(days, &cy, &cm, &cd);
  if (cy != Y || cm != M || cd != D) return false;  /* 2026-02-30 and the like */
  *seconds = (uint32_t)days * SCHED_DAY_S + (uint32_t)h * 3600u + (uint32_t)m * 60u + (uint32_t)s;
  return true;
}

static void Put2(char* out, uint32_t v) {
  out[0] = (char)('0' + (v / 10) % 10);
  out[1] = (char)('0' + v % 10);
}

void Sched_FormatTime(uint32_t seconds, char* out) {
  int32_t y, mo, d;
  CivilFromDays((int32_t)(seconds / SCHED_DAY_S), &y, &mo, &d);
  const uint32_t t = seconds % SCHED_DAY_S;
  out[0] = (char)('0' + (y / 1000) % 10);
  out[1] = (char)('0' + (y / 100) % 10);
  Put2(out + 2, (uint32_t)(y % 100));
  out[4] = '-';
  Put2(out + 5, (uint32_t)mo);
  out[7] = '-';
  Put2(out + 8, (uint32_t)d);
  out[10] = ' ';
  Put2(out + 11, t / 3600u);
  out[13] = ':';
  Put2(out + 14, (t / 60u) % 60u);
  out[16] = ':';
  Put2(out + 17, t % 60u);
  out[19] = 0;
}

bool Sched_InWindow(uint32_t seconds) {
  const uint32_t t = seconds % SCHED_DAY_S;
  return t >= SCHED_WINDOW_START_S && t < SCHED_WINDOW_START_S + SCHED_WINDOW_LENGTH_S;
}

uint32_t Sched_SecondsToNextWindow(uint32_t seconds) {
  const uint32_t t = seconds % SCHED_DAY_S;
  if (Sched_InWindow(seconds)) return 0;
  return t < SCHED_WINDOW_START_S ? SCHED_WINDOW_START_S - t : SCHED_DAY_S - t + SCHED_WINDOW_START_S;
}

uint32_t Sched_SecondsLeftInWindow(uint32_t seconds) {
  if (!Sched_InWindow(seconds)) return 0;
  return SCHED_WINDOW_START_S + SCHED_WINDOW_LENGTH_S - seconds % SCHED_DAY_S;
}

int Sched_TopClass(const float* logits) {
  int best = 0;
  for (int i = 1; i < SCHED_CLASSES; ++i)
    if (logits[i] > logits[best]) best = i;
  return best;
}

uint32_t Sched_TopPercent(const float* logits) {
  const int best = Sched_TopClass(logits);
  float sum = 0.0f;
  for (int i = 0; i < SCHED_CLASSES; ++i) sum += expf(logits[i] - logits[best]);
  const float pct = 100.0f / sum + 0.5f;
  return pct >= 0.0f && pct < 101.0f ? (uint32_t)pct : 0;  /* NaN/inf logits must not reach the cast */
}

static uint32_t PutInt(char* out, int32_t v) {
  uint32_t n = 0;
  uint32_t m = (uint32_t)v;
  if (v < 0) { out[n++] = '-'; m = 0u - m; }
  char tmp[10];
  int d = 0;
  do { tmp[d++] = (char)('0' + m % 10); m /= 10; } while (m > 0);
  while (d > 0) out[n++] = tmp[--d];
  return n;
}

uint32_t Sched_FormatLogLine(char* out, uint32_t seconds, const float* logits) {
  const int best = Sched_TopClass(logits);
  const uint32_t pct = Sched_TopPercent(logits);
  Sched_FormatTime(seconds, out);
  uint32_t n = 19;
  out[n++] = ',';
  for (const char* p = kClassNames[best]; *p;) out[n++] = *p++;
  out[n++] = ',';
  n += PutInt(out + n, (int32_t)pct);
  out[n++] = ',';
  out[n++] = pct >= SCHED_CONFIDENCE_PCT ? 'Y' : 'N';
  for (int i = 0; i < SCHED_CLASSES; ++i) {
    out[n++] = ',';
    /* clamped so a broken logit (NaN, inf, huge) cannot overflow the line: ",-999999" per class is in SCHED_LOG_LINE_MAX */
    float v = logits[i] * 100.0f;
    v = v == v ? (v > 999999.0f ? 999999.0f : (v < -999999.0f ? -999999.0f : v)) : 0.0f;
    n += PutInt(out + n, (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f)));
  }
  out[n++] = '\n';
  return n;
}
