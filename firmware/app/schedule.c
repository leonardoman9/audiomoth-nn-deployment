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

static uint32_t PutUint(char* out, uint32_t v) {
  char tmp[10];
  uint32_t d = 0, n = 0;
  do { tmp[d++] = (char)('0' + v % 10); v /= 10; } while (v > 0);
  while (d > 0) out[n++] = tmp[--d];
  return n;
}

void Sched_WavPath(uint32_t seconds, char* out) {
  char t[20];
  Sched_FormatTime(seconds, t);  /* "YYYY-MM-DD hh:mm:ss" */
  out[0] = t[0]; out[1] = t[1]; out[2] = t[2]; out[3] = t[3]; out[4] = t[5]; out[5] = t[6]; out[6] = t[8]; out[7] = t[9];
  out[8] = '/';
  out[9] = t[11]; out[10] = t[12]; out[11] = t[14]; out[12] = t[15]; out[13] = t[17]; out[14] = t[18];
  out[15] = '.'; out[16] = 'W'; out[17] = 'A'; out[18] = 'V'; out[19] = 0;
}

static void PutLE(uint8_t* p, uint32_t v, int bytes) {
  for (int i = 0; i < bytes; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

void Sched_WavHeader(uint8_t* out, uint32_t samples) {
  static const uint8_t kFmt[36] = {'R', 'I', 'F', 'F', 0, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ',
                                   16, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 16, 0};
  for (uint32_t i = 0; i < 36; ++i) out[i] = kFmt[i];
  const uint32_t junk = SCHED_WAV_HEADER - 36u - 8u - 8u;  /* JUNK payload so that "data" ends at the header end */
  out[36] = 'J'; out[37] = 'U'; out[38] = 'N'; out[39] = 'K';
  PutLE(out + 40, junk, 4);
  for (uint32_t i = 0; i < junk; ++i) out[44 + i] = 0;
  uint8_t* data = out + SCHED_WAV_HEADER - 8u;
  data[0] = 'd'; data[1] = 'a'; data[2] = 't'; data[3] = 'a';
  const uint32_t bytes = samples * 2u;
  PutLE(data + 4, bytes, 4);
  PutLE(out + 4, SCHED_WAV_HEADER - 8u + bytes, 4);
  PutLE(out + 24, SCHED_SAMPLE_RATE, 4);
  PutLE(out + 28, SCHED_SAMPLE_RATE * 2u, 4);
}

uint32_t Sched_WindowCount(uint32_t samples) {
  return samples < SCHED_WINDOW_SAMPLES ? 0 : (samples - SCHED_WINDOW_SAMPLES) / SCHED_HOP_SAMPLES + 1;
}

uint32_t Sched_FormatState(const SchedState* st, char* out) {
  uint32_t n = 0;
  if (st->kind == SCHED_STATE_DONE) {
    for (const char* p = "DONE"; *p;) out[n++] = *p++;
  } else {
    out[n++] = st->kind == SCHED_STATE_REC ? 'R' : 'C';
    out[n++] = st->kind == SCHED_STATE_REC ? 'E' : 'L';
    out[n++] = st->kind == SCHED_STATE_REC ? 'C' : 'S';
    out[n++] = ' ';
    for (const char* p = st->path; *p;) out[n++] = *p++;
    out[n++] = ' ';
    n += PutUint(out + n, st->start);
    if (st->kind == SCHED_STATE_CLS) {
      out[n++] = ' '; n += PutUint(out + n, st->samples);
      out[n++] = ' '; n += PutUint(out + n, st->next);
      out[n++] = ' '; n += PutUint(out + n, st->gaps);
      out[n++] = ' '; n += PutUint(out + n, st->tries);
    }
  }
  out[n++] = '\n';
  while (n < SCHED_STATE_RECORD) out[n++] = ' ';
  return n;
}

static bool ReadUint(const char* s, uint32_t length, uint32_t* pos, uint32_t* value) {
  if (*pos >= length || s[*pos] < '0' || s[*pos] > '9') return false;
  uint64_t v = 0;
  while (*pos < length && s[*pos] >= '0' && s[*pos] <= '9') {
    v = v * 10u + (uint64_t)(s[*pos] - '0');
    if (v > 0xFFFFFFFFu) return false;
    ++*pos;
  }
  *value = (uint32_t)v;
  return true;
}

bool Sched_ParseState(const char* text, uint32_t length, SchedState* st) {
  uint32_t pos = 0;
  if (length >= 4 && text[0] == 'D' && text[1] == 'O' && text[2] == 'N' && text[3] == 'E') {
    st->kind = SCHED_STATE_DONE;
    return true;
  }
  if (length < 4 || text[3] != ' ') return false;
  if (text[0] == 'R' && text[1] == 'E' && text[2] == 'C') st->kind = SCHED_STATE_REC;
  else if (text[0] == 'C' && text[1] == 'L' && text[2] == 'S') st->kind = SCHED_STATE_CLS;
  else return false;
  pos = 4;
  /* path: exactly "YYYYMMDD/HHMMSS.WAV" */
  if (pos + SCHED_PATH_MAX - 1 > length) return false;
  for (uint32_t i = 0; i < SCHED_PATH_MAX - 1; ++i) {
    const char c = text[pos + i];
    const bool digit = c >= '0' && c <= '9';
    const bool ok = i == 8 ? c == '/' : (i == 15 ? c == '.' : (i > 15 ? (c == "WAV"[i - 16]) : digit));
    if (!ok) return false;
    st->path[i] = c;
  }
  st->path[SCHED_PATH_MAX - 1] = 0;
  pos += SCHED_PATH_MAX - 1;
  if (!Expect(text, length, &pos, ' ') || !ReadUint(text, length, &pos, &st->start)) return false;
  st->samples = st->next = st->gaps = st->tries = 0;
  if (st->kind == SCHED_STATE_CLS) {
    if (!Expect(text, length, &pos, ' ') || !ReadUint(text, length, &pos, &st->samples) ||
        !Expect(text, length, &pos, ' ') || !ReadUint(text, length, &pos, &st->next) ||
        !Expect(text, length, &pos, ' ') || !ReadUint(text, length, &pos, &st->gaps) ||
        !Expect(text, length, &pos, ' ') || !ReadUint(text, length, &pos, &st->tries))
      return false;
    if (st->next > Sched_WindowCount(st->samples) || st->gaps > st->samples) return false;
  }
  return pos == length || text[pos] == '\n' || text[pos] == '\r' || text[pos] == 0;
}

uint32_t Sched_FormatSession(char* out, const SchedState* st, uint32_t windows, uint32_t classify_seconds) {
  Sched_FormatTime(st->start, out);
  uint32_t n = 19;
  out[n++] = ',';
  for (const char* p = st->path; *p;) out[n++] = *p++;
  out[n++] = ','; n += PutUint(out + n, st->samples / SCHED_SAMPLE_RATE);
  out[n++] = ','; n += PutUint(out + n, st->gaps);
  out[n++] = ','; n += PutUint(out + n, windows);
  out[n++] = ','; n += PutUint(out + n, classify_seconds);
  out[n++] = '\n';
  return n;
}
