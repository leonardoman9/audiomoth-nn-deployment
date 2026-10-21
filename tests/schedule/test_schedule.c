#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "schedule.h"
/* Reads cases from stdin and prints results: "parse <text>" | "fmt <secs>" | "win <secs>" | "log <secs> <SCHED_CLASSES logits>" |
 * "path <secs>" | "wav <samples>" (header as hex) | "count <samples>" | "state <text>" (parse, then format again) |
 * "session <start> <samples> <gaps> <windows> <classify s>". */
int main(void) {
  char line[16 * SCHED_CLASSES + 256];  /* "log <secs>" + one "%f" per class */
  while (fgets(line, sizeof line, stdin)) {
    if (!strncmp(line, "parse ", 6)) {
      uint32_t s;
      const char* t = line + 6;
      printf(Sched_ParseConfig(t, (uint32_t)strlen(t), &s) ? "%u\n" : "ERR\n", s);
    } else if (!strncmp(line, "fmt ", 4)) {
      char out[20];
      Sched_FormatTime((uint32_t)strtoul(line + 4, 0, 10), out);
      printf("%s\n", out);
    } else if (!strncmp(line, "win ", 4)) {
      uint32_t s = (uint32_t)strtoul(line + 4, 0, 10);
      printf("%d %u %u\n", Sched_InWindow(s), Sched_SecondsToNextWindow(s), Sched_SecondsLeftInWindow(s));
    } else if (!strncmp(line, "log ", 4)) {
      char* p = line + 4;
      uint32_t s = (uint32_t)strtoul(p, &p, 10);
      float lg[SCHED_CLASSES];
      for (int i = 0; i < SCHED_CLASSES; ++i) lg[i] = strtof(p, &p);
      char out[SCHED_LOG_LINE_MAX];
      uint32_t n = Sched_FormatLogLine(out, s, lg);
      if (n >= SCHED_LOG_LINE_MAX) { printf("TOO LONG %u\n", n); continue; }
      out[n] = 0;
      fputs(out, stdout);
    } else if (!strncmp(line, "path ", 5)) {
      char out[SCHED_PATH_MAX];
      Sched_WavPath((uint32_t)strtoul(line + 5, 0, 10), out);
      printf("%s\n", out);
    } else if (!strncmp(line, "wav ", 4)) {
      uint8_t h[SCHED_WAV_HEADER];
      Sched_WavHeader(h, (uint32_t)strtoul(line + 4, 0, 10));
      for (uint32_t i = 0; i < SCHED_WAV_HEADER; ++i) printf("%02x", h[i]);
      printf("\n");
    } else if (!strncmp(line, "count ", 6)) {
      printf("%u\n", Sched_WindowCount((uint32_t)strtoul(line + 6, 0, 10)));
    } else if (!strncmp(line, "state ", 6)) {
      const char* t = line + 6;
      SchedState st;
      memset(&st, 0x55, sizeof st);
      if (!Sched_ParseState(t, (uint32_t)strlen(t), &st)) { printf("ERR\n"); continue; }
      char out[SCHED_STATE_RECORD + 1];
      const uint32_t n = Sched_FormatState(&st, out);
      out[n] = 0;
      /* the record is always SCHED_STATE_RECORD bytes: the line, '\n', spaces */
      const char* nl = strchr(out, '\n');
      bool padded = n == SCHED_STATE_RECORD && nl != NULL;
      for (const char* q = nl ? nl + 1 : out; padded && *q; ++q) padded = *q == ' ';
      if (!padded) { printf("BAD RECORD %u\n", n); continue; }
      fwrite(out, 1, (size_t)(nl - out + 1), stdout);
    } else if (!strncmp(line, "session ", 8)) {
      char* p = line + 8;
      SchedState st = {SCHED_STATE_CLS, "20261021/070000.WAV", 0, 0, 0, 0, 0};
      st.start = (uint32_t)strtoul(p, &p, 10);
      Sched_WavPath(st.start, st.path);
      st.samples = (uint32_t)strtoul(p, &p, 10);
      st.gaps = (uint32_t)strtoul(p, &p, 10);
      const uint32_t windows = (uint32_t)strtoul(p, &p, 10), seconds = (uint32_t)strtoul(p, &p, 10);
      char out[SCHED_SESSION_LINE_MAX + 1];
      const uint32_t n = Sched_FormatSession(out, &st, windows, seconds);
      out[n] = 0;
      fputs(out, stdout);
    }
  }
  return 0;
}
