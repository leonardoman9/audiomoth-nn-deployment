#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "schedule.h"
/* Reads cases from stdin: "parse <text>" | "fmt <secs>" | "win <secs>" | "log <secs> <9 logits>" and prints results. */
int main(void) {
  char line[256];
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
      float lg[9];
      for (int i = 0; i < 9; ++i) lg[i] = strtof(p, &p);
      char out[SCHED_LOG_LINE_MAX];
      uint32_t n = Sched_FormatLogLine(out, s, lg);
      if (n >= SCHED_LOG_LINE_MAX) { printf("TOO LONG %u\n", n); continue; }
      out[n] = 0;
      fputs(out, stdout);
    }
  }
  return 0;
}
