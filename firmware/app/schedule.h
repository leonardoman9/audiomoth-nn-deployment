// Pure logic of the bird-logger application (no hardware): daily 07:00 window, CONFIG.TXT
// parsing, time formatting and the log line. No libc time/printf functions on purpose
// (they cost ~30 KB of flash on this chip).
#ifndef SCHEDULE_H
#define SCHEDULE_H
#include <stdbool.h>
#include <stdint.h>

#define SCHED_WINDOW_START_S  (7u * 3600u)  /* 07:00 on the device clock */
#define SCHED_WINDOW_LENGTH_S 300u          /* 5 minutes */
#define SCHED_DAY_S           86400u
#define SCHED_CLASSES         9
#define SCHED_LOG_LINE_MAX    128
#define SCHED_CONFIDENCE_PCT  50            /* a top class below this is reported as "uncertain" */

#ifdef __cplusplus
extern "C" {
#endif

/* "YYYY-MM-DD hh:mm:ss" (the clock value to set) -> seconds since 1970-01-01 00:00:00 of that
 * clock. Leading/trailing blanks and line ends are ignored; no time zone handling: the clock
 * simply holds the local time that was written in CONFIG.TXT. */
bool Sched_ParseConfig(const char* text, uint32_t length, uint32_t* seconds);
/* Writes "YYYY-MM-DD hh:mm:ss" and a terminating 0 (20 bytes). */
void Sched_FormatTime(uint32_t seconds, char* out);

bool Sched_InWindow(uint32_t seconds);
/* Seconds until the window starts (0 if inside it). */
uint32_t Sched_SecondsToNextWindow(uint32_t seconds);
/* Seconds until the window ends (0 if outside it). */
uint32_t Sched_SecondsLeftInWindow(uint32_t seconds);

/* Class names in the checkpoint's output order (index 8 = non-bird). */
const char* Sched_ClassName(int index);
/* Index of the largest logit. */
int Sched_TopClass(const float* logits);
/* Probability (percent, rounded) of the largest logit under softmax. */
uint32_t Sched_TopPercent(const float* logits);
/* One log line: "<time>,<top class>,<percent>,<Y|N>,<9 logits x100>\n", Y = confident (>= SCHED_CONFIDENCE_PCT);
 * the logits are written as integers (hundredths) so the line can be checked against PyTorch.
 * Returns the number of bytes written (<= SCHED_LOG_LINE_MAX); out must hold SCHED_LOG_LINE_MAX bytes. */
uint32_t Sched_FormatLogLine(char* out, uint32_t seconds, const float* logits);

#ifdef __cplusplus
}
#endif
#endif
