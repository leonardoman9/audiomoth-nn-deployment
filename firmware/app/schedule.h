// Pure logic of the logger application (no hardware): daily classification window, CONFIG.TXT
// parsing, time formatting and the log line. Classes and window come from the model package
// (model_config.h, written by firmware/bird/gen_model_config.py). No libc time/printf functions on purpose
// (they cost ~30 KB of flash on this chip).
#ifndef SCHEDULE_H
#define SCHEDULE_H
#include <stdbool.h>
#include <stdint.h>

#include "model_config.h"

#define SCHED_WINDOW_START_S  APP_WINDOW_START_S   /* on the device clock (default 07:00) */
#define SCHED_WINDOW_LENGTH_S APP_WINDOW_LENGTH_S  /* default 5 minutes */
#define SCHED_DAY_S           86400u
#define SCHED_CLASSES         MODEL_CLASSES
#define SCHED_CONFIDENCE_PCT  APP_CONFIDENCE_PCT   /* a top class below this is reported as "uncertain" */
/* time (19) + 4 commas + name + percent (3) + Y/N + newline = 28 + name, then ",-999999" (8) per class; plus margin */
#define SCHED_LOG_LINE_MAX    (32 + MODEL_CLASS_NAME_MAX + 8 * MODEL_CLASSES)

/* Recording: the window is recorded to the SD card, then classified in 3 s windows every SCHED_HOP_SAMPLES. */
#define SCHED_SAMPLE_RATE     32000u
#define SCHED_WINDOW_SAMPLES  96000u               /* 3 s: the model's input */
#ifndef APP_WINDOW_HOP_SAMPLES
#define APP_WINDOW_HOP_SAMPLES SCHED_WINDOW_SAMPLES /* packages made before the hop setting: no overlap */
#endif
#define SCHED_HOP_SAMPLES     APP_WINDOW_HOP_SAMPLES
#define SCHED_WAV_HEADER      512u                 /* RIFF + fmt + JUNK padding: the samples start on a sector */
#define SCHED_PATH_MAX        20u                  /* "YYYYMMDD/HHMMSS.WAV" + 0 */
#define SCHED_STATE_RECORD    96u                  /* STATE.TXT is always this long (the longest line is 90): rewritten in place, never truncated */
#define SCHED_SESSION_LINE_MAX 96u

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

/* Class names in the model's output order (MODEL_NEGATIVE_CLASS = "nothing of interest"). */
const char* Sched_ClassName(int index);
/* Index of the largest logit. */
int Sched_TopClass(const float* logits);
/* Probability (percent, rounded) of the largest logit under softmax. */
uint32_t Sched_TopPercent(const float* logits);
/* One log line: "<time>,<top class>,<percent>,<Y|N>,<SCHED_CLASSES logits x100>\n", Y = confident (>= SCHED_CONFIDENCE_PCT);
 * the logits are written as integers (hundredths) so the line can be checked against PyTorch.
 * Returns the number of bytes written (<= SCHED_LOG_LINE_MAX); out must hold SCHED_LOG_LINE_MAX bytes. */
uint32_t Sched_FormatLogLine(char* out, uint32_t seconds, const float* logits);

/* Recording file of a session started at `seconds`: "YYYYMMDD/HHMMSS.WAV" (8.3 names, one folder per day). The folder
 * name is the first 8 characters. Writes SCHED_PATH_MAX bytes including the terminating 0. */
void Sched_WavPath(uint32_t seconds, char* out);
/* SCHED_WAV_HEADER-byte header of a 32 kHz mono 16-bit WAV holding `samples` samples: RIFF, fmt and a JUNK chunk
 * that pads the header to one 512-byte sector, so the 4 KB writes of the recording are sector-aligned. */
void Sched_WavHeader(uint8_t* out, uint32_t samples);
/* Number of 3 s windows (step SCHED_HOP_SAMPLES) in a recording of `samples` samples. */
uint32_t Sched_WindowCount(uint32_t samples);

/* Progress of the current recording, kept in STATE.TXT so a reset or power cut resumes where it stopped:
 *   "REC <path> <start>"                          recording in progress (its length is the file's)
 *   "CLS <path> <start> <samples> <next> <gaps> <tries>"  recorded; windows from <next> still to classify; <tries> =
 *                                                  boots that started window <next> without finishing it
 *   "DONE"                                        nothing pending */
typedef enum { SCHED_STATE_DONE, SCHED_STATE_REC, SCHED_STATE_CLS } SchedStateKind;
typedef struct {
  SchedStateKind kind;
  char path[SCHED_PATH_MAX];
  uint32_t start;    /* clock at the first recorded sample (the recording starts on a whole second) */
  uint32_t samples;  /* recorded samples (CLS) */
  uint32_t next;     /* next window to classify (CLS) */
  uint32_t gaps;     /* samples lost to SD stalls and written as silence (CLS) */
  uint32_t tries;    /* resumes that started window <next> (CLS): a window that keeps crashing is skipped after 3 */
} SchedState;
/* Writes exactly SCHED_STATE_RECORD bytes: the line, '\n', then spaces (ignored when parsing). */
uint32_t Sched_FormatState(const SchedState* st, char* out);
bool Sched_ParseState(const char* text, uint32_t length, SchedState* st);
/* One SESSIONS.TXT line: "<start time>,<path>,<seconds recorded>,<gap samples>,<windows>,<classification seconds>\n". */
uint32_t Sched_FormatSession(char* out, const SchedState* st, uint32_t windows, uint32_t classify_seconds);

#ifdef __cplusplus
}
#endif
#endif
