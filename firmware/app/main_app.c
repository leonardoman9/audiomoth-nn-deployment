/*
 * WrenNet logger for the AudioMoth.
 *
 * Every day in the window of the model package (default 07:00 for 5 minutes, device clock): record the whole window
 * at 32 kHz to the SD card (YYYYMMDD/HHMMSS.WAV), then classify the recording in 3 s windows with the on-device model
 * and append one line per window to LOG.TXT ("2026-10-21 07:00:12,Apus apus,83,Y,..."). Every second of the window
 * is heard, and the audio stays on the card to check the results. Outside the window the device stays powered down.
 *
 * Progress is kept in STATE.TXT. A reset or a power cut during the recording or the classification is resumed at the
 * next boot, also when the clock was lost with it (any reset other than an EM4 wake-up clears it): the times needed
 * are in STATE.TXT. SESSIONS.TXT gets one summary line per session.
 *
 * The clock is set from CONFIG.TXT on the SD card (one line "YYYY-MM-DD hh:mm:ss"); the file is
 * overwritten with 'APPLIED <time>' once applied. No USB, GPS or configuration app.
 *
 * Hardware calls follow the original AudioMoth firmware (main.c: makeRecording / DMA handler).
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "audiomoth.h"
#include "bird_extsram.h"
#include "bird_pipeline.h"
#include "digitalfilter.h"
#include "schedule.h"

/* ADC at 384 kHz decimated by 12 gives the 32 kHz the model was trained at. */
#define RAW_SAMPLE_RATE        384000u
#define SAMPLE_RATE_DIVIDER    12u
#define OUTPUT_SAMPLE_RATE     (RAW_SAMPLE_RATE / SAMPLE_RATE_DIVIDER)
#define SAMPLES_PER_TRANSFER   64u                                   /* decimated samples per DMA transfer */
#define RAW_PER_TRANSFER       (SAMPLES_PER_TRANSFER * SAMPLE_RATE_DIVIDER) /* 768 <= 1024 */
#define SETTLE_TRANSFERS       100u                                  /* ~200 ms: microphone and filter settling */
#define DC_BLOCKING_FREQ_HZ    48u
#define MAX_SLEEP_CHUNK_S      (6u * 3600u)

/* Recording: the DMA handler fills a ring in the external SRAM; the main loop copies CHUNK samples at a time to an
 * internal buffer and writes them to the card. The ring absorbs writes that are slow for up to ~1.5 s in total. If
 * the card falls further behind, or a write fails (the SD driver gives up on a card busy for more than 500 ms), the
 * file is reopened and the lost stretch written as silence and counted, so the file keeps the real timeline. On a
 * full card the recording simply ends there. */
#define RING_SAMPLES           49152u                                /* 1.54 s, 96 KB of external SRAM */
#define CHUNK_SAMPLES          2048u                                 /* 64 ms, one 4 KB SD write; divides RING_SAMPLES */
#define SYNC_EVERY_CHUNKS      40u                                   /* ~2.6 s: what a power cut can lose */
#define MAX_WRITE_FAILURES     8u                                    /* then the card is considered gone or full */
#define READ_SAMPLES           2000u                                 /* classification reads: 48 per window */
#define START_DELAY_S          2u                                    /* to create the file before the first sample */
#define MAX_TRIES              3u                                    /* boots that start a window before it is skipped */
#define SAMPLES_UNKNOWN        0xFFFFFFFFu                           /* CLS of a recording not measured yet */

#define LOG_FILE      "LOG.TXT"
#define STATE_FILE    "STATE.TXT"
#define SESSIONS_FILE "SESSIONS.TXT"
#define REPLAY_FILE   "REPLAY.PCM"   /* "RPLY" + 96000 int16 samples + "END!": run the pipeline on a known window */
#define CONFIG_FILE   "CONFIG.TXT"

static int16_t primaryBuffer[RAW_PER_TRANSFER];
static int16_t secondaryBuffer[RAW_PER_TRANSFER];
BIRD_EXTSRAM static int16_t ring[RING_SAMPLES];  /* decimated 32 kHz samples, filled by the DMA handler */
static int16_t chunk[CHUNK_SAMPLES];             /* SD writes and reads go through internal RAM */
static volatile uint32_t ringWrite;              /* samples written by the handler since the capture started */
static volatile uint32_t transfers;
static volatile bool capturing;

/* Callbacks the hardware library expects from the application. */
void AudioMoth_handleSwitchInterrupt(void) {}
void AudioMoth_handleMicrophoneChangeInterrupt(void) {}
void AudioMoth_handleMicrophoneInterrupt(int16_t sample) { (void)sample; }
void AudioMoth_timezoneRequested(int8_t *hours, int8_t *minutes) { *hours = 0; *minutes = 0; }

void AudioMoth_handleDirectMemoryAccessInterrupt(bool isPrimaryBuffer, int16_t **nextBuffer) {
    (void)nextBuffer;
    int16_t *source = isPrimaryBuffer ? primaryBuffer : secondaryBuffer;
    /* The filter runs always so its state is settled when the recording starts. */
    int16_t decimated[SAMPLES_PER_TRANSFER];
    DigitalFilter_applyFilter(source, decimated, SAMPLE_RATE_DIVIDER, RAW_PER_TRANSFER);
    transfers += 1;
    if (!capturing) return;
    /* Never blocks: if the main loop has fallen a whole ring behind, this overwrites audio it has not saved yet;
     * the main loop notices (ringWrite - read too large) and writes silence for it. */
    memcpy(ring + (ringWrite % RING_SAMPLES), decimated, sizeof decimated);
    ringWrite += SAMPLES_PER_TRANSFER;
}

static void Blink(bool green, uint32_t times) {
    for (uint32_t i = 0; i < times; ++i) {
        if (green) AudioMoth_setGreenLED(true); else AudioMoth_setRedLED(true);
        AudioMoth_delay(150);
        AudioMoth_setGreenLED(false);
        AudioMoth_setRedLED(false);
        AudioMoth_delay(150);
        AudioMoth_feedWatchdog();
    }
}

static void SleepFor(uint32_t seconds) {
    if (seconds > MAX_SLEEP_CHUNK_S) seconds = MAX_SLEEP_CHUNK_S;
    if (seconds < 1) seconds = 1;
    AudioMoth_powerDownAndWake(seconds, true);  /* EM4 power-down: RAM is lost, the RTC keeps running; wakes into main() */
}

/* CONFIG.TXT -> set the clock. Returns true when the clock is (now) set. */
static bool ApplyConfigFile(void) {
    if (!AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) return AudioMoth_hasTimeBeenSet();
    bool applied = false;
    if (AudioMoth_doesFileExist(CONFIG_FILE) && AudioMoth_openFileToRead(CONFIG_FILE)) {
        char text[40];
        memset(text, 0, sizeof text);
        const bool read = AudioMoth_readFile(text, sizeof text - 1);
        AudioMoth_closeFile();
        uint32_t seconds;
        if (read && Sched_ParseConfig(text, (uint32_t)strlen(text), &seconds)) {
            AudioMoth_setTime(seconds, 0);
            applied = true;
            /* Mark the file as used. (Not renamed: FatFS's f_rename fails when the target already exists, which
             * would leave a stale CONFIG.TXT behind after the second clock setting.) */
            char mark[48] = "APPLIED ";
            Sched_FormatTime(seconds, mark + 8);
            mark[27] = '\n';
            if (AudioMoth_openFile(CONFIG_FILE)) {
                AudioMoth_writeToFile(mark, 28);
                AudioMoth_closeFile();
            }
        }
    }
    AudioMoth_disableFileSystem();
    return applied || AudioMoth_hasTimeBeenSet();
}

/* Appends to a text file (the library keeps one file open at a time). */
static bool AppendLine(const char *name, const char *line, uint32_t length) {
    if (!AudioMoth_appendFile((char *)name)) return false;
    const bool ok = AudioMoth_writeToFile((void *)line, (uint16_t)length);
    return AudioMoth_closeFile() && ok;
}

static void LogLine(const char *line, uint32_t length) {
    if (!AppendLine(LOG_FILE, line, length)) Blink(false, 2);
}

/* STATE.TXT is a fixed-length record overwritten in place. (Not with openFile: FA_CREATE_ALWAYS truncates the file
 * at once, and a cut before the new content is written would leave it empty, abandoning the session.) */
static bool SaveState(const SchedState *st) {
    char text[SCHED_STATE_RECORD];
    Sched_FormatState(st, text);
    if (!AudioMoth_appendFile(STATE_FILE)) return false;
    const bool ok = AudioMoth_seekInFile(0) && AudioMoth_writeToFile(text, SCHED_STATE_RECORD);
    return AudioMoth_closeFile() && ok;
}

static bool LoadState(SchedState *st) {
    if (!AudioMoth_doesFileExist(STATE_FILE) || !AudioMoth_openFileToRead(STATE_FILE)) return false;
    char text[SCHED_STATE_RECORD];
    memset(text, 0, sizeof text);
    uint32_t size = AudioMoth_getFileSize();
    if (size > SCHED_STATE_RECORD) size = SCHED_STATE_RECORD;
    const bool read = AudioMoth_readFile(text, size);
    AudioMoth_closeFile();
    return read && Sched_ParseState(text, size, st);
}

static void SaveDone(void) {
    const SchedState done = {SCHED_STATE_DONE, {0}, 0, 0, 0, 0, 0};
    if (!SaveState(&done)) Blink(false, 2);
}

/* Samples in a recording, from its size: of the open file, or of `path` (opened and closed). */
static uint32_t SamplesInOpenFile(void) {
    const uint32_t size = AudioMoth_getFileSize();
    return size > SCHED_WAV_HEADER ? (size - SCHED_WAV_HEADER) / 2u : 0;
}

static bool SamplesInFile(const char *path, uint32_t *samples) {
    if (!AudioMoth_openFileToRead((char *)path)) return false;
    *samples = SamplesInOpenFile();
    return AudioMoth_closeFile();
}

/* Microphone: same configuration as the original firmware (384 kHz raw, divider 12 -> 32 kHz, medium gain). */
static void StartMicrophone(void) {
    DigitalFilter_designHighPassFilter(OUTPUT_SAMPLE_RATE, DC_BLOCKING_FREQ_HZ);
    float multiplier = 16.0f / (float)(1u * SAMPLE_RATE_DIVIDER);
    if (AudioMoth_hasInvertedOutput()) multiplier = -multiplier;
    const bool externalMicrophone = AudioMoth_enableMicrophone(AM_NORMAL_GAIN_RANGE, AM_GAIN_MEDIUM, 4, 16, 1);
    if (externalMicrophone) multiplier = -multiplier;
    DigitalFilter_setAdditionalGain(multiplier);
    AudioMoth_initialiseDirectMemoryAccess(primaryBuffer, secondaryBuffer, RAW_PER_TRANSFER);
    transfers = 0;
    capturing = false;
    AudioMoth_startMicrophoneSamples(RAW_SAMPLE_RATE);
    while (transfers <= SETTLE_TRANSFERS) { AudioMoth_sleep(); AudioMoth_feedWatchdog(); }
}

/* Writes n samples of silence (a stretch lost to the card). */
static bool WriteSilence(uint32_t n) {
    memset(chunk, 0, sizeof chunk);
    bool ok = true;
    while (n > 0 && ok) {
        const uint32_t k = n < CHUNK_SAMPLES ? n : CHUNK_SAMPLES;
        ok = AudioMoth_writeToFile(chunk, (uint16_t)(2u * k));
        n -= k;
    }
    return ok;
}

/* After a failed write the library's file object stays in error: reopen the file, continue from what is really in it
 * and fill the stretch up to `timeline` (samples since the capture started) with silence. False if the card is gone. */
static bool RecoverFromWriteFailure(const char *path, uint32_t timeline, uint32_t *written, uint32_t *gaps) {
    AudioMoth_closeFile();
    if (!AudioMoth_appendFile((char *)path)) return false;
    const uint32_t have = SamplesInOpenFile();
    if (!AudioMoth_seekInFile(SCHED_WAV_HEADER + 2u * have)) return false;  /* drops a torn half sample */
    *written = have;
    if (have < timeline) {
        if (!WriteSilence(timeline - have)) return true;  /* reported by the next write */
        *gaps += timeline - have;
        *written = timeline;
    }
    return true;
}

/* Records to the open WAV file `path`, which already holds `base` samples, until it holds `wanted`. Returns the samples
 * in the file: audio, plus silence for stretches the card lost (added to *gaps). Stops early if the card keeps failing. */
static uint32_t RecordToOpenFile(const char *path, uint32_t base, uint32_t wanted, uint32_t *gaps) {
    uint32_t read = 0, chunks = 0, failures = 0;  /* read: samples taken from the ring, a multiple of CHUNK_SAMPLES */
    uint32_t written = base;                      /* invariant: written == min(base + read, wanted) */
    ringWrite = 0;
    capturing = true;
    while (written < wanted) {
        const uint32_t available = ringWrite - read;
        bool ok = true;
        if (available > RING_SAMPLES - CHUNK_SAMPLES) {
            /* Less than one chunk of headroom: skip whole chunks (so a chunk never wraps around the ring) up to one
             * chunk behind the handler and write silence for them, so every later sample keeps its real time. */
            const uint32_t skip = (available - (RING_SAMPLES - CHUNK_SAMPLES) + CHUNK_SAMPLES - 1u) / CHUNK_SAMPLES * CHUNK_SAMPLES;
            const uint32_t n = skip < wanted - written ? skip : wanted - written;
            ok = WriteSilence(n);
            if (ok) { written += n; *gaps += n; }
            read += skip;
        } else if (available < CHUNK_SAMPLES) {
            AudioMoth_sleep();  /* wait for the next DMA interrupt */
            AudioMoth_feedWatchdog();
            continue;
        } else {
            memcpy(chunk, ring + (read % RING_SAMPLES), sizeof chunk);
            if (ringWrite - read > RING_SAMPLES) {  /* lapped by the handler during the copy */
                memset(chunk, 0, sizeof chunk);
                *gaps += CHUNK_SAMPLES;
            }
            const uint32_t n = CHUNK_SAMPLES < wanted - written ? CHUNK_SAMPLES : wanted - written;
            ok = AudioMoth_writeToFile(chunk, (uint16_t)(2u * n));
            if (ok) written += n;
            read += CHUNK_SAMPLES;
            if (ok && ++chunks % SYNC_EVERY_CHUNKS == 0) ok = AudioMoth_syncFile();
        }
        if (!ok) {
            Blink(false, 1);
            if (++failures > MAX_WRITE_FAILURES ||
                !RecoverFromWriteFailure(path, base + read < wanted ? base + read : wanted, &written, gaps)) break;
        }
        AudioMoth_feedWatchdog();
    }
    capturing = false;
    return written;
}

/* Creates YYYYMMDD/ if needed and writes path with a header for 0 samples (a recording that is never finished is
 * resumed from its size); the file is left closed. */
static bool CreateWav(const char *path) {
    char folder[9];
    memcpy(folder, path, 8);
    folder[8] = 0;
    if (!AudioMoth_doesDirectoryExist(folder) && !AudioMoth_makeDirectory(folder)) return false;
    uint8_t header[SCHED_WAV_HEADER];
    Sched_WavHeader(header, 0);
    if (!AudioMoth_openFile((char *)path)) return false;
    const bool ok = AudioMoth_writeToFile(header, SCHED_WAV_HEADER);
    return AudioMoth_closeFile() && ok;
}

/* Measures a recording from its size and rewrites its header. False if the card does not answer. */
static bool FinishWav(const char *path, uint32_t *samples) {
    if (!AudioMoth_appendFile((char *)path)) return false;
    *samples = SamplesInOpenFile();
    uint8_t header[SCHED_WAV_HEADER];
    Sched_WavHeader(header, *samples);
    const bool ok = AudioMoth_seekInFile(0) && AudioMoth_writeToFile(header, SCHED_WAV_HEADER);
    return AudioMoth_closeFile() && ok;
}

/* Classifies the windows of a recording from st->next on, logging one line each and saving the progress after
 * each, then writes the session summary and marks the state DONE. Needs the file system; enables the SRAM.
 * Returns false (state kept, retried at the next boot) if the network or the recording cannot be opened. */
static bool ClassifyRecording(SchedState *st) {
    uint32_t begin, ms;
    AudioMoth_getTime(&begin, &ms);
    uint32_t inFile;
    if (st->samples == SAMPLES_UNKNOWN) {  /* a recording whose header could not be fixed when it was resumed */
        if (!FinishWav(st->path, &inFile)) { Blink(false, 6); return false; }
        st->samples = inFile;
    }
    if (!SamplesInFile(st->path, &inFile)) { Blink(false, 6); return false; }
    if (inFile < st->samples) st->samples = inFile;  /* never classify past the end of the file */
    const uint32_t windows = Sched_WindowCount(st->samples);
    if (st->next < windows && !BirdPipeline_Init()) { Blink(false, 5); return false; }
    for (; st->next < windows; ++st->next) {
        bool ok = AudioMoth_openFileToRead(st->path) &&
                  AudioMoth_seekInFile(SCHED_WAV_HEADER + 2u * st->next * SCHED_HOP_SAMPLES);
        if (ok) {
            BirdPipeline_Begin();
            for (uint32_t i = 0; i < SCHED_WINDOW_SAMPLES / READ_SAMPLES && ok; ++i) {
                ok = AudioMoth_readFile((char *)chunk, 2u * READ_SAMPLES);
                BirdPipeline_Push(chunk, READ_SAMPLES);
                AudioMoth_feedWatchdog();
            }
        }
        AudioMoth_closeFile();
        float logits[SCHED_CLASSES];
        if (!ok || !BirdPipeline_Ready()) {
            Blink(false, 6);  /* unreadable window: skipped */
        } else if (!BirdPipeline_Finish(logits)) {
            Blink(false, 3);
        } else {
            /* A cut between this line and the saved progress below logs this window again at the next boot (same
             * time); the progress is saved after the line so that a window is never lost. */
            char line[SCHED_LOG_LINE_MAX];
            const uint32_t time = st->start + (st->next * SCHED_HOP_SAMPLES) / SCHED_SAMPLE_RATE;
            LogLine(line, Sched_FormatLogLine(line, time, logits));
            if (Sched_TopPercent(logits) >= SCHED_CONFIDENCE_PCT && Sched_TopClass(logits) != MODEL_NEGATIVE_CLASS) Blink(true, 1);
        }
        SchedState progress = *st;
        progress.next = st->next + 1;
        progress.tries = 1;  /* this boot is about to start window next + 1 */
        if (!SaveState(&progress)) Blink(false, 2);
    }
    uint32_t end;
    AudioMoth_getTime(&end, &ms);
    char line[SCHED_SESSION_LINE_MAX];
    if (!AppendLine(SESSIONS_FILE, line, Sched_FormatSession(line, st, windows, end - begin))) Blink(false, 2);
    SaveDone();
    return true;
}

/* A recording or classification interrupted by a reset or a power cut: finish it. Works without a clock (the times
 * are in STATE.TXT). Returns true when nothing is left pending. A window that crashes the board is skipped after
 * MAX_TRIES boots started it; a session that cannot be classified at all (recording unreadable, network not
 * starting) for MAX_TRIES boots is abandoned, so it cannot block the next sessions. */
static bool ResumePending(void) {
    if (!AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) return true;
    bool clear = true;
    SchedState st;
    if (LoadState(&st) && st.kind != SCHED_STATE_DONE) {
        if (!AudioMoth_doesFileExist(st.path)) {  /* the recording is gone (card swapped or edited): nothing to do */
            SaveDone();
        } else {
            if (st.kind == SCHED_STATE_REC) {
                /* The recording stopped early: keep what reached the card (up to the last sync), fix its header. If
                 * the file cannot be opened now, ClassifyRecording() tries again (and gives up after MAX_TRIES). */
                uint32_t samples = SAMPLES_UNKNOWN;
                if (!FinishWav(st.path, &samples)) samples = SAMPLES_UNKNOWN;
                st.kind = SCHED_STATE_CLS;
                st.samples = samples;
                st.next = 0;
                st.gaps = 0;
                st.tries = 0;
            }
            /* tries = boots that started window next, this one included */
            if (++st.tries > MAX_TRIES) { st.next += 1; st.tries = 1; Blink(false, 7); }
            if (!SaveState(&st)) {
                clear = false;
            } else if (!ClassifyRecording(&st)) {
                if (st.tries >= MAX_TRIES) {  /* no progress possible: abandon, with a summary line */
                    char line[SCHED_SESSION_LINE_MAX];
                    if (st.samples == SAMPLES_UNKNOWN) st.samples = 0;
                    AppendLine(SESSIONS_FILE, line, Sched_FormatSession(line, &st, st.next, 0));
                    SaveDone();
                    Blink(false, 7);
                } else {
                    clear = false;
                }
            }
            AudioMoth_disableExternalSRAM();
        }
    }
    AudioMoth_disableFileSystem();
    return clear;
}

/* Records from `start` (a whole second, at least START_DELAY_S from now) until `end` (clock seconds) to
 * YYYYMMDD/HHMMSS.WAV; the state goes REC -> CLS. The microphone must be running and the SRAM and file system
 * enabled. Returns true if a recording to classify is in *st. */
static bool RecordWindow(uint32_t start, uint32_t end, bool saveState, SchedState *st) {
    uint32_t now, ms;
    st->kind = SCHED_STATE_REC;
    st->start = start;
    st->samples = st->next = st->gaps = st->tries = 0;
    if (end < st->start + SCHED_WINDOW_SAMPLES / SCHED_SAMPLE_RATE) return false;  /* not even one window left */
    Sched_WavPath(st->start, st->path);
    /* REC is saved before the recording (one file open at a time), so a cut during it is resumed. */
    if (!CreateWav(st->path) || (saveState && !SaveState(st)) || !AudioMoth_appendFile(st->path)) {
        Blink(false, 2);
        return false;
    }
    /* Start exactly on the second: then the file name and every LOG.TXT time are the times of the samples. */
    do { AudioMoth_sleep(); AudioMoth_feedWatchdog(); AudioMoth_getTime(&now, &ms); } while (now < st->start);
    uint32_t gaps = 0, late = 0;
    if (now > st->start) {  /* creating the file took more than START_DELAY_S: the missed seconds become silence */
        late = (now - st->start) * SCHED_SAMPLE_RATE;
        if (late > (end - st->start) * SCHED_SAMPLE_RATE) late = (end - st->start) * SCHED_SAMPLE_RATE;
        if (WriteSilence(late)) gaps = late; else late = 0;
    }
    const uint32_t wanted = end > st->start ? (end - st->start) * SCHED_SAMPLE_RATE : 0;
    const uint32_t written = RecordToOpenFile(st->path, late, wanted, &gaps);
    AudioMoth_closeFile();
    uint32_t samples = 0;
    if (!FinishWav(st->path, &samples)) { Blink(false, 2); return false; }  /* REC kept: resumed next boot */
    if (samples < written || samples < wanted) Blink(false, 2);
    if (gaps > 0) Blink(false, 1);
    st->kind = SCHED_STATE_CLS;
    st->samples = samples;
    st->gaps = gaps;
    if (saveState && !SaveState(st)) Blink(false, 2);
    return true;
}

/* If a REPLAY.PCM is on the card, classify it instead of the microphone and log the result with a REPLAY
 * prefix: this tests the real chip (front-end, model, SD) with audio whose expected output is known. */
static bool ReplayIfRequested(void) {
    if (!AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) return false;
    bool handled = false;
    if (AudioMoth_doesFileExist(REPLAY_FILE) && AudioMoth_openFileToRead(REPLAY_FILE)) {
        const uint32_t size = AudioMoth_getFileSize();
        if (size <= 1) {  /* already processed (truncated to "x" below): nothing to do, no blinking at every boot */
            AudioMoth_closeFile();
            AudioMoth_disableFileSystem();
            return false;
        }
        /* readFile() does not report short reads: the size check rejects truncated files. */
        bool complete = size == 4u + 2u * SCHED_WINDOW_SAMPLES + 4u;
        char head[4] = {0, 0, 0, 0};
        complete = complete && AudioMoth_readFile(head, 4) && memcmp(head, "RPLY", 4) == 0;
        if (complete && BirdPipeline_Init()) {
            BirdPipeline_Begin();
            for (uint32_t i = 0; i < SCHED_WINDOW_SAMPLES / READ_SAMPLES && complete; ++i) {
                complete = AudioMoth_readFile((char *)chunk, 2u * READ_SAMPLES);
                BirdPipeline_Push(chunk, READ_SAMPLES);
                AudioMoth_feedWatchdog();
            }
            char tail[4] = {0, 0, 0, 0};
            complete = complete && AudioMoth_readFile(tail, 4) && memcmp(tail, "END!", 4) == 0;
            AudioMoth_closeFile();
            float logits[SCHED_CLASSES];
            if (complete && BirdPipeline_Ready() && BirdPipeline_Finish(logits)) {
                uint32_t now, ms;
                AudioMoth_getTime(&now, &ms);
                char line[SCHED_LOG_LINE_MAX + 8];
                memcpy(line, "REPLAY,", 7);
                const uint32_t length = Sched_FormatLogLine(line + 7, now, logits);
                LogLine(line, length + 7);
                handled = true;
            }
        } else {
            AudioMoth_closeFile();
        }
        /* Truncate the file so it is not processed again at the next boot (also when it was malformed). */
        if (AudioMoth_openFile(REPLAY_FILE)) { AudioMoth_writeToFile("x", 1); AudioMoth_closeFile(); }
        if (!handled) Blink(false, 6);
        AudioMoth_disableExternalSRAM();
    }
    AudioMoth_disableFileSystem();
    return handled;
}

/* The daily window: record it from `start` to `end`, then (microphone off) classify the recording. */
static void RunSession(uint32_t start, uint32_t end) {
    if (!AudioMoth_enableExternalSRAM()) { Blink(false, 5); return; }
    if (!AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) { Blink(false, 4); AudioMoth_disableExternalSRAM(); return; }
    StartMicrophone();
    SchedState st;
    const bool recorded = RecordWindow(start, end, true, &st);
    AudioMoth_disableMicrophone();
    if (recorded) ClassifyRecording(&st);
    AudioMoth_disableFileSystem();
    AudioMoth_disableExternalSRAM();  /* the external SRAM draws current whenever it is enabled */
}

#ifdef BIRD_DIAG_RECORD
/* Diagnostic build: no neural network. Records 3 files of 3 s (YYYYMMDD/HHMMSS.WAV) with the same recorder as the
 * normal build, so the microphone chain, the decimation filter and the SD card can be checked on the real device
 * and the audio analysed on a computer (tools/board/analyze_diag_wav.py). */
static void RunDiagRecord(void) {
    if (!AudioMoth_enableExternalSRAM() || !AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) { Blink(false, 4); return; }
    StartMicrophone();
    for (int w = 0; w < 3; ++w) {
        uint32_t now, ms;
        AudioMoth_getTime(&now, &ms);
        SchedState st;
        const uint32_t seconds = SCHED_WINDOW_SAMPLES / SCHED_SAMPLE_RATE;
        if (RecordWindow(now + START_DELAY_S, now + START_DELAY_S + seconds, false, &st) && st.samples == seconds * SCHED_SAMPLE_RATE) Blink(true, 1);
        else Blink(false, 2);
    }
    AudioMoth_disableMicrophone();
    AudioMoth_disableFileSystem();
    AudioMoth_disableExternalSRAM();
}
#endif

int main(void) {
    AudioMoth_initialise();

    /* Set the clock from the SD card on first power-up or whenever it has never been set. */
    if (AudioMoth_isInitialPowerUp() || !AudioMoth_hasTimeBeenSet()) {
        if (!ApplyConfigFile()) {
            /* No clock: any reset other than a wake-up from EM4 clears it (watchdog, brown-out, power cut). A pending
             * recording or classification does not need it, so finish that first; then retry the clock in a minute. */
#ifndef BIRD_DIAG_RECORD
            ResumePending();
#endif
            Blink(false, 3);
            SleepFor(60);
            return 0;
        }
    }

#ifdef BIRD_DIAG_RECORD
    RunDiagRecord();
    SleepFor(600);
    return 0;
#endif
    if (ReplayIfRequested()) Blink(true, 3);
    const bool clear = ResumePending();

    uint32_t now, ms;
    AudioMoth_getTime(&now, &ms);
    /* The board wakes START_DELAY_S before the window to create the file, so the recording starts with it; booted
     * inside the window (clock just set, reset), it records what is left. */
    const uint32_t left = Sched_SecondsLeftInWindow(now), toNext = Sched_SecondsToNextWindow(now);
    uint32_t start = 0, end = 0;
    if (left > 0) { start = now + START_DELAY_S; end = now + left; }
    else if (toNext <= START_DELAY_S) { start = now + toNext; end = start + SCHED_WINDOW_LENGTH_S; }
    /* A session still pending (its classification could not start) keeps STATE.TXT: no new recording over it. */
    if (clear && end >= start + SCHED_WINDOW_SAMPLES / SCHED_SAMPLE_RATE) {
        RunSession(start, end);
        AudioMoth_getTime(&now, &ms);
    }
    /* Wake START_DELAY_S before the next window; inside the window with less than one window left, sleep past its
     * end, not in 1 s steps. */
    uint32_t wait = Sched_SecondsToNextWindow(now);
    if (wait == 0) wait = Sched_SecondsLeftInWindow(now) + 1;
    else if (wait > START_DELAY_S) wait -= START_DELAY_S;
    SleepFor(wait);
    return 0;
}
