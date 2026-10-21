/*
 * WrenNet logger for the AudioMoth.
 *
 * Every day in the window of the model package (default 07:00 for 5 minutes, device clock): record 3 s windows at
 * 32 kHz, classify each with the on-device model and append one line per window to LOG.TXT on the SD card
 * ("2026-10-21 07:00:12,Apus apus,83,Y"). Outside that window the device stays powered down.
 *
 * The clock is set from CONFIG.TXT on the SD card (one line "YYYY-MM-DD hh:mm:ss"); the file is
 * overwritten with 'APPLIED <time>' once applied. No USB, GPS, WAV writing or configuration app.
 *
 * Hardware calls follow the original AudioMoth firmware (main.c: makeRecording / DMA handler).
 */
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "audiomoth.h"
#include "bird_pipeline.h"
#include "digitalfilter.h"
#include "schedule.h"

/* ADC at 384 kHz decimated by 12 gives the 32 kHz the model was trained at. */
#define RAW_SAMPLE_RATE        384000u
#define SAMPLE_RATE_DIVIDER    12u
#define OUTPUT_SAMPLE_RATE     (RAW_SAMPLE_RATE / SAMPLE_RATE_DIVIDER)
#define SAMPLES_PER_TRANSFER   64u                                   /* decimated samples per DMA transfer */
#define RAW_PER_TRANSFER       (SAMPLES_PER_TRANSFER * SAMPLE_RATE_DIVIDER) /* 768 <= 1024 */
#define STAGE_SIZE             1024u                                 /* 32 ms of audio; multiple of SAMPLES_PER_TRANSFER */
#define SETTLE_TRANSFERS       100u                                  /* ~200 ms: microphone and filter settling */
#define DC_BLOCKING_FREQ_HZ    48u
#define MIN_SECONDS_TO_CLASSIFY 6u
#define MAX_SLEEP_CHUNK_S      (6u * 3600u)

#define LOG_FILE      "LOG.TXT"
#define REPLAY_FILE   "REPLAY.PCM"   /* "RPLY" + 96000 int16 samples + "END!": run the pipeline on a known window */
#define CONFIG_FILE   "CONFIG.TXT"

static int16_t primaryBuffer[RAW_PER_TRANSFER];
static int16_t secondaryBuffer[RAW_PER_TRANSFER];
static int16_t stage[STAGE_SIZE];          /* decimated 32 kHz samples, filled by the DMA handler */
static volatile uint32_t stageWrite, stageRead;
static volatile uint32_t transfers;
static volatile bool capturing, overrun;

/* Callbacks the hardware library expects from the application. */
void AudioMoth_handleSwitchInterrupt(void) {}
void AudioMoth_handleMicrophoneChangeInterrupt(void) {}
void AudioMoth_handleMicrophoneInterrupt(int16_t sample) { (void)sample; }
void AudioMoth_timezoneRequested(int8_t *hours, int8_t *minutes) { *hours = 0; *minutes = 0; }

void AudioMoth_handleDirectMemoryAccessInterrupt(bool isPrimaryBuffer, int16_t **nextBuffer) {
    (void)nextBuffer;
    int16_t *source = isPrimaryBuffer ? primaryBuffer : secondaryBuffer;
    /* The filter runs always so its state is settled when a window starts. */
    int16_t decimated[SAMPLES_PER_TRANSFER];
    DigitalFilter_applyFilter(source, decimated, SAMPLE_RATE_DIVIDER, RAW_PER_TRANSFER);
    transfers += 1;
    if (!capturing || transfers <= SETTLE_TRANSFERS) return;
    if (stageWrite - stageRead > STAGE_SIZE - SAMPLES_PER_TRANSFER) { overrun = true; return; }
    memcpy(stage + (stageWrite & (STAGE_SIZE - 1)), decimated, sizeof decimated);
    stageWrite += SAMPLES_PER_TRANSFER;
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

static void LogLine(const char *line, uint32_t length) {
    if (AudioMoth_appendFile(LOG_FILE)) {
        const bool ok = AudioMoth_writeToFile((void *)line, (uint16_t)length);
        AudioMoth_closeFile();
        if (!ok) Blink(false, 2);
    } else {
        Blink(false, 2);
    }
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
    stageWrite = stageRead = 0;
    transfers = 0;
    capturing = false;
    AudioMoth_startMicrophoneSamples(RAW_SAMPLE_RATE);
}

/* If a REPLAY.PCM is on the card, classify it instead of the microphone and log the result with a REPLAY
 * prefix: this tests the real chip (front-end, model, SD) with audio whose expected output is known. */
static bool ReplayIfRequested(void) {
    if (!AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) return false;
    bool handled = false;
    if (AudioMoth_doesFileExist(REPLAY_FILE) && AudioMoth_openFileToRead(REPLAY_FILE)) {
        char head[4] = {0, 0, 0, 0};
        bool complete = AudioMoth_readFile(head, 4) && memcmp(head, "RPLY", 4) == 0;
        if (complete && BirdPipeline_Init()) {
            BirdPipeline_Begin();
            for (uint32_t i = 0; i < 96000u / 256u && complete; ++i) {
                int16_t block[256];
                complete = AudioMoth_readFile((char *)block, sizeof block);
                BirdPipeline_Push(block, 256);
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
    }
    AudioMoth_disableFileSystem();
    return handled;
}

static void RunSession(void) {
    if (!BirdPipeline_Init()) { Blink(false, 5); return; }
    if (!AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) { Blink(false, 4); return; }

    StartMicrophone();

    uint32_t now, ms;
    for (;;) {
        AudioMoth_getTime(&now, &ms);
        if (Sched_SecondsLeftInWindow(now) < MIN_SECONDS_TO_CLASSIFY) break;

        /* Record one 3 s window, feeding the front-end as the samples arrive. */
        BirdPipeline_Begin();
        overrun = false;
        stageRead = stageWrite;
        capturing = true;
        uint32_t guard = 0;
        while (!BirdPipeline_Ready() && !overrun && guard < 2000000u) {
            const uint32_t available = stageWrite - stageRead;
            if (available >= SAMPLES_PER_TRANSFER) {
                BirdPipeline_Push(stage + (stageRead & (STAGE_SIZE - 1)), SAMPLES_PER_TRANSFER);
                stageRead += SAMPLES_PER_TRANSFER;
            } else {
                AudioMoth_sleep();  /* wait for the next DMA interrupt */
            }
            guard += 1;
            AudioMoth_feedWatchdog();
        }
        capturing = false;
        if (overrun || !BirdPipeline_Ready()) { Blink(false, 1); continue; }

        float logits[SCHED_CLASSES];
        AudioMoth_getTime(&now, &ms);
        if (!BirdPipeline_Finish(logits)) { Blink(false, 3); continue; }

        char line[SCHED_LOG_LINE_MAX];
        const uint32_t length = Sched_FormatLogLine(line, now, logits);
        LogLine(line, length);
        if (Sched_TopPercent(logits) >= SCHED_CONFIDENCE_PCT && Sched_TopClass(logits) != MODEL_NEGATIVE_CLASS) Blink(true, 1);
    }
    AudioMoth_disableMicrophone();
    AudioMoth_disableFileSystem();
    AudioMoth_disableExternalSRAM();  /* the external SRAM draws current whenever it is enabled */
}

#ifdef BIRD_DIAG_RECORD
/* Diagnostic build: no neural network. Records 3 windows of 3 s straight to WAV files (HHMMSS.WAV) so the
 * microphone chain, the decimation filter and the SD card can be checked on the real device, and the audio
 * can be analysed on a computer (levels, spectrum, noise floor). */
#include "bird_extsram.h"
BIRD_EXTSRAM static int16_t g_diag_pcm[96000];

static void PutLE(uint8_t *p, uint32_t v, int bytes) { for (int i = 0; i < bytes; ++i) p[i] = (uint8_t)(v >> (8 * i)); }

static bool WriteWav(const char *name, const int16_t *pcm, uint32_t samples) {
    uint8_t h[44] = "RIFFxxxxWAVEfmt \x10\0\0\0\1\0\1\0xxxxxxxx\2\0\x10\0data";
    const uint32_t bytes = samples * 2u;
    PutLE(h + 4, 36u + bytes, 4);
    PutLE(h + 24, OUTPUT_SAMPLE_RATE, 4);
    PutLE(h + 28, OUTPUT_SAMPLE_RATE * 2u, 4);
    PutLE(h + 40, bytes, 4);
    if (!AudioMoth_openFile((char *)name) || !AudioMoth_writeToFile(h, 44)) return false;
    bool ok = true;
    for (uint32_t off = 0; off < samples && ok; off += 256) {
        ok = AudioMoth_writeToFile((void *)(pcm + off), (uint16_t)(256 * 2));
        AudioMoth_feedWatchdog();
    }
    return AudioMoth_closeFile() && ok;
}

static void RunDiagRecord(void) {
    if (!AudioMoth_enableExternalSRAM() || !AudioMoth_enableFileSystem(AM_SD_CARD_NORMAL_SPEED)) { Blink(false, 4); return; }
    StartMicrophone();
    for (int w = 0; w < 3; ++w) {
        uint32_t count = 0;
        overrun = false;
        stageRead = stageWrite;
        capturing = true;
        while (count < 96000u && !overrun) {
            if (stageWrite - stageRead >= SAMPLES_PER_TRANSFER) {
                memcpy(g_diag_pcm + count, stage + (stageRead & (STAGE_SIZE - 1)), SAMPLES_PER_TRANSFER * 2u);
                stageRead += SAMPLES_PER_TRANSFER;
                count += SAMPLES_PER_TRANSFER;
            } else {
                AudioMoth_sleep();
            }
            AudioMoth_feedWatchdog();
        }
        capturing = false;
        uint32_t now, ms;
        AudioMoth_getTime(&now, &ms);
        char name[16];
        char stamp[20];
        Sched_FormatTime(now, stamp);                  /* "YYYY-MM-DD hh:mm:ss" */
        name[0] = stamp[11]; name[1] = stamp[12]; name[2] = stamp[14]; name[3] = stamp[15]; name[4] = stamp[17]; name[5] = stamp[18];
        memcpy(name + 6, ".WAV", 5);
        if (overrun || !WriteWav(name, g_diag_pcm, count)) Blink(false, 2); else Blink(true, 1);
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
            Blink(false, 3);             /* no clock and no valid CONFIG.TXT: try again in a minute */
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

    uint32_t now, ms;
    AudioMoth_getTime(&now, &ms);
    if (Sched_InWindow(now) && Sched_SecondsLeftInWindow(now) >= MIN_SECONDS_TO_CLASSIFY) {
        RunSession();
        AudioMoth_getTime(&now, &ms);
    }
    /* Inside the window but too close to its end to classify: sleep past the end, not in 1 s steps. */
    uint32_t wait = Sched_SecondsToNextWindow(now);
    if (wait == 0) wait = Sched_SecondsLeftInWindow(now) + 1;
    SleepFor(wait);
    return 0;
}
