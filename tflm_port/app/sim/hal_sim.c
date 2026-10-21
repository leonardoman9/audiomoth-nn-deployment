/*
 * Software model of the AudioMoth hardware, to run main_app.c on a computer.
 *
 * Every process is one power cycle: RTC and SD card persist in SIM_DIR between runs
 * (rtc.state, sd/). The microphone is fed from SIM_AUDIO (raw int16 mono @32 kHz, looped):
 * it is turned into 384 kHz "ADC" samples (x/16 held for 12 samples) so that the real
 * DigitalFilter (sum of 12, x16/12 gain, 48 Hz high-pass) gives back the original signal.
 * What it does NOT model: ADC/DMA timing jitter, analog front-end, real SD driver, RTC drift,
 * power consumption, clocks, external SRAM wait states.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audiomoth.h"

#define RAW_PER_TRANSFER 768u
#define RAW_RATE_HZ      384000u

static char dir[512];
static uint64_t now_ms;
static bool time_set, initial_power_up = true;
static int16_t *audio;
static size_t audio_len, audio_pos;
static int16_t *dma_primary, *dma_secondary;
static bool dma_next_is_primary = true, mic_on;
static double ms_residual;
static FILE *file;
static int window_index;
static FILE *window_pcm;

static void Path(char *out, const char *sub, const char *name) { snprintf(out, 1024, "%s/%s%s%s", dir, sub, sub[0] ? "/" : "", name); }

static void SaveState(void) {
    char p[1024];
    Path(p, "", "rtc.state");
    FILE *f = fopen(p, "w");
    if (f) { fprintf(f, "%llu %d\n", (unsigned long long)now_ms, (int)time_set); fclose(f); }
}

__attribute__((constructor)) static void SimInit(void) {
    const char *d = getenv("SIM_DIR");
    if (!d) { fprintf(stderr, "SIM_DIR not set\n"); exit(2); }
    snprintf(dir, sizeof dir, "%s", d);
    char p[1024];
    Path(p, "", "rtc.state");
    FILE *f = fopen(p, "r");
    if (f) {
        unsigned long long ms; int set;
        if (fscanf(f, "%llu %d", &ms, &set) == 2) { now_ms = ms; time_set = set != 0; initial_power_up = false; }
        fclose(f);
    }
    const char *a = getenv("SIM_AUDIO");
    if (a && (f = fopen(a, "rb"))) {
        fseek(f, 0, SEEK_END); audio_len = (size_t)ftell(f) / 2; fseek(f, 0, SEEK_SET);
        audio = malloc(audio_len * 2);
        if (fread(audio, 2, audio_len, f) != audio_len) audio_len = 0;
        fclose(f);
    }
    const char *w = getenv("SIM_WINDOW_INDEX");
    window_index = w ? atoi(w) : 0;
}

/* Delivers the DMA transfers that fall inside the elapsed time. */
static void Advance(uint64_t ms_total_x1000) {
    ms_residual += (double)ms_total_x1000 / 1000.0;
    const double transfer_ms = 1000.0 * RAW_PER_TRANSFER / RAW_RATE_HZ; /* 2 ms */
    while (ms_residual >= transfer_ms) {
        ms_residual -= transfer_ms;
        now_ms += 2;
        if (mic_on && dma_primary && dma_secondary) {
            int16_t *buf = dma_next_is_primary ? dma_primary : dma_secondary;
            for (uint32_t i = 0; i < RAW_PER_TRANSFER / 12; ++i) {
                const int16_t x = audio_len ? audio[audio_pos++ % audio_len] : 0;
                for (int j = 0; j < 12; ++j) buf[i * 12 + (uint32_t)j] = (int16_t)(x / 16);
            }
            int16_t *next = NULL;
            AudioMoth_handleDirectMemoryAccessInterrupt(dma_next_is_primary, &next);
            dma_next_is_primary = !dma_next_is_primary;
        }
    }
}
static void AdvanceMs(uint32_t ms) { Advance((uint64_t)ms * 1000u); }

void AudioMoth_initialise(void) {}
bool AudioMoth_isInitialPowerUp(void) { return initial_power_up; }
bool AudioMoth_hasInvertedOutput(void) { return false; }
bool AudioMoth_enableExternalSRAM(void) { return true; }
void AudioMoth_disableExternalSRAM(void) {}
bool AudioMoth_hasTimeBeenSet(void) { return time_set; }
void AudioMoth_setTime(uint32_t t, uint32_t ms) { now_ms = (uint64_t)t * 1000u + ms; time_set = true; ms_residual = 0; SaveState(); }
void AudioMoth_getTime(uint32_t *t, uint32_t *ms) { if (t) *t = (uint32_t)(now_ms / 1000u); if (ms) *ms = (uint32_t)(now_ms % 1000u); }
void AudioMoth_feedWatchdog(void) {}
void AudioMoth_delay(uint32_t ms) { AdvanceMs(ms); }
void AudioMoth_sleep(void) { if (mic_on) Advance(2000); else AdvanceMs(1); }

void AudioMoth_powerDownAndWake(uint32_t seconds, bool synchronised) {
    (void)synchronised;
    now_ms += (uint64_t)seconds * 1000u;
    SaveState();
    printf("SIM: POWER_DOWN wake_after=%u\n", seconds);
    exit(0);
}

void AudioMoth_setRedLED(bool s) { if (s) printf("SIM: red LED\n"); }
void AudioMoth_setGreenLED(bool s) { if (s) printf("SIM: green LED\n"); }

bool AudioMoth_enableMicrophone(AM_gainRange_t r, AM_gainSetting_t g, uint32_t cd, uint32_t ac, uint32_t os) { (void)r; (void)g; (void)cd; (void)ac; (void)os; return false; }
void AudioMoth_disableMicrophone(void) { mic_on = false; }
void AudioMoth_initialiseDirectMemoryAccess(int16_t *p, int16_t *s, uint16_t n) { dma_primary = p; dma_secondary = s; (void)n; }
void AudioMoth_startMicrophoneSamples(uint32_t rate) { (void)rate; mic_on = true; dma_next_is_primary = true; }

bool AudioMoth_enableFileSystem(AM_sdCardSpeed_t s) { (void)s; return true; }
void AudioMoth_disableFileSystem(void) {}
bool AudioMoth_doesFileExist(char *name) { char p[1024]; Path(p, "sd", name); FILE *f = fopen(p, "rb"); if (f) fclose(f); return f != NULL; }
bool AudioMoth_openFileToRead(char *name) { char p[1024]; Path(p, "sd", name); file = fopen(p, "rb"); return file != NULL; }
bool AudioMoth_readFile(char *buf, uint32_t size) { return file && fread(buf, 1, size, file) > 0; }
bool AudioMoth_appendFile(char *name) { char p[1024]; Path(p, "sd", name); file = fopen(p, "ab"); return file != NULL; }
bool AudioMoth_writeToFile(void *bytes, uint16_t n) { return file && fwrite(bytes, 1, n, file) == n; }
bool AudioMoth_closeFile(void) { if (file) fclose(file); file = NULL; return true; }
bool AudioMoth_openFile(char *name) { char p[1024]; Path(p, "sd", name); file = fopen(p, "wb"); return file != NULL; }  /* create/truncate, as FA_CREATE_ALWAYS */

/* Pipeline wrappers: dump every window's PCM and logits, and charge the classification time. */
void sim_BirdPipeline_Begin(void) {
    BirdPipeline_Begin();
    if (window_pcm) fclose(window_pcm);
    char p[1024], n[64];
    snprintf(n, sizeof n, "win_%04d.pcm", window_index);
    Path(p, "dump", n);
    window_pcm = fopen(p, "wb");
}
void sim_BirdPipeline_Push(const int16_t *pcm, uint32_t n) {
    BirdPipeline_Push(pcm, n);
    if (window_pcm) fwrite(pcm, 2, n, window_pcm);
}
bool sim_BirdPipeline_Finish(float *logits) {
    const bool ok = BirdPipeline_Finish(logits);
    if (window_pcm) { fclose(window_pcm); window_pcm = NULL; }
    if (ok) {
        char p[1024], n[64];
        snprintf(n, sizeof n, "win_%04d.logits", window_index);
        Path(p, "dump", n);
        FILE *f = fopen(p, "wb");
        if (f) { fwrite(logits, 4, 9, f); fclose(f); }
        window_index += 1;
        SaveState();
    }
    AdvanceMs(3000); /* classification takes a few seconds on the real device; the microphone keeps running */
    return ok;
}
