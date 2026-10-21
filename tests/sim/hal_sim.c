/*
 * Software model of the AudioMoth hardware, to run main_app.c on a computer.
 *
 * Every process is one power cycle: RTC and SD card persist in SIM_DIR between runs
 * (rtc.state, sd/). The microphone is fed from SIM_AUDIO (raw int16 mono @32 kHz, looped):
 * it is turned into 384 kHz "ADC" samples (x/16 held for 12 samples) so that the real
 * DigitalFilter (sum of 12, x16/12 gain, 48 Hz high-pass) gives back the original signal.
 * What it does NOT model: ADC/DMA timing jitter, analog front-end, real SD driver, RTC drift,
 * power consumption, clocks, external SRAM wait states.
 *
 * The SD card is a folder (SIM_DIR/sd). File calls behave like the library's FatFS wrappers: readFile() succeeds
 * on a short read, one file is open at a time, appendFile() opens read/write at the end, doesFileExist() and
 * doesDirectoryExist() are f_stat (true for a file or a folder). A failed write leaves half of its bytes on the card
 * and the file object in error (writes, seeks and syncs fail) until the file is opened again, as FatFS's fp->err.
 * Each SD write takes 1 ms of simulated time (SIM_SD_WRITE_MS to change it), during which the microphone keeps
 * delivering samples; a card busy for more than 500 ms fails the write, as the SD driver's timeout does. Opening a
 * file takes SIM_SD_OPEN_MS (default 0).
 *
 * SIM_FAULT (comma-separated) injects failures for this boot: nosd (file system never mounts), nosram (external
 * SRAM does not enable), writefail (every SD write fails), appendfail (LOG.TXT cannot be opened), pipefail
 * (classification fails), powercut=MS (power lost MS ms after boot: the clock is lost, unsynced data too),
 * reset=MS (reset MS ms after boot, e.g. watchdog or brown-out: as on the board, any reset but an EM4 wake-up loses
 * the clock, and unsynced data is lost), sdstall=MS (the 50th SD write of the boot takes MS ms; more than 500 fails),
 * wfail=K (the K-th SD write of the boot fails), diskfull=BYTES (the card has room for BYTES more bytes).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "audiomoth.h"
#include "model_config.h"

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
static long synced_size;  /* size of the open file at its last open/sync: what survives a power cut */
static int window_index;
static FILE *window_pcm;
static bool f_nosd, f_nosram, f_writefail, f_appendfail, f_pipefail;
static long f_powercut_ms = -1, f_reset_ms = -1, f_sdstall_ms = -1, f_wfail = -1, f_diskfull = -1, sd_write_ms = 1, sd_open_ms = 0, sd_writes;
static bool file_err;  /* FatFS fp->err: set by a failed write, cleared when a file is opened */
static uint64_t boot_ms;

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
    const char *fault = getenv("SIM_FAULT");
    if (fault) {
        f_nosd = strstr(fault, "nosd") != NULL; f_nosram = strstr(fault, "nosram") != NULL;
        f_writefail = strstr(fault, "writefail") != NULL; f_appendfail = strstr(fault, "appendfail") != NULL;
        f_pipefail = strstr(fault, "pipefail") != NULL;
        const char *c = strstr(fault, "powercut="); if (c) f_powercut_ms = atol(c + 9);
        const char *r = strstr(fault, "reset="); if (r) f_reset_ms = atol(r + 6);
        const char *st = strstr(fault, "sdstall="); if (st) f_sdstall_ms = atol(st + 8);
        const char *wf = strstr(fault, "wfail="); if (wf) f_wfail = atol(wf + 6);
        const char *df = strstr(fault, "diskfull="); if (df) f_diskfull = atol(df + 9);
    }
    const char *wm = getenv("SIM_SD_WRITE_MS");
    if (wm) sd_write_ms = atol(wm);
    const char *om = getenv("SIM_SD_OPEN_MS");  /* time to open a file (a slow card, a long cluster chain) */
    if (om) sd_open_ms = atol(om);
    boot_ms = now_ms;
}

/* Power cut or reset at a given time after boot. _exit() drops buffered, unclosed file data, as a real cut would. */
static void CheckCut(void) {
    if (f_powercut_ms >= 0 && now_ms - boot_ms >= (uint64_t)f_powercut_ms) {
        char p[1024];
        Path(p, "", "rtc.state");
        remove(p);  /* datasheet: the time is lost when power is lost */
        if (file && ftruncate(fileno(file), synced_size) != 0) perror("ftruncate");  /* unsynced writes are lost */
        printf("SIM: POWER_CUT\n"); fflush(stdout);
        _exit(0);
    }
    if (f_reset_ms >= 0 && now_ms - boot_ms >= (uint64_t)f_reset_ms) {
        char p[1024];
        Path(p, "", "rtc.state");
        remove(p);  /* audiomoth.c: a reset other than an EM4 wake-up clears the time and the "clock set" flag */
        if (file && ftruncate(fileno(file), synced_size) != 0) perror("ftruncate");  /* RAM buffers are lost */
        printf("SIM: RESET\n"); fflush(stdout);
        _exit(0);
    }
}

/* Delivers the DMA transfers that fall inside the elapsed time. */
static void Advance(uint64_t ms_total_x1000) {
    ms_residual += (double)ms_total_x1000 / 1000.0;
    const double transfer_ms = 1000.0 * RAW_PER_TRANSFER / RAW_RATE_HZ; /* 2 ms */
    while (ms_residual >= transfer_ms) {
        ms_residual -= transfer_ms;
        now_ms += 2;
        CheckCut();
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
bool AudioMoth_enableExternalSRAM(void) { return !f_nosram; }
void AudioMoth_disableExternalSRAM(void) {}
bool AudioMoth_hasTimeBeenSet(void) { return time_set; }
void AudioMoth_setTime(uint32_t t, uint32_t ms) { now_ms = (uint64_t)t * 1000u + ms; time_set = true; ms_residual = 0; SaveState(); }
void AudioMoth_getTime(uint32_t *t, uint32_t *ms) { if (t) *t = (uint32_t)(now_ms / 1000u); if (ms) *ms = (uint32_t)(now_ms % 1000u); }
void AudioMoth_feedWatchdog(void) {}
void AudioMoth_delay(uint32_t ms) { AdvanceMs(ms); }
void AudioMoth_sleep(void) { if (mic_on) Advance(2000); else AdvanceMs(1); }

void AudioMoth_powerDownAndWake(uint32_t seconds, bool synchronised) {
    now_ms += (uint64_t)seconds * 1000u;
    if (synchronised && seconds > 0) now_ms -= now_ms % 1000u;  /* as the library: wakes on a whole second of the BURTC */
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

bool AudioMoth_enableFileSystem(AM_sdCardSpeed_t s) { (void)s; return !f_nosd; }
void AudioMoth_disableFileSystem(void) {}
/* The library has one FIL: opening a file while another is open would silently corrupt the first one on the board. */
static void OneFileAtATime(const char *name) {
    if (sd_open_ms > 0) AdvanceMs((uint32_t)sd_open_ms);
    if (file) { fprintf(stderr, "SIM: %s opened while another file is open (the library has a single file object)\n", name); exit(3); }
}
static bool Exists(const char *p) { struct stat s; return stat(p, &s) == 0; }  /* f_stat: file or folder */
static bool IsFolder(const char *p) { struct stat s; return stat(p, &s) == 0 && S_ISDIR(s.st_mode); }  /* f_open refuses these */
bool AudioMoth_doesFileExist(char *name) { char p[1024]; Path(p, "sd", name); return Exists(p); }
bool AudioMoth_openFileToRead(char *name) { OneFileAtATime(name); char p[1024]; Path(p, "sd", name); if (IsFolder(p)) return false; file = fopen(p, "rb"); file_err = false; return file != NULL; }
/* FatFS f_read: FR_OK on a short read or at the end of the file; the caller cannot tell how much was read. */
bool AudioMoth_readFile(char *buf, uint32_t size) { if (!file) return false; size_t got = fread(buf, 1, size, file); (void)got; return !ferror(file); }
bool AudioMoth_appendFile(char *name) {  /* FA_OPEN_ALWAYS | FA_WRITE | FA_READ, then seek to the end */
    OneFileAtATime(name);
    if (f_appendfail && !strcmp(name, "LOG.TXT")) return false;
    char p[1024]; Path(p, "sd", name);
    if (IsFolder(p)) return false;
    file_err = false;
    file = fopen(p, "r+b");
    if (!file) file = fopen(p, "w+b");
    if (!file || fseek(file, 0, SEEK_END) != 0) return false;
    synced_size = ftell(file);
    return true;
}
bool AudioMoth_writeToFile(void *bytes, uint16_t n) {
    if (f_writefail || !file || file_err) return false;
    sd_writes += 1;
    const long busy = sd_writes == 50 && f_sdstall_ms > 0 ? f_sdstall_ms : sd_write_ms;
    const bool fails = busy > 500 || sd_writes == f_wfail;
    AdvanceMs((uint32_t)(busy > 500 ? 500 : busy));
    if (fails) {  /* aborted after part of the data: FR_DISK_ERR, sticky */
        if (fwrite(bytes, 1, (size_t)((n / 2) | 1), file) == 0) perror("fwrite");
        file_err = true;
        return false;
    }
    if (f_diskfull >= 0) {  /* full card: FR_OK with fewer bytes written (not sticky); overwriting needs no room */
        const long pos = ftell(file);
        fseek(file, 0, SEEK_END); const long size = ftell(file); fseek(file, pos, SEEK_SET);
        const long grow = pos + n > size ? pos + n - size : 0;
        const long ok_bytes = grow <= f_diskfull ? n : n - (grow - f_diskfull);
        if (ok_bytes > 0 && fwrite(bytes, 1, (size_t)ok_bytes, file) != (size_t)ok_bytes) return false;
        f_diskfull -= grow <= f_diskfull ? grow : f_diskfull;
        return ok_bytes == n;
    }
    return fwrite(bytes, 1, n, file) == n;
}
bool AudioMoth_closeFile(void) { if (file) fclose(file); file = NULL; synced_size = 0; file_err = false; return true; }
bool AudioMoth_openFile(char *name) { OneFileAtATime(name); char p[1024]; Path(p, "sd", name); if (IsFolder(p)) return false; file = fopen(p, "w+b"); synced_size = 0; file_err = false; return file != NULL; }  /* FA_CREATE_ALWAYS | FA_WRITE | FA_READ */
bool AudioMoth_seekInFile(uint32_t position) { return file && !file_err && fseek(file, (long)position, SEEK_SET) == 0; }
uint32_t AudioMoth_getFileSize(void) {
    if (!file) return 0;
    long here = ftell(file); fseek(file, 0, SEEK_END); long size = ftell(file); fseek(file, here, SEEK_SET);
    return (uint32_t)size;
}
bool AudioMoth_syncFile(void) {  /* unsynced data is what a power cut loses */
    if (!file || fflush(file) != 0) return false;  /* f_sync does not check fp->err: the size reaches the card */
    long here = ftell(file); fseek(file, 0, SEEK_END); synced_size = ftell(file); fseek(file, here, SEEK_SET);
    return true;
}
bool AudioMoth_doesDirectoryExist(char *name) { char p[1024]; Path(p, "sd", name); return Exists(p); }
bool AudioMoth_makeDirectory(char *name) { if (f_writefail) return false; char p[1024]; Path(p, "sd", name); return mkdir(p, 0755) == 0; }

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
    const bool ok = BirdPipeline_Finish(logits) && !f_pipefail;
    if (window_pcm) { fclose(window_pcm); window_pcm = NULL; }
    if (ok) {
        char p[1024], n[64];
        snprintf(n, sizeof n, "win_%04d.logits", window_index);
        Path(p, "dump", n);
        FILE *f = fopen(p, "wb");
        if (f) { fwrite(logits, 4, MODEL_CLASSES, f); fclose(f); }
        window_index += 1;
        SaveState();
    }
    AdvanceMs(3000); /* classification takes a few seconds on the real device; the microphone keeps running */
    return ok;
}
