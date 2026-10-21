/* Runs the application's SD-card call sequences on the real FatFS (with the app's trimmed ffconf.h). */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../../firmware/app/schedule.h"

/* the library wrappers under test */
bool AudioMoth_doesFileExist(char *filename);
bool AudioMoth_openFile(char *filename);
bool AudioMoth_appendFile(char *filename);
bool AudioMoth_openFileToRead(char *filename);
bool AudioMoth_readFile(char *buffer, uint32_t bufferSize);
bool AudioMoth_writeToFile(void *bytes, uint16_t bytesToWrite);
bool AudioMoth_renameFile(char *a, char *b);
bool AudioMoth_closeFile(void);
bool AudioMoth_seekInFile(uint32_t position);
uint32_t AudioMoth_getFileSize(void);
bool AudioMoth_syncFile(void);
bool AudioMoth_doesDirectoryExist(char *folderName);
bool AudioMoth_makeDirectory(char *folderName);
bool Shim_mount(void);
void Shim_unmount(void);

static int fails;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main(void) {
    CHECK(Shim_mount());
    /* 1. clock setting, exactly as ApplyConfigFile() does it */
    CHECK(AudioMoth_doesFileExist("CONFIG.TXT"));
    CHECK(AudioMoth_openFileToRead("CONFIG.TXT"));
    char text[40] = {0};
    CHECK(AudioMoth_readFile(text, sizeof text - 1));
    CHECK(AudioMoth_closeFile());
    uint32_t seconds = 0;
    CHECK(Sched_ParseConfig(text, (uint32_t)strlen(text), &seconds));
    char mark[48] = "APPLIED ";
    Sched_FormatTime(seconds, mark + 8);
    mark[27] = '\n';
    CHECK(AudioMoth_openFile("CONFIG.TXT"));
    CHECK(AudioMoth_writeToFile(mark, 28));
    CHECK(AudioMoth_closeFile());
    /* a marked file must not parse as a clock setting any more */
    CHECK(AudioMoth_openFileToRead("CONFIG.TXT"));
    char again[40] = {0};
    CHECK(AudioMoth_readFile(again, sizeof again - 1));
    CHECK(AudioMoth_closeFile());
    uint32_t dummy;
    CHECK(!Sched_ParseConfig(again, (uint32_t)strlen(again), &dummy));
    /* 2. what rename would have done (the reason it is not used) */
    CHECK(AudioMoth_openFile("OLD.TXT") && AudioMoth_writeToFile("x", 1) && AudioMoth_closeFile());
    CHECK(!AudioMoth_renameFile("CONFIG.TXT", "OLD.TXT"));   /* FR_EXIST */
    /* 3. log lines: open/append/write/close for every window, several sessions */
    float logits[SCHED_CLASSES] = {0.1f, 2.0f, -1.0f};
    for (int i = 0; i < 120; ++i) {
        char line[SCHED_LOG_LINE_MAX];
        uint32_t n = Sched_FormatLogLine(line, seconds + 90 + (uint32_t)i * 6, logits);
        CHECK(AudioMoth_appendFile("LOG.TXT"));
        CHECK(AudioMoth_writeToFile(line, (uint16_t)n));
        CHECK(AudioMoth_closeFile());
    }
    /* 4. a session as RecordSession() + ClassifyRecording() do it: folder, WAV with provisional header, REC state,
     *    300 s in 4 KB writes with a sync every 40, final header, CLS state, windows read back by seek, summary, DONE */
    SchedState st = {SCHED_STATE_REC, {0}, seconds + 90, 0, 0, 0, 0};
    Sched_WavPath(st.start, st.path);
    char folder[9];
    memcpy(folder, st.path, 8); folder[8] = 0;
    CHECK(!AudioMoth_doesDirectoryExist(folder));
    CHECK(AudioMoth_makeDirectory(folder));
    CHECK(AudioMoth_doesDirectoryExist(folder));
    const uint32_t wanted = 300u * SCHED_SAMPLE_RATE;
    uint8_t header[SCHED_WAV_HEADER];
    Sched_WavHeader(header, 0);
    CHECK(AudioMoth_openFile(st.path) && AudioMoth_writeToFile(header, SCHED_WAV_HEADER) && AudioMoth_closeFile());
    char stext[SCHED_STATE_RECORD];
    uint32_t n = Sched_FormatState(&st, stext);
    /* SaveState(): STATE.TXT overwritten in place (appendFile + seek 0), never truncated */
#define SAVE_STATE() CHECK(AudioMoth_appendFile("STATE.TXT") && AudioMoth_seekInFile(0) && \
                           AudioMoth_writeToFile(stext, (uint16_t)n) && AudioMoth_closeFile())
    SAVE_STATE();
    CHECK(AudioMoth_appendFile(st.path));
    CHECK(AudioMoth_getFileSize() == SCHED_WAV_HEADER);
    static int16_t chunk[2048];
    uint32_t written = 0, chunks = 0;
    while (written < wanted) {
        for (int i = 0; i < 2048; ++i) chunk[i] = (int16_t)((written + (uint32_t)i) * 7u);
        const uint32_t k = wanted - written < 2048 ? wanted - written : 2048;
        CHECK(AudioMoth_writeToFile(chunk, (uint16_t)(2 * k)));
        written += k;
        if (++chunks % 40 == 0) CHECK(AudioMoth_syncFile());
    }
    Sched_WavHeader(header, written);
    CHECK(AudioMoth_seekInFile(0) && AudioMoth_writeToFile(header, SCHED_WAV_HEADER));
    CHECK(AudioMoth_closeFile());
    st.kind = SCHED_STATE_CLS; st.samples = written;
    n = Sched_FormatState(&st, stext);
    SAVE_STATE();
    /* a cut while STATE.TXT is being rewritten (open + seek done, write not yet on the card): the card is unmounted
     * without closing the file, as a power cut does; the old record must still be there */
    SchedState other = st;
    other.next = 7;
    char otext[SCHED_STATE_RECORD];
    Sched_FormatState(&other, otext);
    CHECK(AudioMoth_appendFile("STATE.TXT") && AudioMoth_seekInFile(0));
    CHECK(AudioMoth_writeToFile(otext, 10));  /* part of the record, still in the file's buffer */
    Shim_unmount();
    CHECK(Shim_mount());
    {
        char cut[SCHED_STATE_RECORD] = {0};
        SchedState back;
        CHECK(AudioMoth_openFileToRead("STATE.TXT"));
        const uint32_t size = AudioMoth_getFileSize();
        CHECK(size == SCHED_STATE_RECORD && AudioMoth_readFile(cut, size) && Sched_ParseState(cut, size, &back));
        CHECK(AudioMoth_closeFile());
        CHECK(back.kind == SCHED_STATE_CLS && back.next == 0 && back.samples == written);
    }
    /* the resume path: size of a recording from the open file */
    CHECK(AudioMoth_appendFile(st.path));
    CHECK(AudioMoth_getFileSize() == SCHED_WAV_HEADER + 2u * wanted);
    /* the header written at the end says how many samples there are */
    uint8_t back_header[SCHED_WAV_HEADER];
    CHECK(AudioMoth_seekInFile(0) && AudioMoth_readFile((char *)back_header, SCHED_WAV_HEADER));
    Sched_WavHeader(header, wanted);
    CHECK(memcmp(back_header, header, SCHED_WAV_HEADER) == 0);
    CHECK(AudioMoth_closeFile());
    /* windows read back as ClassifyRecording() does (seek, 48 reads of 2000 samples), first, middle and last */
    const uint32_t windows = Sched_WindowCount(written);
    const uint32_t which[3] = {0, windows / 2, windows - 1};
    for (int w = 0; w < 3; ++w) {
        const uint32_t first = which[w] * SCHED_HOP_SAMPLES;
        CHECK(AudioMoth_openFileToRead(st.path) && AudioMoth_seekInFile(SCHED_WAV_HEADER + 2u * first));
        int bad = 0;
        for (uint32_t r = 0; r < SCHED_WINDOW_SAMPLES / 2000u; ++r) {
            CHECK(AudioMoth_readFile((char *)chunk, 4000));
            for (uint32_t i = 0; i < 2000u; ++i) bad += chunk[i] != (int16_t)((first + r * 2000u + i) * 7u);
        }
        CHECK(bad == 0);
        CHECK(AudioMoth_closeFile());
    }
    /* the state file read back as LoadState() does */
    CHECK(AudioMoth_openFileToRead("STATE.TXT"));
    const uint32_t size = AudioMoth_getFileSize();
    char back[SCHED_STATE_RECORD] = {0};
    SchedState parsed;
    CHECK(size == SCHED_STATE_RECORD && AudioMoth_readFile(back, size) && Sched_ParseState(back, size, &parsed));
    CHECK(AudioMoth_closeFile());
    CHECK(parsed.kind == SCHED_STATE_CLS && parsed.samples == written && !strcmp(parsed.path, st.path));
    char sline[SCHED_SESSION_LINE_MAX];
    n = Sched_FormatSession(sline, &st, windows, 600);
    CHECK(AudioMoth_appendFile("SESSIONS.TXT") && AudioMoth_writeToFile(sline, (uint16_t)n) && AudioMoth_closeFile());
    const SchedState done = {SCHED_STATE_DONE, {0}, 0, 0, 0, 0, 0};
    n = Sched_FormatState(&done, stext);
    SAVE_STATE();
    /* reading past the end of the recording: FR_OK with nothing read (why the firmware measures files instead) */
    CHECK(AudioMoth_openFileToRead(st.path) && AudioMoth_seekInFile(SCHED_WAV_HEADER + 2u * wanted));
    memset(chunk, 0x5a, sizeof chunk);
    CHECK(AudioMoth_readFile((char *)chunk, 4000));
    CHECK(chunk[0] == 0x5a5a && chunk[1999] == 0x5a5a);
    CHECK(AudioMoth_closeFile());
    printf("recording %s: %u samples, %u windows\n", st.path, written, windows);
    Shim_unmount();
    printf(fails ? "FS TEST FAILED (%d)\n" : "FS TEST PASSED\n", fails);
    return fails != 0;
}
