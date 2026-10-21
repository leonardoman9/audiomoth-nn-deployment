/* Runs the application's SD-card call sequences on the real FatFS (with the app's trimmed ffconf.h). */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../schedule.h"

/* the library wrappers under test */
bool AudioMoth_doesFileExist(char *filename);
bool AudioMoth_openFile(char *filename);
bool AudioMoth_appendFile(char *filename);
bool AudioMoth_openFileToRead(char *filename);
bool AudioMoth_readFile(char *buffer, uint32_t bufferSize);
bool AudioMoth_writeToFile(void *bytes, uint16_t bytesToWrite);
bool AudioMoth_renameFile(char *a, char *b);
bool AudioMoth_closeFile(void);
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
    float logits[9] = {0.1f, 2.0f, -1, 0, 0, 0, 0, 0, 0.5f};
    for (int i = 0; i < 120; ++i) {
        char line[64];
        uint32_t n = Sched_FormatLogLine(line, seconds + 90 + (uint32_t)i * 6, logits);
        CHECK(AudioMoth_appendFile("LOG.TXT"));
        CHECK(AudioMoth_writeToFile(line, (uint16_t)n));
        CHECK(AudioMoth_closeFile());
    }
    Shim_unmount();
    printf(fails ? "FS TEST FAILED (%d)\n" : "FS TEST PASSED\n", fails);
    return fails != 0;
}
