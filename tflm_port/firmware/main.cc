// AudioMoth boot self-test for the real TFLM inference path.
// Results are left in globals so they can be read with GDB; the LEDs report the
// verdict: 3 green blinks = PASS (every output byte equals the TFLite
// reference), 10 red blinks = FAIL.
extern "C" {
#include "audiomoth.h"
#include "em_cmu.h"
}
#include "selftest.h"

// Callbacks required by the AudioMoth hardware library; this test records nothing.
extern "C" {
void AudioMoth_handleSwitchInterrupt(void) {}
void AudioMoth_handleMicrophoneChangeInterrupt(void) {}
void AudioMoth_handleMicrophoneInterrupt(int16_t) {}
void AudioMoth_handleDirectMemoryAccessInterrupt(bool, int16_t**) {}
}

volatile uint32_t g_result_done = 0;  // 1 when finished
volatile uint32_t g_result_pass = 0;
volatile SelfTestResult g_result;
volatile uint32_t g_core_clock_hz = 0;

static void Blink(bool green, int n) {
  for (int i = 0; i < n; ++i) {
    if (green) AudioMoth_setGreenLED(true); else AudioMoth_setRedLED(true);
    AudioMoth_delay(200);
    AudioMoth_setGreenLED(false);
    AudioMoth_setRedLED(false);
    AudioMoth_delay(200);
    AudioMoth_feedWatchdog();
  }
}

int main(void) {
  AudioMoth_initialise();
  g_core_clock_hz = CMU_ClockFreqGet(cmuClock_CORE);
  SelfTestResult local;
  RunSelfTest(&local);
  g_result = local;
  g_result_pass = local.init_ok && local.mismatches == 0;
  g_result_done = 1;
  for (;;) Blink(g_result_pass, g_result_pass ? 3 : 10);
}
