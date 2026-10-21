/* Simulator stand-in for the AudioMoth hardware library header (subset used by main_app.c). */
#ifndef AUDIOMOTH_SIM_H
#define AUDIOMOTH_SIM_H
#include <stdbool.h>
#include <stdint.h>

typedef enum {AM_LOW_GAIN_RANGE, AM_NORMAL_GAIN_RANGE} AM_gainRange_t;
typedef enum {AM_SD_CARD_NORMAL_SPEED, AM_SD_CARD_HIGH_SPEED} AM_sdCardSpeed_t;
typedef enum {AM_GAIN_LOW, AM_GAIN_LOW_MEDIUM, AM_GAIN_MEDIUM, AM_GAIN_MEDIUM_HIGH, AM_GAIN_HIGH} AM_gainSetting_t;

void AudioMoth_initialise(void);
bool AudioMoth_isInitialPowerUp(void);
bool AudioMoth_hasInvertedOutput(void);
bool AudioMoth_enableExternalSRAM(void);
void AudioMoth_disableExternalSRAM(void);
void AudioMoth_startMicrophoneSamples(uint32_t sampleRate);
void AudioMoth_initialiseDirectMemoryAccess(int16_t *primaryBuffer, int16_t *secondaryBuffer, uint16_t numberOfSamples);
bool AudioMoth_enableMicrophone(AM_gainRange_t gainRange, AM_gainSetting_t gainSetting, uint32_t clockDivider, uint32_t acquisitionCycles, uint32_t oversampleRate);
void AudioMoth_disableMicrophone(void);
bool AudioMoth_hasTimeBeenSet(void);
void AudioMoth_setTime(uint32_t time, uint32_t milliseconds);
void AudioMoth_getTime(uint32_t *time, uint32_t *milliseconds);
void AudioMoth_feedWatchdog(void);
void AudioMoth_delay(uint32_t milliseconds);
void AudioMoth_sleep(void);
void AudioMoth_powerDownAndWake(uint32_t seconds, bool synchronised);
void AudioMoth_setRedLED(bool state);
void AudioMoth_setGreenLED(bool state);
bool AudioMoth_enableFileSystem(AM_sdCardSpeed_t speed);
void AudioMoth_disableFileSystem(void);
bool AudioMoth_doesFileExist(char *filename);
bool AudioMoth_openFileToRead(char *filename);
bool AudioMoth_readFile(char *buffer, uint32_t bufferSize);
bool AudioMoth_appendFile(char *filename);
bool AudioMoth_writeToFile(void *bytes, uint16_t bytesToWrite);
bool AudioMoth_openFile(char *filename);
bool AudioMoth_closeFile(void);
bool AudioMoth_seekInFile(uint32_t position);
uint32_t AudioMoth_getFileSize(void);
bool AudioMoth_syncFile(void);
bool AudioMoth_doesDirectoryExist(char *folderName);
bool AudioMoth_makeDirectory(char *folderName);

/* Application callbacks (implemented by main_app.c, as in the real firmware). */
extern void AudioMoth_handleSwitchInterrupt(void);
extern void AudioMoth_handleMicrophoneChangeInterrupt(void);
extern void AudioMoth_handleMicrophoneInterrupt(int16_t sample);
extern void AudioMoth_handleDirectMemoryAccessInterrupt(bool isPrimaryBuffer, int16_t **nextBuffer);
extern void AudioMoth_timezoneRequested(int8_t *timezoneHours, int8_t *timezoneMinutes);

/* Wrappers around the real pipeline that account simulated time and dump each window. */
#include "bird_pipeline.h"
void sim_BirdPipeline_Begin(void);
void sim_BirdPipeline_Push(const int16_t *pcm, uint32_t n);
bool sim_BirdPipeline_Finish(float *logits);
#endif
