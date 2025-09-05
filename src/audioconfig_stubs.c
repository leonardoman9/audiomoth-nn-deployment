/**
 * @file audioconfig_stubs.c
 * @brief Minimal AudioConfig stubs for AudioMoth firmware
 * 
 * These stubs provide empty implementations of AudioConfig functions
 * that are referenced but not needed for our neural network application.
 */

#include <stdint.h>
#include <stdbool.h>

/**
 * Handle audio configuration event - stub implementation
 */
void AudioConfig_handleAudioConfigurationEvent(void) {
    // Empty stub - no configuration handling needed
}

/**
 * Handle audio configuration packet - stub implementation
 */
void AudioConfig_handleAudioConfigurationPacket(void) {
    // Empty stub - no packet handling needed
}