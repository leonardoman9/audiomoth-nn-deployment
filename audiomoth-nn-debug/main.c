/***************************************************************************//**
 * @file main.c
 * @brief main() function with debug mode option.
 *******************************************************************************
 * # License
 * <b>Copyright 2020 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * The licensor of this software is Silicon Laboratories Inc. Your use of this
 * software is governed by the terms of Silicon Labs Master Software License
 * Agreement (MSLA) available at
 * www.silabs.com/about-us/legal/master-software-license-agreement. This
 * software is distributed to you in Source Code format and is governed by the
 * sections of the MSLA applicable to Source Code.
 *
 ******************************************************************************/
#include "sl_component_catalog.h"
#include "sl_system_init.h"
#include "app.h"
#if defined(SL_CATALOG_POWER_MANAGER_PRESENT)
#include "sl_power_manager.h"
#endif
#if defined(SL_CATALOG_KERNEL_PRESENT)
#include "sl_system_kernel.h"
#else // SL_CATALOG_KERNEL_PRESENT
#include "sl_system_process_action.h"
#endif // SL_CATALOG_KERNEL_PRESENT

// Abilita modalità debug per testare il flashing e i breakpoints
#define DEBUG_MODE 1

int main(void)
{
  // Initialize Silicon Labs device, system, service(s) and protocol stack(s).
  sl_system_init();

  // Initialize the application
  app_init();

#if defined(SL_CATALOG_KERNEL_PRESENT)
  // Start the kernel. Task(s) created in app_init() will start running.
  sl_system_kernel_start();
#else // SL_CATALOG_KERNEL_PRESENT
  while (1) {
    // Process system tasks
    sl_system_process_action();

    // Process application tasks
    app_process_action();

#if defined(SL_CATALOG_POWER_MANAGER_PRESENT)
#if !DEBUG_MODE
    // Normal mode: CPU può andare in sleep
    sl_power_manager_sleep();
#else
    // Debug mode: niente sleep + piccolo delay artificiale
    for (volatile int i = 0; i < 100000; i++);
#endif
#endif
  }
#endif // SL_CATALOG_KERNEL_PRESENT
}
