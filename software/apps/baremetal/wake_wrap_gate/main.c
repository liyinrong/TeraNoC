// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Directed gate for the wake-counter wrap that makes sleeping unsafe.
//
// Snitch banks wakes arriving while a core is AWAKE in a 3-bit counter and
// increments it WITHOUT saturation (snitch.sv:130,138,447), so the eighth
// unconsumed wake discards the other seven. Every sleeping barrier in this
// runtime is safe only because it is credit-neutral -- one wfi spent per wake
// received. This app deliberately breaks that balance to prove two things:
//
//   * the failure is real and reachable, and
//   * hardware/scripts/return_status.sh catches it, because snitch.sv:2987
//     prints "[Missed wake-up Core N]" on exactly this event.
//
// Build with -DWAKE_WRAP_BALANCED to get the safe control: the same number of
// wakes, each spent by a wfi, which must NOT trip the detector.
//
// Expected: default build FAILS the run (detector fires). Balanced build
// passes with retval 0. A default build that passes means the counter is
// saturating -- i.e. the snitch.sv:447 guard of plan section 11.6 has landed,
// and the sleeping barriers no longer need the credit-neutrality argument.

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define OVERFLOW_WAKES 12u // > 8, so the 3-bit counter must wrap

static volatile uint32_t ready __attribute__((section(".l1")));
static volatile uint32_t done __attribute__((section(".l1")));

int main() {
  const uint32_t core = mempool_get_core_id();

  mempool_barrier_init(core);
  if (core == 0u) {
    ready = 0u;
    done = 0u;
  }
  mempool_barrier(NUM_CORES);

  if (core == 1u) {
#ifdef WAKE_WRAP_BALANCED
    // Safe control -- the shape every wait in this runtime uses. Sleeping
    // between pulses means each one is consumed by the wake itself and never
    // banked (snitch.sv:447 increments only while !wfi_q), so no number of
    // pulses can run the counter away.
    __atomic_store_n(&ready, 1u, __ATOMIC_RELEASE);
    do {
      mempool_wfi();
    } while (__atomic_load_n(&done, __ATOMIC_ACQUIRE) == 0u);
#else
    // Stay AWAKE and announce it, so core 0's pulses are banked rather than
    // consumed by an already-sleeping core.
    __atomic_store_n(&ready, 1u, __ATOMIC_RELEASE);
    while (__atomic_load_n(&done, __ATOMIC_ACQUIRE) == 0u)
      mempool_wait(4u);
#endif
    // Nothing to do: the pulses were banked while this core was awake, and
    // the counter has already wrapped.
  } else if (core == 0u) {
    while (__atomic_load_n(&ready, __ATOMIC_ACQUIRE) == 0u)
      mempool_wait(4u);
    // Bank far more wakes on core 1 than its 3-bit counter can hold. Targeted,
    // so no other core's balance is disturbed.
    // Publish the predicate BEFORE the last pulse, never after: a waiter shaped
    // do{wfi}while(!p) spends one pulse per turn, so a predicate that became
    // true only after the final pulse would leave it asleep with nothing left
    // to wake it. Real waiters get this for free -- RedMulE's EOC pulse and its
    // RUNNING_JOB update are the same event.
    for (uint32_t i = 0u; i < OVERFLOW_WAKES; ++i) {
      if (i == OVERFLOW_WAKES - 1u)
        __atomic_store_n(&done, 1u, __ATOMIC_RELEASE);
      wake_up(1u);
      mempool_wait(32u);
    }
  }

  mempool_barrier(NUM_CORES);
  if (core == 0u)
    printf("[wake_wrap_gate] sent=%u\n", (unsigned)OVERFLOW_WAKES);
  mempool_barrier(NUM_CORES);
  return 0;
}
