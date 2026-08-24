// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "runtime.h"

// Detailed markers are intended for the numerically checked toy shapes. Real
// performance-only shapes can execute a stage thousands of times, so keep
// their original single whole-kernel trace region unless explicitly enabled.
#ifndef LLM_CLASSIC_STAGE_PERF
#if defined(LLM_PERF_ONLY) && LLM_PERF_ONLY
#define LLM_CLASSIC_STAGE_PERF 0
#else
#define LLM_CLASSIC_STAGE_PERF 1
#endif
#endif

// The runtime exposes the low 32 bits of mcycle. Toy intervals and their
// per-stage aggregates must remain below 2^32 cycles.
typedef uint32_t llm_stage_cycles_t;

// Keep trace-marker overhead out of the reported stage cycles. The two mcycle
// reads remain visible section boundaries to the trace postprocessor.
static inline mempool_timer_t llm_stage_timer_start(void) {
#if LLM_CLASSIC_STAGE_PERF
  mempool_start_benchmark();
  const mempool_timer_t start = mempool_get_timer();
  asm volatile("" ::: "memory");
  return start;
#else
  return 0;
#endif
}

static inline void llm_stage_timer_stop(mempool_timer_t start,
                                        llm_stage_cycles_t *cycles) {
#if LLM_CLASSIC_STAGE_PERF
  asm volatile("" ::: "memory");
  const mempool_timer_t end = mempool_get_timer();
  const mempool_timer_t elapsed = end - start;
  mempool_stop_benchmark();
  *cycles += elapsed;
#else
  (void)start;
  (void)cycles;
#endif
}
