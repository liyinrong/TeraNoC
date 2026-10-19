// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#define GROUP_ROW (NUM_BANKS / NUM_GROUPS * 4u)
#define GLOBAL_ROW (NUM_BANKS * 4u)
#define PANEL_BYTES 128u
#define PANELS 16u
#define BYTES (PANELS * PANEL_BYTES)
#ifndef BW_REPEATS
#define BW_REPEATS 64u
#endif
#define REPEATS BW_REPEATS
#ifndef BW_TILE_LOCAL
#define BW_TILE_LOCAL 0
#endif
#define LOCAL_ROW (BW_TILE_LOCAL ? GROUP_ROW / NUM_TILES_PER_GROUP : GROUP_ROW)
static uint8_t memory[NUM_GROUPS * NUM_TILES_PER_GROUP * 2u * BYTES]
    __attribute__((section(".l1"), aligned(GLOBAL_ROW)));

// Only core zero runs this probe; every access stays in group zero's stripe.
static void *at(uint32_t byte) {
  return memory + byte / LOCAL_ROW * GLOBAL_ROW + byte % LOCAL_ROW;
}

#define LOAD(reg, i)                                                           \
  asm volatile("vle16.v " reg ", (%0)"                                         \
               :                                                               \
               : "r"(at((i)*PANEL_BYTES))                                      \
               : "memor"                                                       \
                 "y")
#define STORE(reg, i)                                                          \
  asm volatile("vse16.v " reg ", (%0)"                                         \
               :                                                               \
               : "r"(at(BYTES + (i)*PANEL_BYTES))                              \
               : "memory")
#define EACH(OP)                                                               \
  OP("v0", 0);                                                                 \
  OP("v2", 1);                                                                 \
  OP("v4", 2);                                                                 \
  OP("v6", 3);                                                                 \
  OP("v8", 4);                                                                 \
  OP("v10", 5);                                                                \
  OP("v12", 6);                                                                \
  OP("v14", 7);                                                                \
  OP("v16", 8);                                                                \
  OP("v18", 9);                                                                \
  OP("v20", 10);                                                               \
  OP("v22", 11);                                                               \
  OP("v24", 12);                                                               \
  OP("v26", 13);                                                               \
  OP("v28", 14);                                                               \
  OP("v30", 15)
#define COPY(reg, i)                                                           \
  LOAD(reg, i);                                                                \
  STORE(reg, i)
#define READBACK(reg, i) asm volatile("vfmv.f.s %0, " reg : "=f"(sink));

static uint32_t probe(uint32_t mode) {
  float sink;
  asm volatile("vsetvli zero, %0, e16, m2, ta, ma" : : "r"(64u));
  EACH(LOAD);
  EACH(READBACK);
  const uint32_t start = mempool_get_timer();
  for (uint32_t repeat = 0; repeat < REPEATS; ++repeat) {
    if (mode == 0u) {
      EACH(COPY);
    } else if (mode == 1u) {
      EACH(LOAD);
      EACH(STORE);
      // Bound each block's lifetime before reusing all32 registers.
      asm volatile("vle16.v v0, (%0)"
                   :
                   : "r"(at(2u * BYTES - PANEL_BYTES))
                   : "memory");
      asm volatile("vfmv.f.s %0, v0" : "=f"(sink));
    } else if (mode == 2u) {
      EACH(LOAD);
    } else {
      EACH(STORE);
    }
  }
  if (mode == 2u) {
    EACH(READBACK);
  } else {
    // A dependent read after the final store drains the queued VLSU work.
    asm volatile("vle16.v v0, (%0)"
                 :
                 : "r"(at(2u * BYTES - PANEL_BYTES))
                 : "memory");
    asm volatile("vfmv.f.s %0, v0" : "=f"(sink));
  }
  asm volatile("" : : "f"(sink) : "memory");
  return mempool_get_timer() - start;
}

int main(void) {
  const uint32_t core = mempool_get_core_id();
  mempool_barrier_init(core);
  if (!core) {
    for (uint32_t i = 0; i < BYTES / 2u; ++i)
      *(uint16_t *)at(i * 2u) = (uint16_t)(0x3000u + i);
    for (uint32_t mode = 0; mode < 4u; ++mode) {
      const uint32_t cycles = probe(mode);
      uint32_t errors = 0;
      if (mode != 2u)
        for (uint32_t i = 0; i < BYTES / 2u; ++i)
          errors += *(uint16_t *)at(BYTES + i * 2u) != 0x3000u + i;
      printf("[bandwidth_probe] mode=%u bytes=%u repeats=%u cycles=%u "
             "errors=%u ports=%u vlen=%u\n",
             mode, BYTES, REPEATS, cycles, errors, N_FPU, VLEN);
    }
  }
  mempool_barrier(NUM_CORES);
  return 0;
}
