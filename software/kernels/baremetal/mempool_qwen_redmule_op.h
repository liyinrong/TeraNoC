// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __MEMPOOL_QWEN_REDMULE_OP_H__
#define __MEMPOOL_QWEN_REDMULE_OP_H__

#include <stdint.h>

#include "baremetal/mempool_llm_attention_f16.h"

// Execute one dense, packed 2-D GEMM tile. X is shared by output-column
// engines in one row group; W is packed by column group; Z is packed by engine
// id. Setup/clear and result DMA remain outside the timed interval in the app.
// One engine's job from already-resolved operand bases. Split out so a caller
// that places an operand somewhere other than the flat global buffer -- a
// virtual-cluster window, say -- can compute its own bases without this
// header knowing anything about the placement.
static inline void qwen_redmule_op_engine_f16(const __fp16 *xe, const __fp16 *we,
                                              __fp16 *ze,
                                              uint32_t rows_per_engine,
                                              uint32_t tile_n,
                                              uint32_t cols_per_engine) {
  llm_attention_redmule_job((unsigned int)xe, (unsigned int)we,
                            (unsigned int)ze, (uint16_t)rows_per_engine,
                            (uint16_t)tile_n, (uint16_t)cols_per_engine, GEMM);
}

static inline void
qwen_redmule_op_tile_f16(const __fp16 *x, const __fp16 *w, __fp16 *z,
                         uint32_t tile_n, uint32_t rows_per_engine,
                         uint32_t cols_per_engine, uint32_t ne_c,
                         uint32_t active_engines, uint32_t redmule_id) {
  if (redmule_id >= active_engines)
    return;

  const uint32_t row_group = redmule_id / ne_c;
  const uint32_t col_group = redmule_id % ne_c;

  qwen_redmule_op_engine_f16(x + row_group * rows_per_engine * tile_n,
                             w + col_group * tile_n * cols_per_engine,
                             z + redmule_id * rows_per_engine * cols_per_engine,
                             rows_per_engine, tile_n, cols_per_engine);
}

#endif // __MEMPOOL_QWEN_REDMULE_OP_H__
