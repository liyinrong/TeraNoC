// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Virtual-cluster placement variant of ffn_prefill_gate_gemm. Same logical
// shape and same engine budget as the baseline; the placement policy comes from
// APP_DEFINES so one binary directory covers the whole DSE ladder.

#define QWEN_BENCHMARK_NAME "qwen_vc_ffn_prefill_gate_gemm"
#define QWEN_LOGICAL_M 2048
#define QWEN_LOGICAL_N 5120
#define QWEN_LOGICAL_P 17408
#define QWEN_TILE_N 5120
#define QWEN_DECODE 0
#define QWEN_OUTPUT_CLASS 0

#include "../qwen_vc_redmule_op_main.h"
