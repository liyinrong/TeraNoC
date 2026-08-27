// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Virtual-cluster placement variant of gdn_prefill_a_gemm. Same logical shape
// and engine budget as the baseline; the placement policy arrives via
// APP_DEFINES so one directory covers the whole DSE ladder.

#define QWEN_BENCHMARK_NAME "qwen_vc_gdn_prefill_a_gemm"
#define QWEN_LOGICAL_M 2048
#define QWEN_LOGICAL_N 5120
#define QWEN_LOGICAL_P 48
#define QWEN_TILE_N 5120
#define QWEN_DECODE 0
#define QWEN_OUTPUT_CLASS 1

#include "../qwen_vc_redmule_op_main.h"
