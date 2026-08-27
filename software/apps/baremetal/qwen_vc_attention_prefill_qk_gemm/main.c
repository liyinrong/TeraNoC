// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Virtual-cluster placement variant of attention_prefill_qk_gemm. Same logical
// shape and engine budget as the baseline; the placement policy arrives via
// APP_DEFINES so one directory covers the whole DSE ladder.

#define QWEN_BENCHMARK_NAME "qwen_vc_attention_prefill_qk_gemm"
#define QWEN_LOGICAL_M 2048
#define QWEN_LOGICAL_N 256
#define QWEN_LOGICAL_P 2048
#define QWEN_TILE_N 256
#define QWEN_DECODE 0
#define QWEN_OUTPUT_CLASS 0

#include "../qwen_vc_redmule_op_main.h"
