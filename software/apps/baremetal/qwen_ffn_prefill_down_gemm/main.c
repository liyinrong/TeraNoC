// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define QWEN_BENCHMARK_NAME "qwen_ffn_prefill_down_gemm"
#define QWEN_LOGICAL_M 2048
#define QWEN_LOGICAL_N 17408
#define QWEN_LOGICAL_P 5120
#define QWEN_TILE_N 8192
#define QWEN_DECODE 0
#define QWEN_OUTPUT_CLASS 0

#include "../qwen_redmule_op_main.h"
