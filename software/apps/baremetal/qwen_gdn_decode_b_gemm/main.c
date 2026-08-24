// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define QWEN_BENCHMARK_NAME "qwen_gdn_decode_b_gemm"
#define QWEN_LOGICAL_M 32
#define QWEN_LOGICAL_N 5120
#define QWEN_LOGICAL_P 48
#define QWEN_TILE_N 5120
#define QWEN_DECODE 1
#define QWEN_OUTPUT_CLASS 1

#include "../qwen_redmule_op_main.h"
