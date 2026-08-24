// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define QSP_BENCHMARK_NAME "qwen_ffn_decode_silu_mul"
#define QSP_OPERATION 2
#define QSP_LOGICAL_ROWS 32
#define QSP_LOGICAL_COLS 17408
#define QSP_ACTIVE_CORES 128

#include "../qwen_spatz_op_main.h"
