// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define QSP_BENCHMARK_NAME "qwen_attention_prefill_output_gate"
#define QSP_OPERATION 7
#define QSP_LOGICAL_ROWS 2048
#define QSP_LOGICAL_COLS 6144
#define QSP_ACTIVE_CORES 128

#include "../qwen_spatz_op_main.h"
