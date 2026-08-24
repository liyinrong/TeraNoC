// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#define QSP_BENCHMARK_NAME "qwen_gdn_prefill_qk_l2norm"
#define QSP_OPERATION 1
#define QSP_LOGICAL_ROWS 2048
#define QSP_LOGICAL_COLS 128
#define QSP_ACTIVE_CORES 128

#include "../qwen_spatz_op_main.h"
