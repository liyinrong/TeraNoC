// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __QWEN_SPATZ_OP_MAIN_H__
#define __QWEN_SPATZ_OP_MAIN_H__

#include <stdint.h>

#include "printf.h"
#include "runtime.h"
#include "synchronization.h"

#include "baremetal/mempool_qwen_spatz_ops.h"

#define QSP_OP_RMSNORM 1
#define QSP_OP_SILU_MUL 2
#define QSP_OP_ADD 3
#define QSP_OP_ROPE 4
#define QSP_OP_SCALE_MASK 5
#define QSP_OP_SOFTMAX 6
#define QSP_OP_SIGMOID_MUL 7
#define QSP_OP_CONV1D 8
#define QSP_OP_SIGMOID 9
#define QSP_OP_SOFTPLUS 10
#define QSP_OP_DELTA_RULE 11
#define QSP_OP_GATED_RMSNORM 12

#ifndef QSP_BENCHMARK_NAME
#error "QSP_BENCHMARK_NAME must be defined"
#endif
#ifndef QSP_OPERATION
#error "QSP_OPERATION must be defined"
#endif
#ifndef QSP_LOGICAL_ROWS
#define QSP_LOGICAL_ROWS 1
#endif
#ifndef QSP_LOGICAL_COLS
#error "QSP_LOGICAL_COLS must be defined"
#endif
#ifndef QSP_ACTIVE_CORES
#define QSP_ACTIVE_CORES NUM_CORES
#endif

#if VLEN == 512
#define QSP_LOCAL_ROUNDS 8
#define QSP_BASE_LMUL 2
#define QSP_BASE_UNROLL 8
#define QSP_SCHEDULE_NAME "local_m2_u8"
#elif VLEN == 1024
#define QSP_LOCAL_ROUNDS 16
#define QSP_BASE_LMUL 1
#define QSP_BASE_UNROLL 16
#define QSP_SCHEDULE_NAME "local_m1_u16"
#else
#error "Qwen Spatz apps support VLEN 512 or 1024"
#endif

#define QSP_ROW_BLOCKS (QSP_LOGICAL_COLS / QSP_F16_LOCAL_ELEMS)

_Static_assert((QSP_OPERATION != QSP_OP_RMSNORM &&
                QSP_OPERATION != QSP_OP_SOFTMAX &&
                QSP_OPERATION != QSP_OP_GATED_RMSNORM) ||
                   QSP_LOGICAL_COLS % QSP_F16_LOCAL_ELEMS == 0,
               "row reductions must divide the local strip");
_Static_assert(QSP_ACTIVE_CORES > 0 && QSP_ACTIVE_CORES <= NUM_CORES,
               "invalid Qwen Spatz active core count");

#if QSP_OPERATION == QSP_OP_RMSNORM
static __fp16 qsp_in0[QSP_ROW_BLOCKS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_out[QSP_ROW_BLOCKS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
#elif QSP_OPERATION == QSP_OP_SOFTMAX
static __fp16 qsp_in0[QSP_ROW_BLOCKS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
#elif QSP_OPERATION == QSP_OP_GATED_RMSNORM
static __fp16 qsp_in0[QSP_ROW_BLOCKS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_in1[QSP_ROW_BLOCKS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_out[QSP_ROW_BLOCKS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
#elif QSP_OPERATION == QSP_OP_ROPE
static __fp16 qsp_in0[QSP_LOCAL_ROUNDS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_in1[QSP_LOCAL_ROUNDS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_out[QSP_LOCAL_ROUNDS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
#elif QSP_OPERATION == QSP_OP_CONV1D
#define QSP_CONV_OUTPUT_STEPS ((QSP_LOGICAL_ROWS > 1) ? 4 : 1)
static __fp16 qsp_in0[(QSP_CONV_OUTPUT_STEPS + 3) * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_out[QSP_CONV_OUTPUT_STEPS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
#elif QSP_OPERATION == QSP_OP_DELTA_RULE
static float qsp_state[128 * QSP_F32_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static float qsp_delta_out[QSP_F32_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
#else
static __fp16 qsp_in0[QSP_LOCAL_ROUNDS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_in1[QSP_LOCAL_ROUNDS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
static __fp16 qsp_out[QSP_LOCAL_ROUNDS * QSP_F16_ROW_ELEMS]
    __attribute__((aligned(4 * NUM_BANKS), section(".l1_prio")));
#endif

int main() {
  const uint32_t core_id = mempool_get_core_id();
  const uint32_t num_cores = mempool_get_core_count();
  const uint32_t active = core_id < QSP_ACTIVE_CORES;
  mempool_barrier_init(core_id);
  mempool_barrier(num_cores);

  if (core_id == 0) {
    printf("\n[%s] logical=%dx%d vlen=%d active=%d/%d placement=tile_local "
           "sew=16 base_lmul=%d unroll=%d schedule=%s\n",
           QSP_BENCHMARK_NAME, QSP_LOGICAL_ROWS, QSP_LOGICAL_COLS, VLEN,
           QSP_ACTIVE_CORES, NUM_CORES, QSP_BASE_LMUL, QSP_BASE_UNROLL,
           QSP_SCHEDULE_NAME);
#if QSP_OPERATION == QSP_OP_RMSNORM || QSP_OPERATION == QSP_OP_SOFTMAX ||      \
    QSP_OPERATION == QSP_OP_GATED_RMSNORM
    printf("[%s] physical_rows=%d physical_cols=%d storage=f16 "
           "mode=perf_only_uninitialized\n",
           QSP_BENCHMARK_NAME, QSP_ACTIVE_CORES, QSP_LOGICAL_COLS);
#elif QSP_OPERATION == QSP_OP_CONV1D
    printf("[%s] physical_rows=%d physical_cols=%d kernel=4 storage=f16 "
           "mode=perf_only_uninitialized\n",
           QSP_BENCHMARK_NAME, QSP_CONV_OUTPUT_STEPS,
           QSP_ACTIVE_CORES * QSP_F16_LOCAL_ELEMS);
#elif QSP_OPERATION == QSP_OP_DELTA_RULE
    printf("[%s] physical_owners=%d state_per_owner=128x%d storage=f32 "
           "mode=perf_only_uninitialized\n",
           QSP_BENCHMARK_NAME, QSP_ACTIVE_CORES, QSP_F32_LOCAL_ELEMS);
#elif QSP_OPERATION == QSP_OP_ROPE
    printf("[%s] physical_rotations=%d physical_width=64 storage=f16 "
           "mode=perf_only_uninitialized\n",
           QSP_BENCHMARK_NAME, QSP_ACTIVE_CORES * QSP_LOCAL_ROUNDS);
#else
    printf("[%s] physical_rows=%d physical_cols=%d storage=f16 "
           "mode=perf_only_uninitialized\n",
           QSP_BENCHMARK_NAME, QSP_ACTIVE_CORES,
           QSP_LOCAL_ROUNDS * QSP_F16_LOCAL_ELEMS);
#endif
  }

  const uint32_t time_init = mempool_get_timer();
  mempool_start_benchmark();
  if (active) {
#if QSP_OPERATION == QSP_OP_RMSNORM
    qsp_rmsnorm_local_f16(qsp_in0, qsp_out, core_id, QSP_LOGICAL_COLS);
#elif QSP_OPERATION == QSP_OP_SILU_MUL
    qsp_silu_mul_local_f16(qsp_in0, qsp_in1, qsp_out, core_id,
                           QSP_LOCAL_ROUNDS);
#elif QSP_OPERATION == QSP_OP_ADD
    qsp_add_local_f16(qsp_in0, qsp_in1, qsp_out, core_id);
#elif QSP_OPERATION == QSP_OP_ROPE
    qsp_rope_local_f16(qsp_in0, qsp_in1, qsp_out, core_id, QSP_LOCAL_ROUNDS);
#elif QSP_OPERATION == QSP_OP_SCALE_MASK
    qsp_scale_mask_local_f16(qsp_in0, core_id, QSP_LOCAL_ROUNDS);
#elif QSP_OPERATION == QSP_OP_SOFTMAX
    qsp_softmax_local_f16(qsp_in0, core_id, QSP_LOGICAL_COLS);
#elif QSP_OPERATION == QSP_OP_SIGMOID_MUL
    qsp_sigmoid_local_f16(qsp_in0, qsp_in1, qsp_out, core_id, QSP_LOCAL_ROUNDS);
#elif QSP_OPERATION == QSP_OP_CONV1D
    qsp_conv4_local_f16(qsp_in0, qsp_out, core_id, QSP_CONV_OUTPUT_STEPS);
#elif QSP_OPERATION == QSP_OP_SIGMOID
    qsp_sigmoid_local_f16(qsp_in0, 0, qsp_out, core_id, QSP_LOCAL_ROUNDS);
#elif QSP_OPERATION == QSP_OP_SOFTPLUS
    qsp_softplus_local_f16(qsp_in0, qsp_out, core_id, QSP_LOCAL_ROUNDS);
#elif QSP_OPERATION == QSP_OP_DELTA_RULE
    qsp_delta_rule_local_f32(qsp_state, qsp_delta_out, core_id);
#elif QSP_OPERATION == QSP_OP_GATED_RMSNORM
    qsp_rmsnorm_local_f16(qsp_in0, qsp_out, core_id, QSP_LOGICAL_COLS);
    for (uint32_t d = 0; d < QSP_ROW_BLOCKS; ++d)
      llm_silu_mul_f16(qsp_f16_local_const(qsp_in1, core_id, d),
                       qsp_f16_local_const(qsp_out, core_id, d),
                       qsp_f16_local(qsp_out, core_id, d), QSP_F16_LOCAL_ELEMS,
                       0, 1);
#else
#error "unsupported Qwen Spatz operation"
#endif
  }
  mempool_log_barrier(2, core_id);
  mempool_stop_benchmark();
  const uint32_t time_end = mempool_get_timer();

  if (core_id == 0) {
    printf("[%s] stage=spatz cycles=%u\n", QSP_BENCHMARK_NAME,
           time_end - time_init);
    printf("[%s] kernel takes %u clock cycles\n", QSP_BENCHMARK_NAME,
           time_end - time_init);
  }
  mempool_barrier(num_cores);
  return 0;
}

#endif // __QWEN_SPATZ_OP_MAIN_H__
