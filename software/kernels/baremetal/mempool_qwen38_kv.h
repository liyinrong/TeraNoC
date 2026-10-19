// Copyright 2026 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "baremetal/mempool_qwen38_vector.h"
#include "dma.h"
#include "runtime.h"
#include "synchronization.h"
#if VIRTUAL_CLUSTER
#include "vc.h"
#endif

// Persist [key-block][K-transposed heads | V-row-major heads] to L2.
// All cores participate. Temporary storage is reused only after its DMA lands.
static inline void q38_pack_kv_to_l2(const __fp16 *head_major,
                                     __fp16 *temporary, __fp16 *l2_cache,
                                     uint32_t rows, uint32_t heads,
                                     uint32_t dimension, uint32_t block) {
  if (rows % block != 0u)
    __builtin_trap();
  const uint32_t core = mempool_get_core_id();
  const uint32_t half = heads * dimension * block;
  for (uint32_t key = 0; key < rows; key += block) {
    for (uint32_t i = core; i < heads * dimension; i += NUM_CORES) {
      const uint32_t h = i / dimension, d = i % dimension;
      for (uint32_t offset = 0; offset < block;) {
        uint32_t vl;
        asm volatile("vsetvli %0, %1, e16, m2, ta, ma"
                     : "=r"(vl)
                     : "r"(block - offset));
        asm volatile(
            "vlse16.v v0, (%0), %1"
            :
            : "r"(head_major + (h * rows + key + offset) * dimension + d),
              "r"(dimension * 2)
            : "memory");
        asm volatile("vse16.v v0, (%0)"
                     :
                     : "r"(temporary + i * block + offset)
                     : "memory");
        offset += vl;
      }
    }
    for (uint32_t i = core; i < heads * block; i += NUM_CORES) {
      const uint32_t h = i / block, t = i % block;
      q38_copy16(temporary + half + i * dimension,
                 head_major + ((heads + h) * rows + key + t) * dimension,
                 dimension);
    }
    __sync_synchronize();
    mempool_barrier(NUM_CORES);
    if (dma_collective_memcpy_blocking(l2_cache + key / block * 2 * half,
                                       temporary, 4 * half) != 0u)
      __builtin_trap();
  }
}

#if VIRTUAL_CLUSTER
static inline __fp16 *q38_allocate_kv_replica(uint32_t level, uint32_t bytes) {
  vc_init();
  if (level >= VC_LEVELS || bytes % vc_row_stride(level) != 0u)
    __builtin_trap();
  if (vc_reserve(level, bytes) != 0)
    __builtin_trap();
  __fp16 *result = (__fp16 *)vc_malloc(level, bytes);
  if (result == 0)
    __builtin_trap();
  return result;
}

// This is deliberately an ordinary DMA read per VC, not multicast.
static inline void q38_load_kv_replica(__fp16 *replica, const __fp16 *l2_tile,
                                       uint32_t bytes) {
  if (vc_dma_memcpy_auto_blocking(replica, l2_tile, bytes) != 0u)
    __builtin_trap();
}
#endif
