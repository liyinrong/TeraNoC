// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich
//         Matheus Cavalcante, ETH Zurich

#pragma once
#include "addrmap.h"
#include "alloc.h"
#include "encoding.h"
#include <stddef.h>
#include <stdint.h>

#define NUM_BANKS_PER_TILE (NUM_CORES_PER_TILE * BANKING_FACTOR * N_FU)

extern char l1_alloc_base;
static uint32_t volatile *wake_up_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_REG_OFFSET);
static uint32_t volatile *wake_up_group_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_GROUP_REG_OFFSET);

static uint32_t volatile *wake_up_tile_g0_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_0_REG_OFFSET);
static uint32_t volatile *wake_up_tile_g1_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_1_REG_OFFSET);
static uint32_t volatile *wake_up_tile_g2_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_2_REG_OFFSET);
static uint32_t volatile *wake_up_tile_g3_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_3_REG_OFFSET);
static uint32_t volatile *wake_up_tile_g4_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_4_REG_OFFSET);
static uint32_t volatile *wake_up_tile_g5_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_5_REG_OFFSET);
static uint32_t volatile *wake_up_tile_g6_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_6_REG_OFFSET);
static uint32_t volatile *wake_up_tile_g7_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_TILE_7_REG_OFFSET);

static uint32_t volatile *wake_up_stride_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_STRD_REG_OFFSET);
static uint32_t volatile *wake_up_offset_reg =
    (uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                          CONTROL_REGISTERS_WAKE_UP_OFFST_REG_OFFSET);

typedef uint32_t mempool_id_t;
typedef uint32_t mempool_timer_t;

/// Obtain the number of cores in the current cluster.
static inline mempool_id_t mempool_get_core_count() { return NUM_CORES; }

/// Obtain the ID of the current core (within its domain, addrmap.h).
static inline mempool_id_t mempool_get_core_id() {
  mempool_id_t r;
  asm volatile("csrr %0, mhartid" : "=r"(r));
#if DOMAINS > 1
  return r % NUM_CORES;
#else
  return r;
#endif
}

/// Obtain the domain of the current core: 0 unless built with `domains`.
static inline uint32_t mempool_get_domain_id() {
#if DOMAINS > 1
  mempool_id_t r;
  asm volatile("csrr %0, mhartid" : "=r"(r));
  return r / NUM_CORES;
#else
  return 0u;
#endif
}

/// Obtain the number of tiles in the current cluster.
static inline uint32_t mempool_get_tile_count() {
  return NUM_CORES / NUM_CORES_PER_TILE;
}

/// Obtain the ID of the tile the current core is in.
static inline uint32_t mempool_get_tile_id() {
  return mempool_get_core_id() / NUM_CORES_PER_TILE;
}

/// Obtain the number of groups in the current cluster.
static inline uint32_t mempool_get_group_count() { return NUM_GROUPS; }

/// Obtain the ID of the group the current core is in.
static inline uint32_t mempool_get_group_id() {
  return mempool_get_core_id() / (NUM_CORES / NUM_GROUPS);
}

/// Return whether the current core owns its group's control-plane work.
static inline uint32_t mempool_is_group_coordinator() {
  return mempool_get_core_id() % (uint32_t)NUM_CORES_PER_GROUP == 0u;
}

/// Return whether the current core owns its tile-local engine controls.
static inline uint32_t mempool_is_tile_agent() {
  return mempool_get_core_id() % (uint32_t)NUM_CORES_PER_TILE == 0u;
}

/// Return whether the current tile instantiates a RedMulE engine.
static inline uint32_t mempool_tile_has_redmule() {
  if (NUM_REDMULE_TILES == 0)
    return 0u;
  const uint32_t tiles_per_group =
      (uint32_t)NUM_CORES_PER_GROUP / (uint32_t)NUM_CORES_PER_TILE;
  const uint32_t redmules_per_group =
      (uint32_t)NUM_REDMULE_TILES / (uint32_t)NUM_GROUPS;
  return (uint32_t)(mempool_get_tile_id() % tiles_per_group <
                    redmules_per_group);
}

/// Obtain the number of cores per tile in the current cluster
static inline uint32_t mempool_get_core_count_per_tile() {
  return NUM_CORES_PER_TILE;
}

/// Obtain the number of cores per group in the current cluster
static inline uint32_t mempool_get_core_count_per_group() {
  return NUM_CORES / NUM_GROUPS;
}

/// Obtain the number of RedMulEs in the current cluster.
static inline uint32_t mempool_get_redmule_count() { return NUM_REDMULE_TILES; }

/// Obtain the ID of the group the current core is in.
static inline uint32_t mempool_get_redmule_id() {
  if (NUM_REDMULE_TILES == 0) {
    return (uint32_t)(-1);
  } else {
    if (!mempool_is_tile_agent() || !mempool_tile_has_redmule())
      return (uint32_t)(-1);
    const uint32_t tiles_per_group =
        (uint32_t)NUM_CORES_PER_GROUP / (uint32_t)NUM_CORES_PER_TILE;
    const uint32_t redmules_per_group =
        (uint32_t)NUM_REDMULE_TILES / (uint32_t)NUM_GROUPS;
    const uint32_t local_tile = mempool_get_tile_id() % tiles_per_group;
    return mempool_get_group_id() * redmules_per_group + local_tile;
  }
}

/// Initialization
static inline void mempool_init(const uint32_t core_id) {
  if (core_id == 0) {
    // Initialize L1 Interleaved Heap Allocator, unless the whole per-core L1 is
    // sequential (SEQ_MEM_SIZE == full bank set), in which case there is no
    // interleaved region. For scalar builds N_FU=1 and the condition is true,
    // so this is unchanged.
    if (SEQ_MEM_SIZE != L1_BANK_SIZE * BANKING_FACTOR * N_FU) {
      extern uint32_t __heap_start, __heap_end;
      uint32_t heap_size = (uint32_t)&__heap_end - (uint32_t)&__heap_start;
      alloc_init(get_alloc_l1(), &__heap_start, heap_size);
    }

    // Initialize L1 Sequential Heap Allocator per Tile
    extern uint32_t __seq_start;
    // The stack is in the sequential region
    uint32_t seq_heap_offset = NUM_CORES_PER_TILE * STACK_SIZE;
    // preceded by the queues (XQUEUE_SIZE in words)
    seq_heap_offset += NUM_BANKS_PER_TILE * XQUEUE_SIZE * sizeof(uint32_t);
    // The total sequential memory per tile in bytes
    uint32_t seq_total_size = NUM_CORES_PER_TILE * SEQ_MEM_SIZE;
    // The base is the start address + the offset due to the queues and stack
    uint32_t seq_heap_base = (uint32_t)&__seq_start + seq_heap_offset;
    uint32_t seq_heap_size = seq_total_size - seq_heap_offset;
    uint32_t num_tiles = mempool_get_tile_count();
    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
      alloc_t *tile_allocator = get_alloc_tile(tile_id);
      alloc_init(tile_allocator, (uint32_t *)seq_heap_base, seq_heap_size);
      seq_heap_base += seq_total_size;
    }
  }
}

/// Reset a monotonically increasing cycle count.
static inline void mempool_start_benchmark() {
  asm volatile("" ::: "memory");
  write_csr(trace, 1);
  asm volatile("" ::: "memory");
}

/// Obtain a monotonically increasing cycle count.
static inline void mempool_stop_benchmark() {
  asm volatile("" ::: "memory");
  write_csr(trace, 0);
  asm volatile("" ::: "memory");
}

/// Obtain a monotonically increasing cycle count.
static inline mempool_timer_t mempool_get_timer() { return read_csr(mcycle); }

/// Busy loop for waiting
static inline void mempool_wait(uint32_t cycles) {
  asm volatile("1: \n\t"
               "addi %[counter], %[counter], -2 \n\t"
               "bgtz %[counter], 1b \n\t"
               : [counter] "+&r"(cycles)
               :
               : "memory");
}

static inline void mempool_wfi() { asm volatile("wfi"); }

#if DOMAINS > 1
// A domain wakes only its own cores: ids and masks are domain-relative, and
// the per-group tile registers (one per physical group, all 64 of them) reach
// every group, unlike the 32-bit group mask. A tile mask with bits beyond the
// group's tiles is ignored (ctrl_registers), so "all tiles" is the exact mask.
static inline void domain_wake_group(uint32_t group, uint32_t tile_mask) {
  const uint32_t g = mempool_get_domain_id() * NUM_GROUPS + group;
  *(uint32_t volatile *)(CONTROL_REGISTER_OFFSET +
                         CONTROL_REGISTERS_WAKE_UP_TILE_0_REG_OFFSET + 4u * g) =
      tile_mask;
}
static inline void wake_up_group(uint32_t group_mask) {
  for (uint32_t g = 0; g < NUM_GROUPS; ++g)
    if (group_mask >> g & 1u)
      domain_wake_group(g, (1u << NUM_TILES_PER_GROUP) - 1u); // exact mask
}
static inline void wake_up(uint32_t core_id) {
  if (core_id == (uint32_t)-1)
    wake_up_group((uint32_t)-1);
  else
    *wake_up_reg = mempool_get_domain_id() * NUM_CORES + core_id;
}
static inline void wake_up_all() { wake_up((uint32_t)-1); }
static inline void wake_up_all_group() { wake_up_group((uint32_t)-1); }
static inline void wake_up_tile(uint32_t group_id, uint32_t tile_mask) {
  domain_wake_group(group_id, tile_mask);
}
#else
// Wake up core with given core_id by writing in the wake up control register.
// If core_id equals -1, wake up all cores.
static inline void wake_up(uint32_t core_id) { *wake_up_reg = core_id; }
static inline void wake_up_all() { wake_up((uint32_t)-1); }
static inline void wake_up_group(uint32_t group_mask) {
  *wake_up_group_reg = group_mask;
}
static inline void wake_up_all_group() { wake_up_group((uint32_t)-1); }

static inline void wake_up_tile(uint32_t group_id, uint32_t tile_mask) {

  switch (group_id) {
  case 0:
    *wake_up_tile_g0_reg = tile_mask;
    break;
  case 1:
    *wake_up_tile_g1_reg = tile_mask;
    break;
  case 2:
    *wake_up_tile_g2_reg = tile_mask;
    break;
  case 3:
    *wake_up_tile_g3_reg = tile_mask;
    break;
  case 4:
    *wake_up_tile_g4_reg = tile_mask;
    break;
  case 5:
    *wake_up_tile_g5_reg = tile_mask;
    break;
  case 6:
    *wake_up_tile_g6_reg = tile_mask;
    break;
  case 7:
    *wake_up_tile_g7_reg = tile_mask;
    break;
  default:
    *wake_up_tile_g0_reg = tile_mask;
    break;
  }
}
#endif // DOMAINS > 1

static inline void set_wake_up_stride(uint32_t stride) {
  *wake_up_stride_reg = stride;
}
static inline void set_wake_up_offset(uint32_t offset) {
  *wake_up_offset_reg = offset;
}
// Dump a value via CSR
// This is only supported in simulation and an experimental feature. All writes
// to unimplemented CSR registers will be dumped by Snitch. This can be
// exploited to quickly print measurement values from all cores simultaneously
// without the hassle of printf. To specify multiple metrics, different CSRs can
// be used.
// The macro will define a function that will then always print via the same
// CSR. E.g., `dump(errors, 8)` will define a function with the following
// signature: `dump_errors(uint32_t val)`, which will print the given value via
// the 8th register.
// Alternatively, the `write_csr(reg, val)` macro can be used directly.
#define dump(name, reg)                                                        \
  static                                                                       \
      __attribute__((always_inline)) inline void dump_##name(uint32_t val) {   \
    asm volatile("csrw " #reg ", %0" ::"rK"(val));                             \
  }
