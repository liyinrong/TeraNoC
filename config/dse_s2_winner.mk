# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Stage-1 DSE **S2 winner** -- the overall best cell of the 1224-cell stage-1
# grid, with the wide plane of S1 plan B. Its GVSoC counterpart is
#
#     teranoc_v2:system_config=dse_s2_winner_plan_b
#
# S2 geometry: H16xW32 RedMulE (PORT_WIDTH = 64 elements = 128 B), 8-KiB banks
# (32 MiB L1), 1024-bit wide TCDM plane, two superbanks per tile.
# Score at the winner cell: 97.4 % active-engine utilization, 35.34 MGE,
# 11,246 of 13,419 pins, FOM 1806 -- the design point.
#
# The wide plane is built per port as in config/dse_s1_winner.mk. This file
# pins the winner's ROB, FIFO and remap settings explicitly rather than
# inheriting them by accident.

config_build_path := dse_s2_winner

# --- wide TCDM plane (WP2) ports ------------------------------------------
# As S1: 2 RO request and 2 RW response ports per tile, one write port pair
# shared by tiles t and t+2.
noc_req_wide_ro_port_num  := 2
noc_req_wide_wo_port_num  := 1
noc_resp_wide_rw_port_num := 2
noc_resp_wide_wo_port_num := 1
noc_wide_wo_merge         := 2

# --- RedMulE outstanding-transaction credit ------------------------------
# Stock 16. S2's lower per-channel offered load (rho_eng 0.352) does not pay
# for the deeper ROB; the grid measured 24 as neutral-to-worse here.
rob_depth := 16

# --- channel remapping ---------------------------------------------------
# Narrow plane: unchanged, group size 4.
noc_router_remapping := 3
noc_router_remap_group_size := 4
# Wide plane: rotate the RO request and RW response ports over tile pairs.
noc_wide_router_remapping := 3
noc_wide_ro_req_remap_group_size  := 2
noc_wide_rw_resp_remap_group_size := 2

# --- router buffering ----------------------------------------------------
# The winner runs wide request and response FIFOs at 2/2, which is the RTL
# default: no FIFO change is needed to realise this cell.
noc_router_input_fifo_dep  := 2
noc_router_output_fifo_dep := 2

# --- Spatz ---------------------------------------------------------------
# VLEN matches the 1024-bit wide plane; unit-stride vector loads issue as
# bursts (one wide word = 32 words: remote as one wide access, own-tile as
# narrow words on the vector ports), with a 32-entry VLSU reorder buffer.
vlen            := 1024
spatz_burst     := 1
spatz_rob_depth := 32

# --- instruction caches ---------------------------------------------------
# As S1 (config/dse_s1_winner.mk); not swept for S2 yet.
icache_size_byte   := 4096
ro_cache_size_byte := 16384

include $(MEMPOOL_DIR)/config/dse_s2_common.mk
