# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Stage-1 DSE **S1 winner** -- the area-ranked best S1 cell of the 1224-cell
# stage-1 grid, with the wide plane of plan B. Its GVSoC counterpart is
#
#     teranoc_v2:system_config=dse_s1_winner_plan_b
#
# S1 geometry: H8xW32 RedMulE (PORT_WIDTH = 32 elements = 64 B), 512-bit wide
# TCDM plane. The DSE cell has 64 4-KiB banks per tile; this flavor keeps the
# 16 MiB L1 with 32 8-KiB banks per tile (two superbanks), the same bank size as
# S2, which maps onto one pre-generated GF12 cut (see L1 banks below).
# Score at the winner cell: 92.0 % active-engine utilization, 21.20 MGE,
# 7,668 of 9,379 pins, FOM 1422.
#
# The wide plane is built per port, like the narrow one (no group concentrator):
# each tile has 2 RO request ports and 2 RW response ports of its own, plus one
# write request and one write-ack port shared by two tiles.

config_build_path := dse_s1_winner

# --- L1 banks --------------------------------------------------------------
# 32 banks of 8 KiB per tile instead of the DSE's 64 of 4 KiB: same capacity,
# half the banks. 8 KiB (2048x32) is the S2 bank size; in GF12 both map onto
# rf_sp_uhse_rvt_mvt_1024x64m4b1w1, folded (gf12/sourcecode/tc_sram.sv).
banking_factor := 2
l1_bank_size   := 8192

# --- wide TCDM plane (WP2) ports ------------------------------------------
# Reads hash over the 2 RO ports and answer on the paired RW port; writes take
# the WO port, merged 2:1 across tiles t and t+2. In GVSoC (dev model) this is
# 1.008 x the group concentrator on the S1 suite, worst case 0.983.
# This is plan B; config/dse_s1_winner_plan_{a,c,d}.mk are the alternatives.
noc_req_wide_ro_port_num  := 2
noc_req_wide_wo_port_num  := 1
noc_resp_wide_rw_port_num := 2
noc_resp_wide_wo_port_num := 1
noc_wide_wo_merge         := 2

# --- RedMulE outstanding-transaction credit ------------------------------
# The DSE used nominal 24, but ROB_NW = 1 << idx_width(ROB_DEPTH) made it
# 32 physical entries. State that effective depth directly, as in GVSoC.
rob_depth := 32

# --- Spatz burst ---------------------------------------------------------
# Unit-stride vector loads issue as bursts: remote ones as one wide access,
# own-tile ones as narrow words on the core's vector ports.
spatz_burst := 1

# --- channel remapping ---------------------------------------------------
# Narrow plane: unchanged, group size 4.
noc_router_remapping := 3
noc_router_remap_group_size := 4
# Wide plane: rotate the RO request and RW response ports, each over a pair of
# tiles (t, t+2), so no remapper spans the whole group.
noc_wide_router_remapping := 3
noc_wide_ro_req_remap_group_size  := 2
noc_wide_rw_resp_remap_group_size := 2

# --- router buffering ----------------------------------------------------
# Every router at 2. With two RW response ports per tile the read data no longer
# needs the deeper response FIFOs the concentrator did.
noc_router_input_fifo_dep  := 2
noc_router_output_fifo_dep := 2

# --- cores ----------------------------------------------------------------
# One Snitch-only core per tile for scheduling and RedMulE/DMA control, three
# Snitch-Spatz for the vector work. Banks per tile are unchanged.
num_scalar_cores_per_tile := 1

# --- instruction caches ---------------------------------------------------
# A 4 KiB tile L1 behind 8-line L0s, backed by a 16 KiB group read-only cache:
# the attention loop's code for a tile's four cores does not fit 4 KiB, but the
# group cache holds it. In GVSoC within 1 % of the 8 KiB L1 / 1024-bit lines.
icache_size_byte   := 4096
ro_cache_size_byte := 16384

include $(MEMPOOL_DIR)/config/dse_s1_common.mk
