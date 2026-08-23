# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Stage-1 DSE **S2 winner** -- the overall best cell of the 1224-cell stage-1
# grid. Its GVSoC counterpart is exactly
#
#     teranoc_v2:system_config=dse_s2_m4_both_rob16_rq2_rs2
#
# S2 geometry: H16xW32 RedMulE (PORT_WIDTH = 64 elements = 128 B), 8-KiB banks
# (32 MiB L1), 1024-bit wide TCDM plane, two superbanks per tile.
# Score at the winner cell: 97.4 % active-engine utilization, 35.34 MGE,
# 11,246 of 13,419 pins, FOM 1806 -- the design point.
#
# Channel mix is M4: (RO, RWreq, RWresp, WO) = (4, 1, 4, 1) per group, i.e.
# identical to config/dse_s2_m4.mk. This file exists so the winner's ROB, FIFO
# and remap settings are pinned explicitly rather than inherited by accident.

config_build_path := dse_s2_winner

# --- wide TCDM plane (WP2) channel mix: M4 -------------------------------
noc_req_wide_ro_channel_num  := 4
noc_req_wide_rw_channel_num  := 1
noc_resp_wide_rw_channel_num := 4
noc_resp_wide_wo_channel_num := 1

# --- RedMulE outstanding-transaction credit ------------------------------
# Stock 16. S2's lower per-channel offered load (rho_eng 0.352) does not pay
# for the deeper ROB; the grid measured 24 as neutral-to-worse here.
rob_depth := 16

# --- channel remapping ---------------------------------------------------
# Narrow plane: unchanged, group size 4.
noc_router_remapping := 3
noc_router_remap_group_size := 4
# Wide plane: 3 = rotate both aliasing classes (RO request, RW read-data
# response). Without this the group concentrator's `i % N` pins each of the four
# tiles to one channel for the whole run. The batch sizes default to the class
# counts, which is 4 and 4 here -- exactly the model's ro/rw_resp_remap_batch_size.
noc_wide_router_remapping := 3

# --- router buffering ----------------------------------------------------
# The winner runs wide request and response FIFOs at 2/2, which is the RTL
# default: no FIFO change is needed to realise this cell.
noc_router_input_fifo_dep  := 2
noc_router_output_fifo_dep := 2

# =========================================================================
# MODEL/RTL GAPS AT THIS CONFIG -- ALL CLOSED
#
# GAP 1 (wide-plane remapper) -- CLOSED. mempool_group_floonoc_wrapper.sv now
#   instantiates i_floo_tcdm_wide_ro_req_remapper and
#   i_floo_tcdm_wide_rw_resp_remapper on the outgoing edge, gated by
#   noc_wide_router_remapping above. Both are GroupSize 4, Interleaved.
#
# GAP 2/3 (remap group size, per-class FIFO depth) never bound at S2: both
#   rotating classes are 4 wide and every FIFO depth is the default 2. The
#   per-class FIFO knobs now exist regardless (config/config.mk), all defaulting
#   to the global, so this file stays at 2/2 without restating them.
#
# REMAINING: the RedMulE X-shift register moved off the software header's 0x18
#   (which is LEFT_PARAMS in redmule_pkg.sv and is driven by redmule_tiler.sv).
#   See hardware/deps/redmule/sw/archi_redmule.h for the current offset; the
#   GVSoC model must match it.
#
# See gvsoc/docs/manual/stage1-dse-winner-rtl-handoff.md for the full mapping.
# =========================================================================

include $(MEMPOOL_DIR)/config/dse_s2_common.mk
