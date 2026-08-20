# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Stage-1 DSE **S1 winner** -- the area-ranked best S1 cell of the 1224-cell
# stage-1 grid. Its GVSoC counterpart is exactly
#
#     teranoc_v2:system_config=dse_s1_m6_both_rob24_rq2_rs4
#
# S1 geometry: H8xW32 RedMulE (PORT_WIDTH = 32 elements = 64 B), 4-KiB banks
# (16 MiB L1), 512-bit wide TCDM plane, four superbanks per tile.
# Score at the winner cell: 92.0 % active-engine utilization, 21.20 MGE,
# 7,668 of 9,379 pins, FOM 1422.
#
# Channel mix is M6: (RO, RWreq, RWresp, WO) = (8, 2, 4, 2) per group.
# S1 offers ~33 % more load per engine channel than S2 (rho_eng 0.469 vs 0.352),
# which is why it buys eight RO channels and a deeper wide-response FIFO.

config_build_path := dse_s1_winner

# --- wide TCDM plane (WP2) channel mix: M6 -------------------------------
noc_req_wide_ro_channel_num  := 8
noc_req_wide_rw_channel_num  := 2
noc_resp_wide_rw_channel_num := 4
noc_resp_wide_wo_channel_num := 2

# --- RedMulE outstanding-transaction credit ------------------------------
# 24, not the stock 16: S1's deeper offered load needs the extra credit.
rob_depth := 24

# --- channel remapping ---------------------------------------------------
# Narrow plane: unchanged, group size 4.
noc_router_remapping := 3
noc_router_remap_group_size := 4
# Wide plane: 3 = rotate both aliasing classes (RO request, RW read-data
# response). For S1 this is a PREREQUISITE, not a tuning knob: the concentrator
# selects with `requester_index % nb_channels` over 4 tiles, so without rotation
# RO channels 4..7 are unreachable and this M6 mix degenerates to M3.
noc_wide_router_remapping := 3
# Per-class batch sizes. These default to the class counts -- 8 for RO and 4 for
# RW response -- which is exactly the model's ro/rw_resp_remap_batch_size, so the
# single narrow group size above no longer has to cover both.
noc_wide_ro_req_remap_group_size  := 8
noc_wide_rw_resp_remap_group_size := 4

# --- router buffering ----------------------------------------------------
# Globals stay at 2; only the wide RESPONSE network goes to 4, as the model runs
# it. A response flit carries the payload while a request is metadata only, so
# raising the globals instead would over-provision the other three classes for
# no measured gain (`rq2` was optimal at all 1224 grid cells).
# NOTE the model carries ONE depth per class where the RTL router has an input
# and an output FIFO. The RTL default 2/2 is what the model's 2 was calibrated
# against, so the model's 4 is carried over as 4/4 -- both scale together. If a
# future calibration wants the model's single depth to mean the input FIFO
# alone, only the output line below changes.
noc_router_input_fifo_dep  := 2
noc_router_output_fifo_dep := 2
noc_router_wide_resp_input_fifo_dep  := 4
noc_router_wide_resp_output_fifo_dep := 4

# =========================================================================
# MODEL/RTL GAPS AT THIS CONFIG -- ALL CLOSED
#
# GAP 1 (wide-plane remapper) -- CLOSED. mempool_group_floonoc_wrapper.sv now
#   instantiates i_floo_tcdm_wide_ro_req_remapper and
#   i_floo_tcdm_wide_rw_resp_remapper on the outgoing edge, gated by
#   noc_wide_router_remapping above. The channels-vs-ports guards in
#   hardware/Makefile and mempool_group.sv now permit an over-wide class exactly
#   when its remapper is on, which is what lets RO = 8 exist at 4 tile ports.
#
# GAP 2 (remap group size) -- CLOSED. noc_wide_{ro_req,rw_resp}_remap_group_size
#   are independent of the narrow noc_router_remap_group_size.
#
# GAP 3 (per-class FIFO depth) -- CLOSED. noc_router_{narrow,wide}_{req,resp}_*
#   split the two globals four ways, each defaulting to its global.
#
# REMAINING: the RedMulE X-shift register moved off the software header's 0x18
#   (which is LEFT_PARAMS in redmule_pkg.sv and is driven by redmule_tiler.sv).
#   See hardware/deps/redmule/sw/archi_redmule.h for the current offset; the
#   GVSoC model must match it.
#
# See gvsoc/docs/plans/teranoc-s1-s2-rtl-handoff.md for the full mapping.
# =========================================================================

include $(MEMPOOL_DIR)/config/dse_s1_common.mk
