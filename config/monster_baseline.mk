# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Monster reference flavor: four engine tiles per group on a 4x4 group mesh.
# Each tile contains two Snitch-Spatz cores, 32 L1 banks, and one H8xW32
# RedMulE. Both TCDM planes retain their original baseline channel classes.

# Set geometry and channel counts before including the TensorPool flavor: it
# provides its defaults with ?=, so an earlier := wins.
config_build_path ?= monster_baseline

num_cores          ?= 128
num_groups         ?= 16
num_cores_per_tile ?= 2
banking_factor     ?= 4

num_x        ?= 4
noc_topology ?= 0

# Narrow plane: 2 RW request + 2 RW response channels per group and mesh
# direction. Wide plane on with 1 RW request + 1 RW response.
noc_req_narrow_ro_channel_num  := 0
noc_req_narrow_rw_channel_num  := 2
noc_req_narrow_wo_channel_num  := 0
noc_resp_narrow_rw_channel_num := 2

noc_req_wide_ro_channel_num   := 0
noc_req_wide_rw_channel_num   := 1
noc_resp_wide_rw_channel_num  := 1
noc_resp_wide_wo_channel_num  := 0

noc_wide_intra_group_channel_num_per_tile := 1
noc_wide_inter_group_req_channel_num_per_tile := 1
noc_wide_inter_group_resp_channel_num_per_tile := 1

num_redmule_tiles ?= 64
redmule_height    ?= 8
redmule_width     ?= 32
redmule_regs      ?= 3
rob_depth         ?= 16

include $(MEMPOOL_DIR)/config/tensorpool256_spatz4.mk
