# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Monster reference flavor: four engine tiles per group on a 4x4 group mesh.
# Each tile contains two Snitch-Spatz cores, 32 L1 banks, and one H8xW32
# RedMulE. Both TCDM planes retain their original baseline channel classes.

# Set geometry and channel modes before including the TensorPool flavor because
# it provides defaults with ?= and evaluates the narrow-channel mode immediately.
config_build_path ?= monster_baseline

num_cores          ?= 128
num_groups         ?= 16
num_cores_per_tile ?= 2
banking_factor     ?= 4

num_x        ?= 4
noc_topology ?= 0

narrow_channel_config_mode := baseline
wide_channel_config_mode   := baseline
noc_wide_intra_group_channel_num_per_tile := 1
noc_wide_inter_group_req_channel_num_per_tile := 1
noc_wide_inter_group_resp_channel_num_per_tile := 1

num_redmule_tiles ?= 64
redmule_height    ?= 8
redmule_width     ?= 32
redmule_regs      ?= 3
rob_depth         ?= 16

include $(MEMPOOL_DIR)/config/tensorpool256_spatz4.mk
