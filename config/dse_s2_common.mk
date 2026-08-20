# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Stage-1 DSE S2 common geometry: four H16xW32 RedMulE tiles per group on the
# fixed 4x4 group mesh, two Snitch-Spatz cores and 64 8-KiB L1 banks per tile.
# Channel-mix wrappers set the four WP2 class counts before including this file.

num_cores          ?= 128
num_groups         ?= 16
num_cores_per_tile ?= 2
banking_factor     ?= 8
l1_bank_size       ?= 8192

num_x        ?= 4
noc_topology ?= 0

# The narrow counts are per tile. Keep two RW request and response channels for
# irregular Snitch/Spatz traffic; this plane is not a DSE axis.
noc_req_narrow_ro_channel_num  := 0
noc_req_narrow_rw_channel_num  := 2
noc_req_narrow_wo_channel_num  := 0
noc_resp_narrow_rw_channel_num := 2

wide_tcdm_data_width ?= 1024

num_intra_group_ports_per_tile := 1
num_inter_group_ports_per_tile := 1

num_redmule_tiles ?= 64
redmule_height    ?= 16
redmule_width     ?= 32
redmule_regs      ?= 3
rob_depth         ?= 16

# config/config.mk derives axi_width_interleaved=16 from the 256 banks/group
# and 64-byte AXI beat. Keep the fold unchanged when growing the bank depth.
include $(MEMPOOL_DIR)/config/tensorpool256_spatz4.mk
