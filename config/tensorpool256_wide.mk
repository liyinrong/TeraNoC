# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# tensorpool256 with the WP2 wide TCDM plane switched ON: one 512-bit wide
# channel each way per group and mesh direction (1 RW request + 1 RW response;
# the meta-only RO request and WO response classes stay off). One wide payload
# is one 512-bit RedMulE TCDM beat.
#
# Geometry, narrow channel classes, RedMulE and L2 are inherited unchanged from
# tensorpool256, so running this flavor against tensorpool256 isolates the wide
# plane and nothing else.
#
# Set the knobs BEFORE including the base flavor: it and config/config.mk both
# provide their defaults with ?=, so an earlier := wins.

config_build_path ?= tensorpool256_wide

# Wide payload width: one 512-bit RedMulE TCDM beat.
wide_tcdm_data_width := 512

# The minimal wide plane: one port of each class per tile.
noc_req_wide_ro_port_num  := 1
noc_req_wide_wo_port_num  := 1
noc_resp_wide_rw_port_num := 1
noc_resp_wide_wo_port_num := 1

# One dedicated wide link per tile for intra-group traffic.
num_intra_group_ports_per_tile := 1

include $(MEMPOOL_DIR)/config/tensorpool256.mk
