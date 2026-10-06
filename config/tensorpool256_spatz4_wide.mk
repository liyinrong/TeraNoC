# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# tensorpool256_spatz4 with the WP2 wide TCDM plane switched ON: one RW request
# and one RW response channel per group and mesh direction. The meta-only RO
# request and WO response classes stay off, so this is the cheapest plane that
# can carry RedMulE line traffic.
#
# Everything else -- geometry, narrow channel classes, RedMulE, Spatz, L2 -- is
# inherited unchanged from tensorpool256_spatz4, so a run of this flavor against
# tensorpool256_spatz4 isolates the wide plane and nothing else.
#
# Set the knobs BEFORE including the base flavor: it and config/config.mk both
# provide their defaults with ?=, so an earlier := wins.

config_build_path ?= tensorpool256_spatz4_wide

# One wide payload is one 512-bit RedMulE TCDM beat in this flavor.
wide_tcdm_data_width := 512

# The minimal wide plane: one port of each class per tile.
noc_req_wide_ro_port_num  := 1
noc_req_wide_wo_port_num  := 1
noc_resp_wide_rw_port_num := 1
noc_resp_wide_wo_port_num := 1

# One dedicated wide link per tile for intra-group traffic.
num_intra_group_ports_per_tile := 1

include $(MEMPOOL_DIR)/config/tensorpool256_spatz4.mk
