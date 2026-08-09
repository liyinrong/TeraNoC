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
# Set the knobs BEFORE including the base flavor: it provides its defaults with
# ?= and config/config.mk evaluates the wide-channel mode after the include.

config_build_path ?= tensorpool256_spatz4_wide

# One wide payload is one 512-bit RedMulE TCDM beat in this flavor.
wide_tcdm_data_width := 512

# 1 RW req + 1 RW resp per group (RO req and WO resp disabled).
wide_channel_config_mode := baseline

# One logical wide port per tile for intra-group traffic and one each for the
# inter-group request and response paths.
noc_wide_intra_group_channel_num_per_tile      := 1
noc_wide_inter_group_req_channel_num_per_tile  := 1
noc_wide_inter_group_resp_channel_num_per_tile := 1

include $(MEMPOOL_DIR)/config/tensorpool256_spatz4.mk
