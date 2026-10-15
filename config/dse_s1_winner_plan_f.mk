# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# S1 winner, wide-plane plan F: plan D with the RO request ports rotated over all
# four tiles of the group instead of tile pairs. RO requests are header-only, so
# the wider remap is cheap; the RW response network is plan D's. In GVSoC (dev
# model, S1 suite) 1.003 x plan D, prefill 1.005.

include $(MEMPOOL_DIR)/config/dse_s1_winner_plan_d.mk

# After the include: the winner assigns it with :=.
config_build_path := dse_s1_winner_plan_f
noc_wide_ro_req_remap_group_size := 4
