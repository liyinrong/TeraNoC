# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# S1 winner, wide-plane plan D: plan A with the write request and write-ack
# channels shared by tile pairs (0/2, 1/3) as in plan B, instead of by all four
# tiles.

include $(MEMPOOL_DIR)/config/dse_s1_winner_plan_a.mk

# After the include: plan A assigns it with :=.
config_build_path := dse_s1_winner_plan_d
noc_wide_wo_merge := 2
