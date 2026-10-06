# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# S1 winner, wide-plane plan C: plan B with the write request and write-ack
# channels shared by all four tiles instead of by pairs. In GVSoC (dev model,
# S1 suite) 1.004 x the group concentrator, worst case 0.966.

include $(MEMPOOL_DIR)/config/dse_s1_winner.mk

# After the include: the winner assigns it with :=.
config_build_path := dse_s1_winner_plan_c
noc_wide_wo_merge := 4
