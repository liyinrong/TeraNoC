# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# S1 winner, wide-plane plan B: the winner itself (config/dse_s1_winner.mk),
# named for side-by-side builds with plans A, C and D. Two RW response ports per
# tile with pair remaps and 2-deep FIFOs; the write channels are shared by tile
# pairs. In GVSoC (dev model, S1 suite) 1.008 x the group concentrator, worst
# case 0.983.

include $(MEMPOOL_DIR)/config/dse_s1_winner.mk

config_build_path := dse_s1_winner_plan_b
