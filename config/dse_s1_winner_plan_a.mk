# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# S1 winner, wide-plane plan A: ONE RW response port per tile. Fewer read-data
# wires, but all of a tile's read data shares one mesh, so the RW response ports
# rotate over all four tiles of the group and the wide response routers keep
# 4-deep FIFOs. The write request and write-ack channels are shared by all four
# tiles. In GVSoC (dev model, S1 suite) 0.993 x the group concentrator, worst
# case 0.922.

include $(MEMPOOL_DIR)/config/dse_s1_winner.mk

# After the include: the winner assigns these with :=.
config_build_path := dse_s1_winner_plan_a
noc_resp_wide_rw_port_num := 1
noc_wide_rw_resp_remap_group_size := 4
noc_wide_wo_merge := 4
noc_router_wide_resp_input_fifo_dep  := 4
noc_router_wide_resp_output_fifo_dep := 4
