# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Stage-2 scale-up: the **S2 winner** on an 8x8 group mesh. Its GVSoC
# counterpart is exactly
#
#     teranoc_v2:system_config=dse_s2_winner_8x8
#
# Everything inside a group and on every link is inherited unchanged from
# config/dse_s2_winner.mk -- same RedMulE, same banks, same wide/narrow channel
# mix, same ROB depth, same router FIFO depths, same one wide and one narrow
# inter-group port per tile. Only the mesh grows: 16 groups to 64, so 256 tiles,
# 512 Snitch-Spatz cores and 256 engines.
#
# What cannot be inherited is the L2 attach. HBM stays on the cluster perimeter,
# so an NxN mesh offers 4N channels: 16 for 16 groups at 4x4, but 32 for 64
# groups at 8x8. Two groups therefore share a channel, and config/config.mk now
# carries that ratio into the L2 fold granule
#
#     L2BankBeWidth * Interleave = (num_groups/l2_banks) * NumBanksPerGroup * 4
#
# so `axi_width_interleaved` becomes 32 here. That makes bank b serve groups 2b
# and 2b+1, which are vertically adjacent -- group ids run down a column -- and
# lets one channel sit close to both. Leaving the granule at 16 would pair group
# g with g+32 instead: same row, four columns apart.
#
# The bank id -> mesh port map in config/floo_noc_terapool_8x8.yml follows the
# same rule as the 4x4 yml (each group as close as possible to its own bank,
# fewest crossings) and is derived by
# gvsoc/tools/teranoc-analysis/l2_bank_placement.py. Cost: 104 mesh hops over 64
# groups, 1.63 mean, worst case 3, worst link load 3. The 4x4 map scores 0.50
# mean and 1 worst -- edge-attached memory on a 2x larger mesh is what pays for
# that, not this placement.
#
# Not a free scale-up, and deliberately so: this is the "hold every group and
# link resource, grow only the mesh" control point. Per-group L2 bandwidth
# halves (32 channels for 64 groups against 16 for 16), so any shortfall against
# the 4x4 winner is attributable to the L2 attach and the longer mesh, not to a
# changed group.

# Mesh. num_cores/num_groups/num_x/num_redmule_tiles are `?=` in the S2 common
# file, so setting them here wins.
num_cores         := 512
num_groups        := 64
num_x             := 8
num_redmule_tiles := 256

# 4N perimeter channels. `l2_banks` is `?=` in tensorpool256_spatz4.mk.
l2_banks := 32

include $(MEMPOOL_DIR)/config/dse_s2_winner.mk

# Retaken after the include: dse_s2_winner.mk pins both with `:=`.
config_build_path := dse_s2_8x8_winner
floo_noc_cfg      := floo_noc_terapool_8x8.yml
