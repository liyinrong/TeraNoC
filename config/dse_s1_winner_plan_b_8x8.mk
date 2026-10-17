# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# S1 winner, wide-plane plan B, on an 8x8 group mesh. GVSoC counterpart:
#
#     teranoc_v2:system_config=dse_s1_winner_plan_b_8x8
#
# Everything inside a group and on every link is plan B's
# (config/dse_s1_winner_plan_b.mk); only the mesh grows to 64 groups, 256 tiles,
# 1024 cores and 256 RedMulE engines.
#
# HBM stays on the perimeter: 32 channels for 64 groups. config/config.mk folds
# L2 over two groups' L1 width, so bank b serves groups 2b and 2b+1, which are
# vertical neighbours under Morton ids as well. The bank -> port map in
# config/floo_noc_terapool_8x8.yml is solved for Morton ids by
# hardware/scripts/gen_l2_bank_map.py: 104 hops over 64 groups, mean 1.63,
# worst case 3.
#
# L2 is 1 GiB (32 x 32 MiB), twice the 4x4 flavors, to hold a full Qwen layer.

# `?=` in the S1 common file and tensorpool256_spatz4.mk, so these win.
num_cores         := 1024
num_groups        := 64
num_x             := 8
num_redmule_tiles := 256
l2_banks          := 32
l2_size           := 1073741824  # 40000000

include $(MEMPOOL_DIR)/config/dse_s1_winner_plan_b.mk

# Retaken after the include, which pins both with `:=`.
config_build_path := dse_s1_winner_plan_b_8x8
floo_noc_cfg      := floo_noc_terapool_8x8.yml
