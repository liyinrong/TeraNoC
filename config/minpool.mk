# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Yichao Zhang, ETH Zurich

###########################
## 1. Architecture Config
###########################

# FlooNoC configuration file
floo_noc_cfg ?= floo_noc_minpool.yml

# Number of cores
num_cores ?= 16

# Number of groups
num_groups ?= 4

# Number of cores per MemPool tile
num_cores_per_tile ?= 4

# L1 scratchpad banking factor
banking_factor ?= 4

# Number of shared divsqrt units per MemPool tile
# Defaults to 1 if xDivSqrt is activated
num_divsqrt_per_tile ?= 1

#####################
## 2. NoC Config
#####################

# FlooNoC configuration
num_directions ?= 5
num_x          ?= 2

# Topology
# 0: 2D mesh, 1: torus
noc_topology ?= 0

# Routing algorithm
# 0: xy, 1: id_table
noc_routing_algorithm ?= 0

# NoC remapping configuration
# 0: no remapping, 1: req remapping, 2: resp remapping 3: req+resp remapping
noc_router_remapping ?= 0

# Virtual channel number
noc_virtual_channel_num ?= 1

# Narrow (32-bit) TCDM plane: physical channels per group and mesh
# direction, by class. RO = meta-only read request, RW = payload
# request/response, WO = write-only request. At least one write-capable
# request channel (RW or WO) and one RW response channel are required.
noc_req_narrow_ro_channel_num ?= 0
noc_req_narrow_rw_channel_num ?= 1
noc_req_narrow_wo_channel_num ?= 0
noc_resp_narrow_rw_channel_num ?= 1

# Print configuration info for debugging
$(info [DEBUG:][noc_req_narrow_ro_channel_num]----$(noc_req_narrow_ro_channel_num))
$(info [DEBUG:][noc_req_narrow_rw_channel_num]--$(noc_req_narrow_rw_channel_num))
$(info [DEBUG:][noc_req_narrow_wo_channel_num]----$(noc_req_narrow_wo_channel_num))
$(info [DEBUG:][noc_resp_narrow_rw_channel_num]------$(noc_resp_narrow_rw_channel_num))

# Router buffer configuration
noc_router_input_fifo_dep  ?= 2
noc_router_output_fifo_dep ?= 2

# Router remapping configuration
noc_router_remap_group_size ?= 2

###########################
## 3. AXI and DMA Config
###########################

# AXI bus data width (in bits)
axi_data_width ?= 256

# Read-only cache line width in AXI interconnect (in bits)
ro_line_width ?= 256

# Radix for hierarchical AXI interconnect
axi_hier_radix ?= 2

# Number of AXI masters per group
axi_masters_per_group ?= 1

# Number of DMA backends in each group
dmas_per_group ?= 1  # Burst Length = 16

# L2 Banks/Channels
l2_size               ?= 4194304   # 400000
l2_banks              ?= 4
axi_width_interleaved ?= 2
