# Copyright 2021 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Author: Samuel Riedel, ETH Zurich
#         Matheus Cavalcante, ETH Zurich

######################
##  MemPool flavor  ##
######################

# Choose a MemPool flavor (see config/*.mk and the README). When none is selected
# explicitly the build falls back to `mempool` and warns once, so a forgotten
# `config=` is visible while flavor-independent targets (toolchains, bender, the
# boot ROM, opcode tables) keep working. Pass `config=<flavor>` or set the
# MEMPOOL_CONFIGURATION env var to select another flavor.
ifndef config
  ifdef MEMPOOL_CONFIGURATION
    config := $(MEMPOOL_CONFIGURATION)
  else
    config := mempool
    # Warn only at the top-level invocation so recursive sub-makes stay quiet.
    ifeq ($(MAKELEVEL),0)
      $(warning No MemPool `config` selected; defaulting to `mempool`. Pass `config=<flavor>` (e.g. minpool, terapool, minpool_spatz4_fpu) to choose another; run `ls config/*.mk` for the list.)
    endif
  endif
endif
include $(MEMPOOL_DIR)/config/$(config).mk

#############################
##  Address configuration  ##
#############################

# Boot address (in dec)
boot_addr ?= 2684354560 # A0000000

# L2 memory configuration (in dec)
l2_base  ?= 2147483648 # 80000000

# L1 size per bank (in dec)
l1_bank_size ?= 1024

# Size of sequential memory per core (in bytes)
# (must be a power of two)
seq_mem_size ?= 512

# Size of stack in sequential memory per core (in bytes)
stack_size ?= 512

#########################
##  AXI configuration  ##
#########################
# AXI bus data width (in bits)
axi_data_width ?= 512

# Read-only cache line width in AXI interconnect (in bits)
ro_line_width ?= 512

#############################
##  Xqueues configuration  ##
#############################

# XQueue extension's queue size in each memory bank (in words)
xqueue_size ?= 0

################################
##  Optional functionalities  ##
################################

# Enable the XpulpIMG extension
xpulpimg ?= 1

# Enable FPU extensions
zfinx ?= 1

# Enable FPU extensions
zquarterinx ?= 0

# DivSqrt deactivated by default
xDivSqrt ?= 0

# DRAMsys co-simulation: dram/sram
l2_sim_type ?= sram
axi_width_interleaved ?= 16

# Enable SPM bank id remapping inside of each tile
spm_bank_id_remap ?= 0

# Enable tile id remapping inside of each group
tile_id_remap ?= 0

# Enable the spm access pattern profiling
spm_profiling ?= 0

# Enable the interconnect access pattern profiling
noc_profiling ?= 0

# Wide TCDM payload width in bits. A zero default is safe while the plane is
# disabled; RedMulE-enabled flavors set the width of one engine TCDM beat.
wide_tcdm_data_width ?= 0

# Wide TCDM plane (WP2): per-group, per-class channel counts for the separate,
# single-beat wide FlooNoC network carrying RedMulE line traffic. The plane is
# read-dominated and structurally asymmetric, so the classes are sized apart:
#   RO   req  - meta only, ~46 wires: cheap, buy several
#   RW req  - meta + WIDE_W + strb, ~627 wires: expensive, carries write data
#   RW resp - meta + WIDE_W, ~540 wires: read data or a write ACK
#   WO resp - meta only, ~28 wires: a write ACK, not a data response
# RW req and RW resp are MANDATORY when the plane is on (only they can carry
# write data / read data). RO req and WO resp are optional; 0 disables the class.
#
# Selected by wide_channel_config_mode, mirroring narrow_channel_config_mode:
#   off      - plane absent (default), with every class count set to zero.
#   baseline - the original one-class plane: every request on RW, every
#              response on RW. Kept so earlier WP2 results stay reproducible.
#   split    - all four classes present, one channel each. The cheapest config
#              that exercises the meta-only RO request and WO response paths.
#   readopt  - the DSE plan's §6.6 recommended mix: many cheap read requests, few
#              expensive write requests, wide read responses, cheap write acks.
#              2048 b/cyc read + 1024 b/cyc write, balanced 6 req/cyc <-> 6
#              resp/cyc, at 76% of terapool's narrow-plane wire budget. Requires
#              at least six tile request/response ports per group.
# Counts are physical channels per group and mesh direction, shared by all tile
# logical wide ports through the group concentrator/dispatcher. RedMulE-enabled
# flavors may pin or sweep these presets without editing the shared config.
# Flavors without a RedMulE wide endpoint must keep the plane off.
wide_channel_config_mode ?= off

ifeq ($(strip $(wide_channel_config_mode)), off)
noc_req_wide_ro_channel_num ?= 0
noc_req_wide_rw_channel_num ?= 0
noc_resp_wide_rw_channel_num ?= 0
noc_resp_wide_wo_channel_num ?= 0
else ifeq ($(strip $(wide_channel_config_mode)), baseline)
noc_req_wide_ro_channel_num ?= 0
noc_req_wide_rw_channel_num ?= 1
noc_resp_wide_rw_channel_num ?= 1
noc_resp_wide_wo_channel_num ?= 0
else ifeq ($(strip $(wide_channel_config_mode)), split)
noc_req_wide_ro_channel_num ?= 1
noc_req_wide_rw_channel_num ?= 1
noc_resp_wide_rw_channel_num ?= 1
noc_resp_wide_wo_channel_num ?= 1
else ifeq ($(strip $(wide_channel_config_mode)), readopt)
noc_req_wide_ro_channel_num ?= 4
noc_req_wide_rw_channel_num ?= 2
noc_resp_wide_rw_channel_num ?= 4
noc_resp_wide_wo_channel_num ?= 2
else
$(error Unsupported wide_channel_config_mode: $(wide_channel_config_mode))
endif

# Logical wide ports at each tile. Intra-group ports are dedicated links and do
# not enter the group mapper. Inter-group ports are class-neutral at the tile;
# the group maps reads/writes onto its physical RO/RW/WO channel classes.
# All counts remain parameters even though the current RedMulE integration is
# validated only for one intra-group plus one inter-group request/response path.
noc_wide_intra_group_channel_num_per_tile ?= 1
noc_wide_inter_group_req_channel_num_per_tile ?= 1
noc_wide_inter_group_resp_channel_num_per_tile ?= 1

# Number of RedMulE tensor-core tiles in the cluster
num_redmule_tiles ?= 0
redmule_height ?= 0
redmule_width  ?= 0
redmule_regs   ?= 0

# Enable the Spatz vector unit (RVV)
spatz ?= 0

# Spatz vector-unit parameters
n_fpu ?= 1
n_ipu ?= 1
vlen  ?= 0
rvf   ?= 0
rvd   ?= 0

# Spatz cores and a RedMulE engine can share a tile: the per-core data ports
# (NumDataPortsPerCore) and the RedMulE master ports are flattened onto one
# local-port axis (RedMulE slots start at NumSoCPorts), and MetaIdWidth is the
# max over snitch/Spatz/RedMulE. See config/minpool_spatz4_redmule.mk.

# ---------------------------------------------------------------------------
# L2 interleave (folding) granularity -- DERIVED, not hardcoded.
#
# INVARIANT: the L2 fold granularity must equal the L1 width of one group:
#     L2BankBeWidth * Interleave  ==  NumBanksPerGroup * 4 bytes
#   (L2BankBeWidth = axi_data_width/8; Interleave = axi_width_interleaved)
#
# NumBanksPerGroup = num_cores * NumFUsPerCore * banking_factor / num_groups, so
# it scales with the flavor -- including the Spatz N_FU multiplier. Hardcoding
# axi_width_interleaved (previously 16 in every flavor) silently mis-folds L2 for
# any flavor whose group L1 width isn't 16*axi_data_width/8: the whole tensorpool
# family was off by 2x (tensorpool256/64 need 8, tensorpool256_spatz4 needs 32),
# while mempool/terapool/minpool happened to match. Deriving it keeps L2 folding
# aligned with the group L1 width automatically as the geometry changes.
# A command-line `axi_width_interleaved=N` still overrides this.
num_fus_per_core    := $(shell [ $(n_ipu) -gt $(n_fpu) ] && echo $(n_ipu) || echo $(n_fpu))
num_banks_per_group := $(shell echo $$(( $(num_cores) * $(num_fus_per_core) * $(banking_factor) / $(num_groups) )))
axi_width_interleaved := $(shell echo $$(( $(num_banks_per_group) * 4 / ($(axi_data_width) / 8) )))
