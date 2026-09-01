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
spm_profiling ?= 1

# Enable the interconnect access pattern profiling
noc_profiling ?= 1

# Per-core Spatz engine trace (one file per Spatz core, a line per active cycle).
# Inactive on non-Spatz flavors regardless.
spatz_profiling ?= 1

# FlooNoC router buffering. The two globals below are the fallback for every
# router in both planes; flavors have always set them to 2. The stage-1 DSE
# sweeps the four classes independently and its S1 winner wants a deeper wide
# RESPONSE buffer alone -- a response flit carries the payload (~1053 b at a
# 1024-bit plane) against a request's metadata (~46 b), so a response slot costs
# roughly 23x a request slot and `rq2` was optimal at every one of the 1224 grid
# cells. Raising the globals instead would over-provision the other three
# classes. Each per-class knob defaults to the matching global, so a flavor that
# sets only the globals is bit-identical to before this split existed.
noc_router_input_fifo_dep  ?= 2
noc_router_output_fifo_dep ?= 2

noc_router_narrow_req_input_fifo_dep   ?= $(noc_router_input_fifo_dep)
noc_router_narrow_req_output_fifo_dep  ?= $(noc_router_output_fifo_dep)
noc_router_narrow_resp_input_fifo_dep  ?= $(noc_router_input_fifo_dep)
noc_router_narrow_resp_output_fifo_dep ?= $(noc_router_output_fifo_dep)
noc_router_wide_req_input_fifo_dep     ?= $(noc_router_input_fifo_dep)
noc_router_wide_req_output_fifo_dep    ?= $(noc_router_output_fifo_dep)
noc_router_wide_resp_input_fifo_dep    ?= $(noc_router_input_fifo_dep)
noc_router_wide_resp_output_fifo_dep   ?= $(noc_router_output_fifo_dep)

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
# Counts are physical channels per group and mesh direction, shared by all tile
# logical wide ports through the group concentrator/dispatcher. The defaults
# below leave the plane absent, which is bit-identical to a build without it;
# a flavor turns it on by setting the counts directly before including its base
# flavor (see config/tensorpool256_wide.mk). noc_req_wide_rw_channel_num >= 1 is
# what enables the plane. Flavors without a RedMulE wide endpoint must keep it
# off.
noc_req_wide_ro_channel_num ?= 0
noc_req_wide_rw_channel_num ?= 0
noc_resp_wide_rw_channel_num ?= 0
noc_resp_wide_wo_channel_num ?= 0

# Logical wide ports at each tile. Intra-group ports are dedicated links and do
# not enter the group mapper. Inter-group ports are class-neutral at the tile;
# the group maps reads/writes onto its physical RO/RW/WO channel classes.
# A tile's wide ports are symmetric, so one count sizes both directions of each
# kind. Both remain parameters even though the current RedMulE integration is
# validated only for one intra-group plus one inter-group path.
num_intra_group_ports_per_tile ?= 1
num_inter_group_ports_per_tile ?= 1

# INVARIANT: every wide channel class must be a multiple of the inter-group
# ports per tile. The inter-group ports are batched by port id -- port r of
# every tile shares one batch of channels, and a request issued on port r must
# be carried and answered inside batch r alone. Two pieces of logic have to
# agree on that partition: the group flattens its mapper inputs as
# i = tile*ports + port and picks a channel with `i % class_count`
# (mempool_group_channel_mapper.sv, driven from gen_wide_{req,resp}_mapper_t/r),
# while the endpoint crossbars recover the port with `channel % ports`
# (gen_wide_slave_req_sel / gen_wide_master_resp_sel). They agree if and only if
# the class count divides evenly; otherwise the batches overlap and a response
# comes back on a port whose requester never sent the request -- a silent
# mis-route with no X, no hang and no assertion, found late and looking like a
# data-corruption or ROB bug rather than a config one. 0 is fine: it disables
# the class. mempool_group.sv (p_wide_group_channel_count) asserts the same
# thing at elaboration and the GVSoC model in pulp/pulp/teranoc_v2/arch.py
# (TeranocWideConfig.__post_init__) at model build; checking it here fails a bad
# flavor at build configuration instead. The magnitude limits on the same four
# counts live next to the wide vlog_defs in hardware/Makefile.
# This check is placed after the port counts rather than next to the channel
# counts above because it needs both.
ifneq (0,$(strip $(num_inter_group_ports_per_tile)))
wide_ro_req_channel_batch_remainder := $(shell printf '%s' \
	$$(( $(noc_req_wide_ro_channel_num) % $(num_inter_group_ports_per_tile) )))
wide_rw_req_channel_batch_remainder := $(shell printf '%s' \
	$$(( $(noc_req_wide_rw_channel_num) % $(num_inter_group_ports_per_tile) )))
wide_rw_resp_channel_batch_remainder := $(shell printf '%s' \
	$$(( $(noc_resp_wide_rw_channel_num) % $(num_inter_group_ports_per_tile) )))
wide_wo_resp_channel_batch_remainder := $(shell printf '%s' \
	$$(( $(noc_resp_wide_wo_channel_num) % $(num_inter_group_ports_per_tile) )))
ifneq (0,$(strip $(wide_ro_req_channel_batch_remainder)))
$(error noc_req_wide_ro_channel_num=$(noc_req_wide_ro_channel_num) must be a multiple of the num_inter_group_ports_per_tile=$(num_inter_group_ports_per_tile) inter-group ports per tile, so the per-port wide channel batches partition)
endif
ifneq (0,$(strip $(wide_rw_req_channel_batch_remainder)))
$(error noc_req_wide_rw_channel_num=$(noc_req_wide_rw_channel_num) must be a multiple of the num_inter_group_ports_per_tile=$(num_inter_group_ports_per_tile) inter-group ports per tile, so the per-port wide channel batches partition)
endif
ifneq (0,$(strip $(wide_rw_resp_channel_batch_remainder)))
$(error noc_resp_wide_rw_channel_num=$(noc_resp_wide_rw_channel_num) must be a multiple of the num_inter_group_ports_per_tile=$(num_inter_group_ports_per_tile) inter-group ports per tile, so the per-port wide channel batches partition)
endif
ifneq (0,$(strip $(wide_wo_resp_channel_batch_remainder)))
$(error noc_resp_wide_wo_channel_num=$(noc_resp_wide_wo_channel_num) must be a multiple of the num_inter_group_ports_per_tile=$(num_inter_group_ports_per_tile) inter-group ports per tile, so the per-port wide channel batches partition)
endif
endif

# Wide TCDM channel remapping (stage-1 DSE). The group concentrator binds a tile
# to a channel with `i % class_count` over the flattened input index i, so the
# binding is static: a tile always injects on the same channel, and a class wider
# than the aggregate tile-port count can never reach its upper channels at all.
# A floo_remapper on the OUTGOING edge -- between the concentrator and the group
# boundary -- rotates that permutation once per cycle, which decorrelates the
# per-tile choice and makes an over-wide class fully reachable. This is the
# largest single utilization lever the stage-1 DSE measured (+33-41 % per MGE).
# Only the RO request and RW response classes are remapped: RW request and WO
# response do not alias the same way and rotating them buys nothing.
#   0: none, 1: RO request, 2: RW response, 3: both
# Nothing on the return path depends on which physical channel a request used --
# responses route on the src_tile_id / src_id carried in the flit header and the
# responder's group picks its response channel with its own `i % N` -- which is
# what makes rotating the request channel safe. See the ports-per-tile caveat in
# hardware/Makefile next to the wide remap validation.
noc_wide_router_remapping ?= 0

# floo_remapper GroupSize for each remapped class: the number of channels one
# rotation spans. Must be a power of two >= 2 that divides the class count.
# Defaulting to the class count gives a single batch, i.e. one rotation across
# the whole class, which is what both stage-1 DSE designs use.
noc_wide_ro_req_remap_group_size  ?= $(noc_req_wide_ro_channel_num)
noc_wide_rw_resp_remap_group_size ?= $(noc_resp_wide_rw_channel_num)

# Number of RedMulE tensor-core tiles in the cluster
num_redmule_tiles ?= 0
redmule_height ?= 0
redmule_width  ?= 0
redmule_regs   ?= 0

# Enable the Spatz vector unit (RVV)
spatz ?= 0

# Spatz burst emission. Requires a burst-expanding memory side; see
# hardware/src/tcdm_burst_expander.sv. 0 = word-interleaved path only.
spatz_burst ?= 0



# Dumps (address, data) for every vector load response, so a narrow run and a
# wide+burst run can be diffed by address. Debug only -- one line per load.
spatz_ld_trace ?= 0

# Words (32 bit) in one vector-load burst. A burst is unpacked from a SINGLE wide
# access, so this is a property of the wide plane, not of Spatz: it must be
# wide_tcdm_data_width / 32. Derived by default; override only to burst narrower
# than the plane. It need not divide the VLSU's memory-port count -- a burst that
# is not a whole multiple simply gives the low lanes one beat more than the high
# ones. When the wide plane is off the value is unused, so keep it legal (1).
ifeq ($(strip $(wide_tcdm_data_width)),0)
spatz_max_burst_words ?= 1
else
spatz_max_burst_words ?= $(shell expr $(wide_tcdm_data_width) / 32)
endif

# Spatz VLSU reorder-buffer depth. Also the width root of the ROB id that
# mempool_cc packs into meta_id alongside the store-ack flag, so MetaIdWidth
# follows this knob (mempool_pkg.sv) rather than constraining it.
# 16 is the largest depth that keeps meta_id at 5 bits. 32 and 64 both run
# correctly (verified: sp_matmul/dotp/axpy_f32 and sp_gemv_f16, cycle-identical
# across all three depths), but each doubling widens meta_id by a bit and with it
# every mesh link, so raise this only when a measurement pays for the area.
spatz_rob_depth ?= 16

# Spatz vector-unit parameters
n_fpu ?= 1
n_ipu ?= 1
vlen  ?= 0
rvf   ?= 0
rvd   ?= 0

# Spatz cores and a RedMulE engine can share a tile: the per-core data ports
# (NumDataPortsPerCore) and the RedMulE master ports are flattened onto one
# local-port axis (RedMulE slots start at NumSoCPorts), and MetaIdWidth is the
# max over snitch/Spatz/RedMulE. See config/tensorpool256_spatz4.mk.

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
# HBM sits on the cluster perimeter, so a large mesh has fewer L2 banks than
# groups (4N channels for N^2 groups) and each bank serves num_groups/l2_banks
# of them. The invariant above is then the *set* of groups behind one bank:
#     L2BankBeWidth * Interleave == (num_groups/l2_banks) * NumBanksPerGroup * 4
# The ratio is 1 for every 4x4 flavor, so this is a no-op there. Getting it
# wrong does not corrupt anything -- it changes *which* groups share a bank,
# and at a ratio of 2 the alternative pairs a group with the one half a mesh
# away instead of its vertical neighbour.
groups_per_l2_bank  := $(shell echo $$(( $(num_groups) / $(l2_banks) )))
axi_width_interleaved := $(shell echo $$(( $(groups_per_l2_bank) * $(num_banks_per_group) * 4 / ($(axi_data_width) / 8) )))
