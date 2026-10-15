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

# The system address map. Everything that decodes an address -- the SoC xbar,
# the control registers, the FlooNoC SAM, the linker scripts and the runtime --
# reads the three bases below, so a map move is one edit here plus a
# regenerated FlooNoC package. `make update-floogen` fails the build if the
# generated package and these knobs ever disagree.
#
#   0x0000_0000  TCDM                     1.5 GiB reserved (WP3 VC windows)
#   0x6000_0000  bootrom                  bootrom_base
#   0x7000_0000  control regs / periphs   periph_base, +0x2_0000 window
#   0x7002_0000  RedMulE config regs      periph_base + 0x2_0000
#   0x7100_0000  testbench UART           periph_base + 0x100_0000
#   0x8000_0000  L2 / HBM                 l2_base, 2 GiB, nothing above it

# Bootrom base address (in dec)
bootrom_base ?= 1610612736 # 60000000

# Boot address (in dec). This is the core reset vector, which has to land on the
# first bootrom instruction, so it follows the bootrom rather than standing on
# its own. The bootrom image is PC-relative -- regenerate it when this moves.
boot_addr ?= $(bootrom_base)

# Control register / peripheral region base (in dec)
periph_base ?= 1879048192 # 70000000

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

# Wide TCDM plane (WP2): per-tile port counts for the separate, single-beat wide
# FlooNoC network carrying RedMulE line and Spatz burst traffic. Built like the
# narrow plane: every inter-group port of every tile owns its own router, so
# nothing has to sit at the centre of the group. The traffic is read-dominated
# and a read request carries no data, so the ports split by direction:
#   RO req  - meta only, ~46 wires: read requests, port picked by address hash
#   WO req  - meta + WIDE_W + strb, ~627 wires: write requests
#   RW resp - meta + WIDE_W, ~540 wires: read data, on the RW port paired with
#             the RO port the read came from (RO port k -> RW port k % count)
#   WO resp - meta only, ~28 wires: write acks
# All four are mandatory when the plane is on; noc_req_wide_ro_port_num >= 1
# turns it on. The defaults leave the plane absent, which is bit-identical to a
# build without it. Flavors without a RedMulE or Spatz-burst client keep it off.
noc_req_wide_ro_port_num  ?= 0
noc_req_wide_wo_port_num  ?= 0
noc_resp_wide_rw_port_num ?= 0
noc_resp_wide_wo_port_num ?= 0

# Writes are rare, so the write channels are shared: noc_wide_wo_merge tiles of
# a group (tile t and t + tiles/merge) meet on one WO request and one WO response
# channel, through an arbiter on the way out and a crossbar to the target tile on
# the way in. 1 gives every tile its own write channels.
noc_wide_wo_merge ?= 1

# The group concentrator these replace counted channels per group. Its knobs
# would now mean something else, so a flavor still setting them stops here.
wide_legacy_channels := $(shell echo $$(( 0$(strip $(noc_req_wide_ro_channel_num)) + \
	0$(strip $(noc_req_wide_rw_channel_num)) + 0$(strip $(noc_resp_wide_rw_channel_num)) + \
	0$(strip $(noc_resp_wide_wo_channel_num)) + 0$(strip $(num_inter_group_ports_per_tile)) )))
ifneq (0,$(strip $(wide_legacy_channels)))
$(error The wide-plane group concentrator is gone: replace noc_*_wide_*_channel_num and num_inter_group_ports_per_tile with the per-tile noc_req_wide_{ro,wo}_port_num, noc_resp_wide_{rw,wo}_port_num and noc_wide_wo_merge)
endif

# Dedicated same-group wide links per tile. They do not enter the wide NoC.
num_intra_group_ports_per_tile ?= 1

# Wide router remapping, as the narrow plane's noc_router_remapping: a
# floo_remapper rotates port p of a group of tiles once per cycle, so a burst
# that would pin one router spreads over the group.
#   0: none, 1: RO request ports, 2: RW response ports, 3: both
# The merged WO channels are never remapped.
noc_wide_router_remapping ?= 0

# floo_remapper GroupSize for each remapped class: the number of tiles one
# rotation spans for a port. Interleaved, so a group of 2 pairs tiles t and
# t + tiles/2. Must be a power of two that divides the tiles per group.
noc_wide_ro_req_remap_group_size  ?= 2
noc_wide_rw_resp_remap_group_size ?= 2

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

# Single-cycle reservation of each lane's reorder-buffer id window for a burst. 0 = the
# allocator walks one id per lane per cycle, putting that latency in front of every burst
# request; 1 = every lane is granted its whole window at once. The window is one lane's
# share of a full burst, so the buffer depth must be at least two windows.
spatz_block_alloc ?= 0

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

# Instruction caches: L1 bytes per tile and read-only cache bytes per group.
# Lines (256 bit), L1 ways and L0 lines are fixed in mempool_pkg.sv.
icache_size_byte   ?= $(shell echo $$(( 512 * $(num_cores_per_tile) )))
ro_cache_size_byte ?= 8192

# Snitch-only cores per tile (no Spatz, no FP): the first ones of every tile,
# the rest are Snitch-Spatz. Spatz builds only.
num_scalar_cores_per_tile ?= 0
ifneq ($(strip $(num_scalar_cores_per_tile)),0)
ifneq ($(strip $(spatz)),1)
$(error num_scalar_cores_per_tile needs a Spatz build (spatz=1))
endif
ifeq ($(shell [ $(num_scalar_cores_per_tile) -lt $(num_cores_per_tile) ] && echo ok),)
$(error num_scalar_cores_per_tile must be below num_cores_per_tile)
endif
endif

# ---------------------------------------------------------------------------
# WP3 virtual-cluster address windows -- DERIVED, emitted to BOTH the RTL
# (vlog_defs) and the runtime (DEFINES) so hardware and software cannot drift.
#
# A level's window is the FULL L1 of its 2^k x 2^k group rectangle, so its
# aperture is 4^k x the per-group L1 (memory [[max-size-vc-windows-and-delete-
# chunk-base]]; chunk_base is deleted, which is what makes the window maximal).
# Windows are packed DESCENDING by size immediately above the global region:
# every aperture is a power of two, so descending order lands each window on
# its natural alignment with zero padding, and the whole map costs 1.33x L1 at
# every scale.
#
#     vc_window_size(k) = vc_group_bytes * 4^k
#     vc_window_base(k) = l1_total + sum_{j>k} vc_window_size(j)
#
# The TOP level k = log2(min(num_x,num_y)) deliberately has NO window: on a
# square mesh its permuter is the identity, so the global region already IS
# that window. Levels therefore run k = 0 .. vc_levels-1.
#
# Sizing worked example, terapool: 4 MiB L1 / 16 groups = 256 KiB per group,
# vc_levels = 2, so window 1 = 1 MiB at 0x40_0000 and window 0 = 256 KiB at
# 0x50_0000; the region ends at 0x54_0000, i.e. 5.25 MiB = 1.33 x 4 MiB.
# Enable the virtual-cluster remap at the three COMPUTE scrambler sites. On by
# default: it only decodes addresses above the global L1 region, so code that
# never touches a VC window runs cycle-identically either way. Needs a square
# mesh and tile_id_remap=0. The standalone replay testbench
# (hardware/tb/tb_vc_scrambler.sv) exercises the permuter either way.
virtual_cluster   ?= 1

num_y             := $(shell echo $$(( $(num_groups) / $(num_x) )))
vc_group_bytes    := $(shell echo $$(( $(num_banks_per_group) * $(l1_bank_size) )))
l1_total_bytes    := $(shell echo $$(( $(vc_group_bytes) * $(num_groups) )))
vc_levels         := $(shell awk 'BEGIN{n=($(num_x)<$(num_y))?$(num_x):$(num_y); k=0; while (n>1) {n/=2; k++}; print k}')
vc_level_list     := $(shell seq 0 $$(( $(vc_levels) - 1 )) 2>/dev/null)

# Per-level base/size, as `vc_window_base_<k>` / `vc_window_size_<k>`.
$(foreach k,$(vc_level_list),\
  $(eval vc_window_size_$(k) := $(shell awk 'BEGIN{printf "%d", $(vc_group_bytes) * (4 ^ $(k))}'))\
  $(eval vc_window_base_$(k) := $(shell awk 'BEGIN{acc=$(l1_total_bytes); for (j=$(vc_levels)-1; j>$(k); j--) acc += $(vc_group_bytes) * (4 ^ j); printf "%d", acc}')))

# End of the whole VC region. The RTL derives the same map independently and
# fails elaboration if the two disagree, so this one number pins the agreement.
vc_region_end     := $(shell awk 'BEGIN{acc=$(l1_total_bytes); for (j=0; j<$(vc_levels); j++) acc += $(vc_group_bytes) * (4 ^ j); printf "%d", acc}')

# Tile window, right after the level-0 window: one tile's L1, which the
# scrambler re-places into the requester's own tile. Only with vc_levels > 0.
vc_tile_window_base := $(vc_region_end)
vc_tile_window_size := $(shell echo $$(( $(vc_group_bytes) * $(num_groups) * $(num_cores_per_tile) / $(num_cores) )))
