# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Yinrong Li, ETH Zurich
#
# TensorPool-256 benchmark flavor with both RedMulE and Spatz enabled. This is
# the software-build target used by the dense LLM benchmarks: RedMulE executes
# GEMM/GEMV stages while Spatz handles RMSNorm, RoPE, softmax, and residuals.

include $(MEMPOOL_DIR)/config/tensorpool256.mk

config_build_path ?= tensorpool256_spatz4

# Use the tp256_spatz4-specific FlooNoC config: its L2/HBM System Address Map
# matches this flavor's 512 MiB L2 (l2_size below). Without this override the
# included tensorpool256.mk falls back to floo_noc_terapool.yml, whose SAM lays
# L2 out as 1 MiB/bank (16 MiB total) -> L2 accesses above 16 MiB misroute to the
# wrong bank, corrupting instruction fetch and wedging the boot/barrier.
floo_noc_cfg := floo_noc_tensorpool256_spatz4.yml

# Real-size dense-LLM weights, KV state, and spill buffers live in L2. Keep the
# inherited TensorPool L1 geometry and extend L2 up to the bootrom boundary.
l2_size := 536870912

spatz ?= 1
vlen ?= 512
n_ipu ?= 4
n_fpu ?= 4

rvf ?= 1
rvd ?= 0
xpulpimg ?= 0
zfinx ?= 0
