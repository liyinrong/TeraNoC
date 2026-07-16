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
