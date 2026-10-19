# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Invoke with make -f /path/to/this/Makefile from an isolated build directory.
BENCH_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ifndef MEMPOOL_DIR
$(error Pass MEMPOOL_DIR=/absolute/path/to/TeraNoC-dev)
endif
COMPILER := llvm
include $(MEMPOOL_DIR)/software/runtime/runtime.mk

# Runtime and linker products must never leak into the shared source tree.
RUNTIME_NAMES := alloc async_task dma_collective mempool_checks printf serial string synchronization vc
RUNTIME := $(addprefix $(CURDIR)/,$(addsuffix .c.o,$(RUNTIME_NAMES))) $(CURDIR)/crt0.S.o
LINKER_SCRIPT := $(CURDIR)/arch.ld

.PHONY: all
all: softmax_probe.elf

bandwidth_probe.o: $(BENCH_DIR)/bandwidth_probe.c
	$(RISCV_CC) $(RISCV_CCFLAGS) -fno-vectorize -fno-slp-vectorize -c $< -o $@

bandwidth_probe.elf: bandwidth_probe.o $(RUNTIME) $(LINKER_SCRIPT)
	$(RISCV_CC) -o $@ $< $(RUNTIME) $(RISCV_LDFLAGS) -L$(CURDIR) -T$(ROOT_DIR)/link.ld

$(CURDIR)/%.c.o: $(ROOT_DIR)/%.c
	$(RISCV_CC) $(RISCV_CCFLAGS) -c $< -o $@

$(CURDIR)/crt0.S.o: $(ROOT_DIR)/crt0.S
	$(RISCV_CC) $(RISCV_CCFLAGS) -c $< -o $@

$(LINKER_SCRIPT): $(ROOT_DIR)/arch.ld.c
	$(RISCV_CC) -P -E $(DEFINES) $< -o $@

softmax_probe.o: $(BENCH_DIR)/softmax_probe.c softmax_data.h $(KERNELS_DIR)/baremetal/mempool_qwen38_online_softmax.h $(KERNELS_DIR)/baremetal/mempool_llm_ffn_f16.h
	$(RISCV_CC) $(RISCV_CCFLAGS) -I$(CURDIR) -fno-vectorize -fno-slp-vectorize -c $< -o $@

softmax_probe.elf: softmax_probe.o $(RUNTIME) $(LINKER_SCRIPT)
	$(RISCV_CC) -o $@ $< $(RUNTIME) $(RISCV_LDFLAGS) -L$(CURDIR) -T$(ROOT_DIR)/link.ld

LAYER_SOURCES := $(addprefix $(BENCH_DIR)/,layer_resident.c q38_vc.c q38_projection.c q38_ffn.c q38_ffn_l2.c q38_attention.c q38_attn_vc.c q38_gdn.c)
LAYER_SOURCES += $(addprefix $(BENCH_DIR)/,q38_attn_vc_epi.c q38_attn_vc_sweep.c q38_attn_vc_softmax.c)

layer_resident.o: $(LAYER_SOURCES) layer_data.h $(KERNELS_DIR)/baremetal/mempool_qwen38_vector.h $(KERNELS_DIR)/baremetal/mempool_qwen38_online_softmax.h $(KERNELS_DIR)/baremetal/mempool_qwen38_kv.h $(KERNELS_DIR)/baremetal/mempool_llm_ffn_f16.h
	$(RISCV_CC) $(RISCV_CCFLAGS) -I$(CURDIR) -fno-vectorize -fno-slp-vectorize -c $< -o $@

layer_resident.elf: layer_resident.o dma_stage_queue.o $(RUNTIME) $(LINKER_SCRIPT)
	$(RISCV_CC) -o $@ layer_resident.o dma_stage_queue.o $(RUNTIME) $(RISCV_LDFLAGS) -L$(CURDIR) -T$(ROOT_DIR)/link.ld

dma_coordinator_probe.o: $(BENCH_DIR)/dma_coordinator_probe.c
	$(RISCV_CC) $(RISCV_CCFLAGS) -fno-vectorize -fno-slp-vectorize -c $< -o $@

dma_stage_queue.o: $(BENCH_DIR)/dma_stage_queue.c $(BENCH_DIR)/dma_stage_queue.h
	$(RISCV_CC) $(RISCV_CCFLAGS) -fno-vectorize -fno-slp-vectorize -c $< -o $@

dma_coordinator_probe.elf: dma_coordinator_probe.o $(RUNTIME) $(LINKER_SCRIPT)
	$(RISCV_CC) -o $@ dma_coordinator_probe.o $(RUNTIME) $(RISCV_LDFLAGS) -L$(CURDIR) -T$(ROOT_DIR)/link.ld

dma_stage_queue_probe.o: $(BENCH_DIR)/dma_stage_queue_probe.c $(BENCH_DIR)/dma_stage_queue.h
	$(RISCV_CC) $(RISCV_CCFLAGS) -fno-vectorize -fno-slp-vectorize -c $< -o $@

dma_stage_queue_probe.elf: dma_stage_queue_probe.o dma_stage_queue.o $(RUNTIME) $(LINKER_SCRIPT)
	$(RISCV_CC) -o $@ dma_stage_queue_probe.o dma_stage_queue.o $(RUNTIME) $(RISCV_LDFLAGS) -L$(CURDIR) -T$(ROOT_DIR)/link.ld

triple_engine_probe.o: $(BENCH_DIR)/triple_engine_probe.c $(KERNELS_DIR)/baremetal/mempool_qwen38_vector.h
	$(RISCV_CC) $(RISCV_CCFLAGS) -fno-vectorize -fno-slp-vectorize -c $< -o $@

triple_engine_probe.elf: triple_engine_probe.o $(RUNTIME) $(LINKER_SCRIPT)
	$(RISCV_CC) -o $@ $< $(RUNTIME) $(RISCV_LDFLAGS) -L$(CURDIR) -T$(ROOT_DIR)/link.ld
