// Copyright 2025 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Yinrong Li, ETH Zurich

// L1-budget helpers for the dense-LLM benchmarks.
//
// The total L1 scratchpad size is already a compile-time constant in software:
// the config makefiles emit NUM_BANKS and L1_BANK_SIZE (via runtime.mk), and
// the interleaved region is what is left after the per-core sequential region
// (stacks). These macros expose that budget so the kernels can (a) size and
// (b) statically assert their resident working set, and so the workload SHAPE
// stays decoupled from the TILE size.
//
// The one derived tile knob is the FFN weight-streaming PANEL_I. It is computed
// from this budget by the data generator (the build-time tiling solver, see
// gendatalib.py::_llm_pick_panel_i), which receives L1_SIZE_BYTES / PORT_WIDTH
// from runtime.mk and emits PANEL_I/NUM_IP into the data header. The asserts
// below are the defence-in-depth check that the emitted tiling actually fits
// the link-time budget of the active config.

#pragma once

#ifndef NUM_BANKS
#error "mempool_llm_tiling.h needs NUM_BANKS (from the config via runtime.mk)"
#endif

// Total physical L1 (bytes) = NUM_BANKS * L1_BANK_SIZE.
#define L1_SIZE_BYTES ((long)NUM_BANKS * (long)L1_BANK_SIZE)

// Sequential region reserved for the per-core stacks (bytes).
#define L1_SEQ_BYTES ((long)NUM_CORES * (long)SEQ_MEM_SIZE)

// Interleaved L1 available to the app's .l1 / .l1_prio tiles (bytes). A small
// slack is kept for runtime .l1 allocations. The data generator uses a
// conservative 0.85 factor on total L1 because it does not receive the stack
// size separately.
#define L1_TILE_BUDGET_BYTES ((L1_SIZE_BYTES - L1_SEQ_BYTES) * 9 / 10)

// True when a resident working set of `elems` f16 elements fits the budget.
#define LLM_F16_FITS_L1(elems) (((long)(elems)*2) <= L1_TILE_BUDGET_BYTES)
