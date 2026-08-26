// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Author: Samuel Riedel, ETH Zurich

#ifndef _ADDRMAP_H_
#define _ADDRMAP_H_

// The system address map lives in config/config.mk; `periph_base` reaches the
// runtime as PERIPH_BASE through software/runtime/runtime.mk. Only the window
// base moves -- every register offset inside it is unchanged.
#ifndef PERIPH_BASE
#error "PERIPH_BASE undefined: the runtime must be built through runtime.mk"
#endif

#define CONTROL_REGISTER_OFFSET (PERIPH_BASE)
#define REDMULE_BASE_ADD (PERIPH_BASE + 0x20000)

// Per-tile fractal_sync shim. DERIVED from the RedMulE window exactly as the
// RTL derives it (mempool_pkg.sv: FSyncBaseAddr = RMBaseAddr + 0x1_0000), so
// the two cannot drift and both follow the pending address-map move
// (branch addrmap-move: peripherals -> PERIPH_BASE, default 0x7000_0000).
// NOTE the constants above are still literals and WILL need rebasing when that
// branch merges; deriving this one keeps the list of things to fix shorter.
#define FSYNC_BASE (REDMULE_BASE_ADD + 0x10000)

#include "control_registers.h"

#endif // _DMA_H_
