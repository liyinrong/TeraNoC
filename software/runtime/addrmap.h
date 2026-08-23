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

#include "control_registers.h"

#endif // _DMA_H_
