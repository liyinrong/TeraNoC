// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#if defined(DOMAINS) && DOMAINS > 1
#include "runtime.h"
#endif

extern char fake_uart;

void _putchar(char character) {
#if defined(DOMAINS) && DOMAINS > 1
  // The domains run the same program; only domain 0 reports.
  if (mempool_get_domain_id())
    return;
#endif
  // send char to console
  fake_uart = character;
}
