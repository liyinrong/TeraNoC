// Copyright 2021 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

/* This file will get processed by the precompiler to expand all macros. */

MEMORY {
  l1 (R) : ORIGIN = 0x00000000, LENGTH = (NUM_CORES * N_FU * BANKING_FACTOR * L1_BANK_SIZE)
#if defined(DOMAINS) && DOMAINS > 1
  // A domain's interleaved data lives in its top-level VC window, above the
  // bank rows of the per-tile sequential regions (stacks stay physical).
  l1d (R) : ORIGIN = DOMAIN_L1_BASE + NUM_CORES * SEQ_MEM_SIZE,
            LENGTH = (NUM_CORES * N_FU * BANKING_FACTOR * L1_BANK_SIZE) - NUM_CORES * SEQ_MEM_SIZE
#endif
  l2     : ORIGIN = L2_BASE   , LENGTH = L2_SIZE
  rom (R): ORIGIN = BOOT_ADDR , LENGTH = 0x00001000
}

#if defined(DOMAINS) && DOMAINS > 1
REGION_ALIAS("l1_data", l1d);
#else
REGION_ALIAS("l1_data", l1);
#endif

SECTIONS {
  // Start end end of memories
  __l1_start = ORIGIN(l1);
  __l1_end = ORIGIN(l1) + LENGTH(l1);
  __l2_start = ORIGIN(l2);
  __l2_end = ORIGIN(l2) + LENGTH(l2);
  __rom_start = ORIGIN(rom);
  __rom_end = ORIGIN(rom) + LENGTH(rom);

  // Stack size
  __stack_start = __l1_start;
  __stack_end = __l1_start + (NUM_CORES * STACK_SIZE);

  // Sequential region size
  __seq_start = __l1_start;
  __seq_end = __l1_start + (NUM_CORES * SEQ_MEM_SIZE);

  // Heap size (start address is re-assigned in link.ld)
  __heap_start = __l1_start;
  __heap_end = ORIGIN(l1_data) + LENGTH(l1_data);

#if defined(DMA_DOMAIN_HOME_LO)
  // A raised domain home base (runtime/dma_fold.h) must sit above the image.
  ASSERT(__l2_bss_end <= ORIGIN(l2) + DMA_DOMAIN_HOME_LO,
         "L2 image reaches into the domain home region (DMA_DOMAIN_HOME_LO)")
#elif defined(DOMAIN_L2_EXACT) && DOMAIN_L2_EXACT
  // The exact fold at the default home base (64 KiB): any image past it lies in
  // folded L2, where wrapped home writes overwrite code. Set DMA_DOMAIN_HOME_LO.
  ASSERT(__l2_bss_end <= ORIGIN(l2) + 0x10000,
         "DOMAIN_L2_EXACT: L2 image reaches into the domain home region; set DMA_DOMAIN_HOME_LO above it")
#endif

  // Testbench UART, off-cluster behind the SoC xbar's default port.
  // Kept in step with UARTBaseAddr in hardware/tb/mempool_tb.sv.
  fake_uart              = PERIPH_BASE + 0x01000000;
}
