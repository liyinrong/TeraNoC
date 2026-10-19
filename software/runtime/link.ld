/* Copyright 2021 ETH Zurich and University of Bologna. */
/* Licensed under the Apache License, Version 2.0, see LICENSE for details. */
/* SPDX-License-Identifier: Apache-2.0 */

/* Include the file processed by the preprocessor called arch.ld.c */
INCLUDE arch.ld

OUTPUT_ARCH("riscv")
ENTRY(_start)


SECTIONS {
  /* Sequential region on L1 */
  /* Use dynamic allocation with `domain_malloc` */
  .l1_seq (NOLOAD): {
    . = __seq_end;
  } > l1

  /* Interleaved region on L1 */
  .l1 (NOLOAD): {
    *(.l1_prio)
    *(.l1)
    *(.bss)
    /* Phase scratch last: a phase may overlay it and the heap above it. */
    *(.l1_tail)
    __l1_alloc_base = ALIGN(0x10);
    __heap_start = .;
  } > l1_data /* = l1, or a domain's top-level VC window (arch.ld.c) */

  /* Instructions on L2 */
  .text : {
    *(.text.init)
    *(.text.startup)
    /* Hot DMA collective code, kept contiguous and line-aligned so it spans as
       few 64-byte instruction-cache lines as possible (dma_collective.c). */
    . = ALIGN(0x40);
    __dma_hot_start = .;
    *(.text.dma_hot)
    . = ALIGN(0x40);
    __dma_hot_end = .;
    *(.text)
    *(.text.dma_cold)
    *(.text.end)
    . = ALIGN(0x40);
    _etext = .;
  } > l2

  /* RO Data on L2 */
  .rodata : {
    *(.rodata .rodata.* .gnu.linkonce.r.*)
  } > l2
  .rodata1 : {
    *(.rodata1)
    /* Align non-cacheable regions to 4KiB boundary */
    . = ALIGN(0x1000);
    _erodata = .;
  } > l2

  /* Data on L2 */
  .data           : {
    . = ALIGN(0x10);
    *(.data)
  } > l2
  .sdata2 : {
    *(.sdata2 .sdata2.* .gnu.linkonce.s2.*)
  } > l2
  .sdata : {
    __global_pointer$ = . + 0x800;
    *(.srodata.cst16) *(.srodata.cst8) *(.srodata.cst4) *(.srodata.cst2) *(.srodata .srodata.*)
    *(.sdata .sdata.* .gnu.linkonce.s.*)
  } > l2

  .bss : {
    __bss_start = .;
    *(.bss)
    *(.sbss2 .sbss2.* .gnu.linkonce.sb2.*);
    __bss_end = .;
  } > l2

  .l2 : {
    . = ALIGN(0x10);
    *(.l2)
    __l2_alloc_base = ALIGN(0x10);
  } > l2

  /* Uninitialized L2 (NOLOAD): real-size weights reserved here carry no ELF
     data, so a full Llama layer costs address space only (perf-only runs). */
  .l2_bss (NOLOAD) : {
    . = ALIGN(0x10);
    *(.l2_bss)
    __l2_bss_end = .;
  } > l2

  .comment : {
    *(.comment)
  } > l2
}
