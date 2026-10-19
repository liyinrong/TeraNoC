# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Data-only ELF32 backing, kept separate from the executable and compiler."""

import hashlib
from pathlib import Path
import struct


GUEST_TYPE = {"float16": "__fp16", "float32": "float", "uint32": "uint32_t"}
WIDTH = {"float16": 2, "float32": 4, "uint32": 4}


def memory_layout(l2_mib):
    # Scrambler bit permutation requires a power-of-two L2 aperture. UART
    # remains fixed at 0xc0000000, so ROM cannot simply follow extended L2.
    if l2_mib not in (512, 1024):
        raise ValueError("Supported L2 capacities: 512/1024 MiB")
    size = l2_mib * 1024 * 1024
    rom = 0xa0000000 if l2_mib == 512 else 0xd0000000
    end = 0x80000000 + size
    if rom < end or rom <= 0xc0000000 < rom + 0x1000:
        raise ValueError("ROM overlaps L2 or UART")
    return size, rom


def boot_rom(rom, l2_base=0x80000000):
    """Dev boot ROM image (pulp/teranoc_v2/bootrom.S) for a ROM at `rom`.

    The ROM parks in WFI and then jumps PC-relative to L2, so the stock image,
    linked at 0xa0000000, only works there; extended L2 moves the ROM.
    """
    offset = (l2_base - rom) % 2 ** 32
    if offset % 4096:
        raise ValueError("ROM-to-L2 offset must be 4 KiB aligned")
    words = (offset | 10 << 7 | 0x17,  # auipc a0, (l2_base - rom) >> 12
             0x00050513,  # addi a0, a0, 0
             0x10500073,  # wfi
             0x00050067)  # jr a0
    return b"".join(word.to_bytes(4, "little") for word in words)


class DataElf:
    def __init__(self, base=0x90000000, limit=0xa0000000):
        if base % 65536 or not 0 <= base < limit <= 2 ** 32:
            raise ValueError("Invalid backing address range")
        self.base, self.limit = base, limit
        self.cursor = base
        self.arrays = []

    def add(self, name, value, stream):
        if value.dtype.name not in WIDTH or not value.size:
            raise ValueError("Backing tensors must be nonempty FP16/FP32/U32")
        self.reserve(name, value.dtype.name, value.nbytes, stream)
        self.arrays[-1] = (name, self.arrays[-1][1], value)

    def reserve(self, name, dtype, nbytes, stream):
        """Place a tensor and declare its pointer without holding its data.

        The footprint probe needs the addresses and the capacity verdict for
        geometries it cannot materialize; `write` refuses such a pack."""
        if any(item[0] == name for item in self.arrays):
            raise ValueError("Duplicate backing symbol")
        if dtype not in WIDTH or nbytes <= 0:
            raise ValueError("Backing tensors must be nonempty FP16/FP32/U32")
        address = (self.cursor + 65535) & -65536
        self.cursor = address + nbytes
        if self.cursor > self.limit:
            raise ValueError("Tensor backing exceeds configured L2 aperture")
        self.arrays.append((name, address, None))
        stream.write("#define %s ((%s *)0x%08xu)\n" % (
            name, GUEST_TYPE[dtype], address))

    def write(self, path):
        # One read/write PT_LOAD per tensor, no entry point or section table.
        if len(self.arrays) >= 65535:
            raise ValueError("Too many ELF segments")
        if any(value is None for _, _, value in self.arrays):
            raise ValueError("Reserved tensors carry no data to write")
        offset = (52 + 32 * len(self.arrays) + 65535) & -65536
        headers, records = [], []
        for name, address, value in self.arrays:
            headers.append(struct.pack('<8I', 1, offset, address, address,
                                       value.nbytes, value.nbytes, 6, 65536))
            records.append({"name": name, "address": address,
                            "offset": offset, "bytes": value.nbytes})
            offset = (offset + value.nbytes + 65535) & -65536
        ident = b'\x7fELF\x01\x01\x01' + bytes(9)
        with Path(path).open('wb') as output:
            output.write(struct.pack('<16sHHIIIIIHHHHHH', ident, 2, 243, 1,
                                     0, 52, 0, 0, 52, 32,
                                     len(headers), 0, 0, 0))
            output.writelines(headers)
            for (_, _, value), record in zip(self.arrays, records):
                output.seek(record['offset'])
                data = value.astype(value.dtype.newbyteorder('<'), copy=False)
                data.tofile(output)
        with Path(path).open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        return {"base": self.base, "end": self.cursor, "segments": records,
                "sha256": digest}


def validate_loads(binary, sidecar, l2_base, l2_end):
    """Reject unsafe inputs before the loader can warn and continue."""
    from elftools.elf.elffile import ELFFile

    ranges = []
    for path in (binary, sidecar):
        with Path(path).open('rb') as source:
            elf = ELFFile(source)
            if (elf.elfclass != 32 or not elf.little_endian or
                    elf.header["e_machine"] != "EM_RISCV"):
                raise ValueError("Expected little-endian ELF32")
            if path == binary and elf.header["e_entry"] != l2_base:
                raise ValueError("Executable entry differs from loader entry")
            if path == sidecar and elf.header['e_entry'] != 0:
                raise ValueError("Data sidecar must have no executable entry")
            for seg in elf.iter_segments():
                if seg['p_type'] != 'PT_LOAD' or seg['p_memsz'] == 0:
                    continue
                start = seg['p_paddr']
                end = start + seg['p_memsz']
                file_end = seg["p_offset"] + seg["p_filesz"]
                if file_end > Path(path).stat().st_size:
                    raise ValueError("Truncated ELF load segment")
                if seg['p_filesz'] > seg['p_memsz']:
                    raise ValueError("ELF file segment exceeds memory extent")
                if path == sidecar and not l2_base <= start < end <= l2_end:
                    raise ValueError("Sidecar segment outside L2")
                for old_start, old_end in ranges:
                    if start < old_end and old_start < end:
                        raise ValueError("ELF load segments overlap")
                ranges.append((start, end))
