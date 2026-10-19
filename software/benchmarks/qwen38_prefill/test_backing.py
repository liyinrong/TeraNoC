# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from elftools.elf.elffile import ELFFile

from backing import DataElf, boot_rom, validate_loads, memory_layout


class BackingTests(unittest.TestCase):
    def test_extended_map_preserves_uart(self):
        self.assertEqual(memory_layout(512), (0x20000000, 0xa0000000))
        self.assertEqual(memory_layout(1024), (0x40000000, 0xd0000000))

    def test_boot_rom(self):
        # Stock pulp/teranoc_v2/rom.bin, linked at 0xa0000000.
        self.assertEqual(boot_rom(0xa0000000).hex(),
                         "170500e0130505007300501067000500")
        self.assertEqual(boot_rom(0xd0000000)[:4].hex(), "170500b0")
        with self.assertRaises(ValueError):
            boot_rom(0xd0000800)
        with self.assertRaises(ValueError):
            memory_layout(1536)

    def test_data_elf_bits_alignment_and_load_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a = np.array([0x8000, 0x3c00, 0x7bff], '<u2').view('<f2')
            pack = DataElf()
            declarations = io.StringIO()
            pack.add('weights', a, declarations)
            pack.add('gold', np.array([1.25, -0.0], '<f4'), declarations)
            sidecar = root / 'data.elf'
            manifest = pack.write(sidecar)
            self.assertIn('0x90000000u', declarations.getvalue())
            with sidecar.open('rb') as stream:
                elf = ELFFile(stream)
                self.assertEqual(elf.header['e_entry'], 0)
                segments = list(elf.iter_segments())
                self.assertEqual(segments[0].data(), a.tobytes())
                self.assertEqual(segments[1]['p_paddr'], 0x90010000)
            self.assertEqual(manifest['end'], 0x90010008)
            code = DataElf(0x80000000, 0x90000000)
            code.add('code', a, io.StringIO())
            binary = root / 'code.elf'
            code.write(binary)
            with binary.open("r+b") as stream:
                stream.seek(24)
                stream.write(struct.pack("<I", 0x80000000))
            validate_loads(binary, sidecar, 0x80000000, 0xa0000000)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                duplicate = root / 'overlap.elf'
                code.write(duplicate)
                validate_loads(binary, duplicate, 0x80000000, 0xa0000000)
            with self.assertRaisesRegex(ValueError, 'outside L2'):
                validate_loads(binary, sidecar, 0x80000000, 0x90000000)
            with sidecar.open('r+b') as stream:
                stream.seek(24)
                stream.write(struct.pack('<I', 0x90000000))
            with self.assertRaisesRegex(ValueError, 'entry'):
                validate_loads(binary, sidecar, 0x80000000, 0xa0000000)

    def test_reserved_tensors_place_but_never_write(self):
        pack = DataElf()
        declarations = io.StringIO()
        pack.reserve('weights', 'float16', 3 * 2, declarations)
        pack.reserve('gold', 'float32', 2 * 4, declarations)
        self.assertIn('0x90010000u', declarations.getvalue())
        self.assertEqual(pack.cursor, 0x90010008)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'no data'):
                pack.write(Path(temp) / 'reserved.elf')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pack.reserve('gold', 'float32', 4, declarations)
        with self.assertRaises(ValueError):
            pack.reserve('empty', 'float16', 0, declarations)

    def test_range_and_duplicate_guards(self):
        pack = DataElf(0x90000000, 0x90000002)
        pack.add('one', np.ones(1, '<f2'), io.StringIO())
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            pack.add('one', np.ones(1, '<f2'), io.StringIO())
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            pack.add('two', np.ones(1, '<f2'), io.StringIO())
        with self.assertRaises(ValueError):
            DataElf(0x90000001)


if __name__ == '__main__':
    unittest.main()
