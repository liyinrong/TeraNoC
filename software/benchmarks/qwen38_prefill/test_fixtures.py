# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import tempfile
import unittest

import numpy as np

from fixtures import generate, load_tensor


class FixtureTests(unittest.TestCase):
    def test_packs_are_reproducible_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            a = generate(root / "a", "attention", tokens=3)
            b = generate(root / "b", "attention", tokens=3)
            self.assertEqual(a, b)
            out = load_tensor(root / "a", a["tensors"]["output"])
            self.assertEqual(out.shape, (1, 3, 256))
            self.assertTrue(np.isfinite(out).all())
            path = root / "a" / a["tensors"]["output"]["file"]
            data = bytearray(path.read_bytes())
            data[0] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "Hash mismatch"):
                load_tensor(root / "a", a["tensors"]["output"])
            with self.assertRaises(FileExistsError):
                generate(root / "b", "attention", tokens=3)

    def test_gdn_pack_contains_persistent_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "gdn"
            manifest = generate(root, "gdn", tokens=5)
            recurrent = load_tensor(
                root, manifest["tensors"]["state_recurrent"])
            self.assertEqual(recurrent.dtype, np.dtype("<f4"))
            self.assertEqual(recurrent.shape, (1, 6, 32, 32))
            self.assertTrue(np.any(recurrent != 0))


if __name__ == "__main__":
    unittest.main()
