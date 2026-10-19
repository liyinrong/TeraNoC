# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import textwrap
import unittest

from ctrlcheck import analyze, bad


LISTING = textwrap.dedent("""\
    00000100 <sched>:
         100:\t12345678          \taddi\tsp,sp,-16
         104:\t12345678          \tjal\tra,200 <helper>
         108:\t12345678          \tfence\trw,rw
         10c:\t12345678          \tamoadd.w\ta0,a1,(a2)

    00000200 <helper>:
         200:\t12345678          \tlw\ta0,0(a1)
         204:\t12345678          \tcsrr\ta0,mhartid
         208:\t12345678          \tret

    00000300 <compute>:
         300:\t12345678          \tfmv.w.x\tft0,zero
         304:\t5208a157          \tpv.sll.c\tsp,a7,zero
    """)


class CtrlCheckTests(unittest.TestCase):
    def test_closure_excludes_unreachable_compute_code(self):
        result = analyze(LISTING, ("sched",))
        self.assertEqual(result["functions"], ["helper", "sched"])
        self.assertTrue(result["clean"])

    def test_reachable_fp_and_xpulp_are_rejected(self):
        listing = LISTING.replace("jal\tra,200 <helper>",
                                  "jal\tra,300 <compute>")
        result = analyze(listing, ("sched",))
        self.assertFalse(result["clean"])
        self.assertEqual([v[4] for v in result["violations"]],
                         ["F/Zfh", "Xpulp"])

    def test_classification(self):
        self.assertEqual(bad("vsetvli", "t0,a0,e16,m2"), "V")
        self.assertEqual(bad("0xa27c257", ""), "unknown encoding")
        self.assertEqual(bad("csrw", "fcsr,a0"), "fcsr")
        self.assertEqual(bad("flw", "ft0,0(a0)"), "F/Zfh")
        self.assertIsNone(bad("fence.i", ""))
        self.assertIsNone(bad("csrr", "a0,mhartid"))
        self.assertIsNone(bad("mul", "a0,a1,a2"))

    def test_missing_entry_is_an_error(self):
        with self.assertRaises(ValueError):
            analyze(LISTING, ("absent",))


if __name__ == "__main__":
    unittest.main()
