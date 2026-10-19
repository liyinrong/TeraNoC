# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import textwrap
import unittest

from stackdepth import deepest, parse


LISTING = textwrap.dedent("""\
    00000100 <main>:
         100:\t12345678          \taddi\tsp,sp,-48
         104:\t12345678          \tjal\tra,200 <alpha>
         108:\t12345678          \tjalr\t-16(ra) # 00000300 <beta>
         10c:\t12345678          \tbeqz\ta0,110 <main+0x10>
         110:\t12345678          \tj\t114 <main+0x14>

    00000200 <alpha>:
         200:\t12345678          \taddi\tsp,sp,-240
         204:\t12345678          \tjalr\ta5
         208:\t12345678          \tjal\tra,300 <beta>

    00000300 <beta>:
         300:\t12345678          \taddi\tsp,sp,-16
    """)


class StackDepthTests(unittest.TestCase):
    def test_frames_direct_and_indirect_calls_are_separated(self):
        frames, edges, indirect = parse(LISTING)
        self.assertEqual(frames, {"main": 48, "alpha": 240, "beta": 16})
        self.assertEqual(edges["main"], {"alpha", "beta"})
        self.assertEqual(edges["alpha"], {"beta"})
        # A self branch is not a call; a register jump has no target.
        self.assertEqual(edges["beta"], set())
        self.assertEqual(indirect, {"main": 0, "alpha": 1, "beta": 0})

    def test_deepest_path_sums_the_worst_chain(self):
        result = deepest("main", *parse(LISTING))
        self.assertEqual(result["bytes"], 304)
        self.assertEqual([name for name, _ in result["path"]],
                         ["main", "alpha", "beta"])
        self.assertEqual(result["unresolved"], 1)
        self.assertEqual(result["recursive"], [])

    def test_recursion_is_reported_and_terminates(self):
        listing = LISTING + textwrap.dedent("""\

            00000400 <loop>:
                 400:\t12345678          \taddi\tsp,sp,-32
                 404:\t12345678          \tjal\tra,400 <loop>
            """)
        result = deepest("loop", *parse(listing))
        self.assertEqual(result["bytes"], 32)
        self.assertEqual(result["recursive"], ["loop"])

    def test_tail_call_through_jr_is_a_call_edge(self):
        listing = LISTING + textwrap.dedent("""\

            00000500 <entry>:
                 500:\t12345678          \taddi\tsp,sp,-48
                 504:\t12345678          \taddi\tsp,sp,48
                 508:\t12345678          \tjr\t-512(t1) # 00000200 <alpha>
                 50c:\t12345678          \tjr\tra
                 510:\t12345678          \tjr\ta5
            """)
        frames, edges, indirect = parse(listing)
        self.assertEqual(edges["entry"], {"alpha"})
        # `jr ra` returns; `jr a5` is unresolved.
        self.assertEqual(indirect["entry"], 1)
        self.assertEqual(deepest("entry", frames, edges, indirect)["bytes"],
                         48 + 240 + 16)

    def test_internal_branch_is_not_a_self_call(self):
        _, edges, _ = parse(LISTING)
        self.assertNotIn("main", edges["main"])


if __name__ == "__main__":
    unittest.main()
