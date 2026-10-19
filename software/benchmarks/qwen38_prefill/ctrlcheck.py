# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
"""Scheduler-core ISA gate over an ELF's direct call graph.

The scheduler core of a 1 + 3 tile is rv32ima: an F, V, Zfh or Xpulp
instruction, or an fcsr access, traps to the boot ROM and parks the core,
which shows up as a hang. This gate walks every function reachable from the
scheduler entry and rejects any such instruction. It reads the disassembly of
the Xpulp-aware GCC objdump, which prints RVV either as raw words or as pv.*;
both are rejected, so the check errs on the safe side.

Indirect calls are not followed; `unresolved` counts them.

Usage: ctrlcheck.py OBJDUMP ELF [ENTRY ...]   (default entry avs_sched_stream)
"""
import re
import subprocess
import sys

from stackdepth import parse

HEADER = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
BODY = re.compile(r"^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)\s*(.*)$")
FP_CSRS = re.compile(r"\b(fcsr|frm|fflags|0x00[123])\b")
SCALAR_F = re.compile(r"^f(?!ence)")


def bad(mnemonic, operands):
    """Why an instruction is illegal on rv32ima, or None."""
    if mnemonic.startswith((".", "unknown", "0x")) or mnemonic.isdigit():
        return "unknown encoding"
    if SCALAR_F.match(mnemonic):
        return "F/Zfh"
    if mnemonic.startswith("v"):
        return "V"
    if mnemonic.startswith(("pv.", "p.", "lp.")):
        return "Xpulp"
    if mnemonic.startswith("csr") and FP_CSRS.search(operands):
        return "fcsr"
    return None


def instructions(text):
    """Per function: (address, mnemonic, operands) of every instruction."""
    out, current = {}, None
    for line in text.splitlines():
        head = HEADER.match(line)
        if head:
            current = head[2]
            out.setdefault(current, [])
            continue
        body = BODY.match(line)
        if current and body:
            out[current].append((body[1], body[3], body[4].strip()))
    return out


def analyze(text, entries):
    """The call closure of `entries` in a disassembly and its violations."""
    _, edges, indirect = parse(text)
    code = instructions(text)
    missing = [e for e in entries if e not in code]
    if missing:
        raise ValueError(f"Entry symbols {missing} are absent")
    seen, stack = set(), list(entries)
    while stack:
        name = stack.pop()
        if name in seen or name not in code:
            continue
        seen.add(name)
        stack.extend(edges.get(name, ()))
    violations = [(name, addr, mnem, ops, why)
                  for name in sorted(seen)
                  for addr, mnem, ops in code[name]
                  if (why := bad(mnem, ops))]
    return {"functions": sorted(seen),
            "unresolved": sum(indirect.get(n, 0) for n in seen),
            "violations": violations,
            "clean": not violations}


def check(objdump, binary, entries=("avs_sched_stream",)):
    text = subprocess.check_output([objdump, "-d", str(binary)], text=True)
    return analyze(text, entries)


def main(argv):
    if len(argv) < 3:
        raise SystemExit(__doc__)
    result = check(argv[1], argv[2], tuple(argv[3:]) or ("avs_sched_stream",))
    print(f"{len(result['functions'])} functions reachable, "
          f"{result['unresolved']} indirect calls not followed")
    for name, addr, mnem, ops, why in result["violations"]:
        print(f"  {why:16s} {name}+{addr}: {mnem} {ops}")
    print("clean" if result["clean"] else
          f"{len(result['violations'])} illegal instructions")
    return 0 if result["clean"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
