# Copyright 2026 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

"""Static worst-case stack depth over the ELF's direct call graph.

The per-core stack is `stack_size` bytes of the sequential L1 region
(`config/config.mk:48`, `software/runtime/arch.ld.c:23`) and cannot grow
without moving `seq_mem_size`, which is a hardware parameter. Overflow is
silent: the guest keeps storing below its own region and the first symptom is
an invalid access inside `printf`. This gate refuses such a binary at build
time instead of after an hour of simulation.

Indirect calls are not resolvable here; `unresolved` reports how many were
seen so a caller can decide whether the bound is trustworthy.
"""

import re
import subprocess


HEADER = re.compile(r"^([0-9a-f]+) <([^>]+)>:$")
BODY = re.compile(r"^\s*[0-9a-f]+:\s+[0-9a-f]+\s+(\S+)\s*(.*)$")
FRAME = re.compile(r"^sp,sp,-(\d+)$")
# An auipc/jalr pair prints its resolved callee in a trailing comment, so the
# symbol is taken from the last <...> on the line either way.
TARGET = re.compile(r"<([^>]+)>\s*$")
CALLS = ("jal", "jalr", "j", "call", "tail")
# A tail call through auipc/jr (`jr imm(t1) # <callee>`) leaves the caller's
# frame popped, but it is counted as a full call edge: conservative, and the
# caller's frame is usually the smaller one. `jr ra` is a return; any other
# untargeted `jr` (a jump table or a call through a pointer) is unresolved.
TAIL_JUMPS = ("jr",)


def disassemble(objdump, binary):
    return parse(subprocess.check_output([objdump, "-d", str(binary)],
                                         text=True))


def parse(text):
    frames, edges, indirect, current = {}, {}, {}, None
    for line in text.splitlines():
        head = HEADER.match(line)
        if head:
            current = head[2]
            frames.setdefault(current, 0)
            edges.setdefault(current, set())
            indirect.setdefault(current, 0)
            continue
        body = BODY.match(line)
        if not current or not body:
            continue
        mnemonic, operands = body[1], body[2].strip()
        if mnemonic == "addi":
            size = FRAME.match(operands)
            if size:
                frames[current] = max(frames[current], int(size[1]))
        elif mnemonic in CALLS or mnemonic in TAIL_JUMPS:
            target = TARGET.search(operands)
            if not target:
                if mnemonic in CALLS or operands != "ra":
                    indirect[current] += 1
                continue
            name, _, offset = target[1].partition("+")
            # `j <self+0x..>` is an internal branch; `jal ra,<self>` is real
            # recursion and must stay visible to the cycle check.
            if name != current or not offset:
                edges[current].add(name)
    return frames, edges, indirect


def deepest(entry, frames, edges, indirect):
    """Longest simple path; recursion is reported rather than hidden."""
    best, worst, cycles, unresolved = {}, [], set(), 0

    def walk(name, stack):
        nonlocal worst, unresolved
        if name in stack:
            cycles.add(name)
            return 0, []
        if name in best:
            return best[name]
        own = frames.get(name, 0)
        unresolved += indirect.get(name, 0)
        depth, path = 0, []
        for callee in sorted(edges.get(name, ())):
            sub, trail = walk(callee, stack | {name})
            if sub > depth:
                depth, path = sub, trail
        result = (own + depth, [(name, own)] + path)
        best[name] = result
        return result

    total, path = walk(entry, frozenset())
    worst = path
    return {"bytes": total, "path": worst, "recursive": sorted(cycles),
            "unresolved": unresolved}


def check(objdump, binary, limit, entry="main"):
    frames, edges, indirect = disassemble(objdump, binary)
    if entry not in frames:
        raise ValueError(f"Entry symbol {entry} is absent from {binary}")
    result = deepest(entry, frames, edges, indirect)
    result["limit"] = limit
    result["fits"] = result["bytes"] <= limit
    return result
