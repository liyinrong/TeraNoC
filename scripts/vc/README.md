<!--
Copyright 2026 ETH Zurich and University of Bologna.
Licensed under the Apache License, Version 2.0, see LICENSE for details.
SPDX-License-Identifier: Apache-2.0
-->

# Virtual-cluster address permuter: reference and golden vectors

`vc_morton.py` is the Python reference for the virtual-cluster (VC) address
permuter in `hardware/src/mempool_addr_scrambler.sv`, with Morton (Z-order)
group ids. It checks the mapping and emits the golden vectors that
`hardware/tb/tb_vc_scrambler.sv` replays against the RTL.

## Files

| Path | What it is |
|---|---|
| `vc_morton.py` | reference permuter, its checks, and the vector generator |
| `golden/vc_golden_morton_<geometry>_v1.txt` | vectors for `terapool`, `dse_s1_winner` and `monsterpool8x8` |

## Running

From the repository root, in the `teranoc` conda environment:

```bash
python3 scripts/vc/vc_morton.py                           # check every geometry
python3 scripts/vc/vc_morton.py --emit scripts/vc/golden  # also regenerate the vectors
make -C hardware vc-replay config=minpool                 # replay them on the RTL
```

`vc_morton.py` exits non-zero on any failure and writes vectors only when every
check passes. The replay testbench instantiates its own geometries, so any
config compiles it.

## What the permuter does

A level-k virtual cluster is a 2^k x 2^k square of groups. Morton ids make its
groups one contiguous id range, so the requester's rectangle base is a single
mask on the group id. A window-relative offset is split as

```
[ row | vc_group (2k bits) | tile | bank | byte ]
```

and mapped to the physical address `[ row | base_gid | vc_group | tile | bank |
byte ]`, where `base_gid = gid & ~(4^k - 1)` (`MortonRule.permute`). The top
level has no window: its permuter is the identity.

The checks, per geometry and level:

- the permuter equals the software arithmetic of `vc_global_addr`
  (`MortonRule.alias`);
- every address stays inside the requester's rectangle;
- no two offsets map to the same address;
- every rectangle's group ids are contiguous and form a square.

## Geometries

| Geometry | Config | Groups | Tiles/group | Banks/tile | Words/bank |
|---|---|---|---|---|---|
| `terapool` | `config/terapool.mk` | 4x4 | 16 | 16 | 256 |
| `dse_s1_winner` | `config/dse_s1_winner.mk` | 4x4 | 4 | 64 | 1024 |
| `monsterpool8x8` | former `dse_s1_8x8_winner` | 8x8 | 4 | 64 | 1024 |

`monster_baseline` is checked but not emitted: the testbench does not replay
it.

## Vector format

`#` lines are comments, including the geometry. Every other line is one vector,
whitespace separated, all hex without `0x`:

```
k  gx_i  gy_i  gid  chunk_base  chunk_rows  offset  phys
```

`gid` is the flat Morton id of the requester (`gx_i`, `gy_i` are its
coordinates). `chunk_base` is always 0 and `chunk_rows` always words per bank:
a level's window is the whole L1 of its rectangle. `offset` is relative to the
level's window and `phys` to the start of L1.

### Replaying from SystemVerilog

`tb_vc_scrambler.sv` drives `vc_window_base(k) + offset` with `group_id = gid`
and expects `TCDMBaseAddr + phys`:

```systemverilog
if ($sscanf(line, "%h %h %h %h %h %h %h %h",
            k, gxi, gyi, gid, cbase, crows, off, phys) != 8) continue;
address_i  = i_dut.vc_window_base(k) + off;
group_id_i = gid;
// expect address_o == TCDMBase + phys
```
