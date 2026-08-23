<!--
Copyright 2026 ETH Zurich and University of Bologna.
Licensed under the Apache License, Version 2.0, see LICENSE for details.
SPDX-License-Identifier: Apache-2.0
-->

# WP3 virtual-cluster address permuter — Python reference oracle

Track A of the WP3 virtual-cluster plan: a parameterised Python reference for
the VC address permuter, an exhaustive self-check, and versioned golden
vectors that the RTL branch (Track C) and the GVSoC model (Track D) are both
replayed against.

Reference implementation and tests only. No RTL, no model changes.

## Files

| Path | What it is |
|---|---|
| `vc_permute.py`  | the reference: geometry, level rules, the permuter, an independent arithmetic twin, the ND-strided DMA descriptor, descriptor legality |
| `vc_selfcheck.py`| exhaustive bijection / alias-equivalence / confinement / injected-violation / legality checks |
| `vc_golden.py`   | golden-vector generator and verifier |
| `golden/vc_golden_<geom>_v1.txt`  | legal vectors, replayable line by line |
| `golden/vc_illegal_<geom>_v1.txt` | descriptors the hardware must refuse |

## Running

From the repository root, in the `teranoc` conda environment:

```bash
python3 scripts/vc/vc_permute.py --geometry terapool   # print the layout
python3 scripts/vc/vc_selfcheck.py                     # terapool + 8x8, ~2 min
python3 scripts/vc/vc_selfcheck.py --all               # every fixture
python3 scripts/vc/vc_selfcheck.py --quick             # ~10 s, for iteration
python3 scripts/vc/vc_selfcheck.py -g monsterpool8x8 --byte-exhaustive
python3 scripts/vc/vc_golden.py                        # regenerate vectors
python3 scripts/vc/vc_golden.py --verify               # re-derive and check
python3 scripts/vc/vc_golden.py --self-test            # prove --verify bites
python3 scripts/vc/vc_golden.py --profile general      # keep chunk_base != 0
```

All three exit non-zero on any failure, so they drop straight into CI.
Measured on one core: `--quick` 7 s, the default pair 106 s (terapool 24 s,
8x8 82 s), `--all` 166 s, `-g monsterpool8x8 --byte-exhaustive` 233 s
(540 MB resident).

## What the permuter does

For a VC-relative byte offset `O` at level `k` (an `S x S` square of groups,
`S = 2^k`), issued by the group at mesh coordinates `(gx_i, gy_i)`:

```
byte      = O[ByteOffset-1 : 0]
bank      = O[.. +: clog2(NumBanksPerTile)]
tile      = O[.. +: clog2(NumTilesPerGroup)]
vc_gy     = O[.. +: k]
vc_gx     = O[.. +: k]
chunk_row = O[.. :]                                  // remaining high bits
base_gx   = gx_i & ~(S-1)     base_gy = gy_i & ~(S-1)   // PER AXIS
out.gx    = base_gx | vc_gx   out.gy  = base_gy | vc_gy
out.word  = chunk_base | chunk_row
out       = canonical{ out.word, {out.gx, out.gy}, tile, bank, byte }
```

Specified in `docs/plan/wp2-wp4-implementation-roadmap.md` §T2.1; the physical
field order comes from the same document's §"Canonical layout"; the owner
decisions behind it (fixed fractal pow2-square levels, one rule per level,
computed base, per-level heaps) are in
`docs/plan/wp2-wp4-design-rationale.md` §"VC address model".

All three compositions are **OR, not addition**. The two invariants are
exactly what keeps OR equal to addition:

* **#4** (roadmap §2.2 item 4) the base mask is **per axis**. Masking the flat
  row-major group id instead selects a strip along one axis, not a square.
* **#5** (roadmap §2.2 item 5) `chunk_base` must be pow2-aligned to the chunk
  extent, or its set bits collide with `chunk_row`.

`vc_permute.LevelRule.alias_addr` recomputes the same address with `+`, `*`
and `//` only. It is the alias-equivalence oracle: the two agree **iff** both
invariants hold, which is why the self-check compares them on every probe.

## Geometry fixtures

Nothing in the reference is hardcoded to a flavour; every field width is
derived from `num_x`, `num_y`, `NumTilesPerGroup`, `NumBanksPerTile`,
`TCDMSizePerBank` and `DataWidth`. The flavour numbers appear only in
`vc_permute.GEOMETRIES`, and they were read out of the make configuration
rather than assumed (`NumBanks = num_cores * num_fus_per_core *
banking_factor`, `hardware/src/mempool_pkg.sv:70-71`).

| fixture | config | groups | tiles/group | banks/tile | words/bank | L1 | levels |
|---|---|---|---|---|---|---|---|
| `terapool`         | `config/terapool.mk`          | 4x4 = 16 | 16 | 16 | 256  | 4 MiB  | k=0,1,2 |
| `monsterpool8x8`   | `config/dse_s1_8x8_winner.mk` | 8x8 = 64 |  4 | 64 | 1024 | 64 MiB | k=0..3 |
| `dse_s1_winner`    | `config/dse_s1_winner.mk`     | 4x4 = 16 |  4 | 64 | 1024 | 16 MiB | k=0,1,2 |
| `monster_baseline` | `config/monster_baseline.mk`  | 4x4 = 16 |  4 | 32 | 512  | 4 MiB  | k=0,1,2 |

`terapool` reproduces the roadmap's canonical-layout example bit for bit:
`byte[1:0] bank[5:2] tile[9:6] gy[11:10] gx[13:12] word[21:14]`.

**Group ids are `gx*num_y + gy`, so `gy` is the LOW half of the group field.**
`hardware/src/mempool_cluster_floonoc_wrapper.sv:249` drives
`group_id_i = group_id_t'({group_id.x, group_id.y})` and line 250 numbers the
FlooNoC node `x*NumY + y`. That ordering is what makes the two y-adjacent
groups of a 2x2 rectangle contiguous in the global alias, and it is why a 2x2
VC at base 0 is groups `{0,1,4,5}`.

## The golden vector format

`golden/vc_golden_<geometry>_v1.txt`, format id `teranoc-vc-golden`, version 1.
The full contract is restated in each file's own header — the file is
self-describing on purpose, so a testbench author never has to read this
README.

* `#`-prefixed lines are comments. `# @key value` lines are machine readable
  parameters (geometry, every field position, the sampling constants, the row
  count and a CRC-32 of the data lines).
* Every other line is one vector, whitespace separated, **all hex, no `0x`**:

  ```
  k gx_i gy_i chunk_base chunk_rows offset phys gx gy tile bank word byte
  ```

  `chunk_base` and `chunk_rows` are in **word-planes**, not bytes. `offset`
  and `phys` are byte addresses relative to the VC window and to
  `TCDMBaseAddr` respectively.

* Vectors are **window relative**. Roadmap §T2.2 reserves a fixed address
  window per level but does not fix the constants, so a testbench drives
  `window_base[k] + offset` and expects `TCDMBaseAddr + phys`. Whatever
  window map Track C picks, these vectors stay valid.

Sampling rule (also in the header): every level, crossed with a requester set
(all groups when `num_groups <= 16`, otherwise `{0, 1, S-1, S, N-S, N-1}`
clipped, per axis), crossed with the profile's chunk set, crossed with a
boundary set plus a stride walk `offset_i = (i * 0x9E3779B1) mod aperture`.
The boundary set holds the first and last offset of every level, each field's
low bit, high bits and the carry point below each, and the first and last
offset of chunk planes `{0, 1, 2, rows/2, rows-2, rows-1}`.

**Profiles** (`--profile`, recorded as `@profile` in the header):

| | chunk set | walk | shipped |
|---|---|---|---|
| `adopted` (default) | `(0, whole bank)` only | 80 | yes |
| `general` | plus `(W/4, W/4)` and `(W-W/4, W/4)` | 16 | on demand |

`adopted` matches the design settled in the 2026-08-23 review (finding S12):
a level's window is the full L1 of its rectangle, so `chunk_base` can only be
0 and a vector with a non-zero one is unprogrammable on that hardware.
`general` still emits them because the reference keeps the parameter and the
original acceptance evidence rests on it. `--verify` enforces the profile:
relabelling a `general` file as `adopted` is rejected.

`vc_golden.py --verify` re-derives every vector from the reference, checks the
CRC and the row count, checks alias equivalence and confinement per row, and
audits that those named boundary cases are actually present.
`vc_golden.py --self-test` proves the verifier is not blind: it corrupts a
copy four ways — a dropped vector, a flipped `phys` bit, a missing first/last
boundary offset, a missing highest-legal `chunk_base` — and requires each to
be reported.

`golden/vc_illegal_<geometry>_v1.txt` lists descriptors the programming path
must refuse (non-pow2 extent, misaligned `chunk_base`, over-capacity, level
above the mesh) — the input for the illegal-descriptor SVA of roadmap §T2.3
item 4. Columns there are **decimal**.

### Replaying from SystemVerilog

```systemverilog
int fd = $fopen(path, "r");
string line;
int unsigned k, gxi, gyi, cbase, crows, off, phys;
while ($fgets(line, fd)) begin
  if (line.substr(0, 0) == "#") continue;
  if ($sscanf(line, "%h %h %h %h %h %h %h", k, gxi, gyi, cbase, crows,
              off, phys) != 7) continue;
  // program the level rule {k, cbase, crows}, drive window_base[k] + off
  // from the group at (gxi, gyi), expect TCDMBaseAddr + phys
end
```

## What the self-check proves

Run on `terapool` (38 checks) and `monsterpool8x8` (43 checks); both clean.
`--all` adds `dse_s1_winner` and `monster_baseline` for 142 checks, 0 failed.

* **Bijection** — every rectangle of every level, over its **whole** aperture,
  every offset. The rectangles of one level are swept into a single map, so
  the run also proves they tile the entire L1 exactly once, at every level
  (terapool 4 194 304 bytes; the 8x8 16 777 216 words by default and
  67 108 864 bytes under `--byte-exhaustive`).
* **Alias equivalence** — on every probe, the bit-OR permuter equals the
  arithmetic twin. The ND-strided DMA descriptor is checked separately across
  all levels, chunk configurations and requesters.
* **Confinement** — on every probe, the output group is inside the `S x S`
  bounding box of the requester's rectangle.
* **Requester invariance** — every member of a rectangle sees the same map as
  its corner. Exhaustive where affordable; on the 8x8 exhaustive over every
  `(vc_gx, vc_gy)` pair and sampled elsewhere.
* **Coexisting heaps** — all levels of one disjoint per-level budget swept
  into one map, proving they never share a physical word.
* **Injected violations** — three readings of the invariant-#4 bug and the
  invariant-#5 bug each produce a printed colliding pair; the packed flat-mask
  reading additionally escapes the bounding box. The specified permuter over
  the identical sweep is the control and is clean.
* **Legality** — six illegal descriptor classes are rejected, and the highest
  legal `chunk_base` is accepted.

### Sweep granularity

The permuter is a function of a **byte** offset, so "exhaustive" has to say
which offsets were enumerated. `sweep_level` has one knob: `step = 1` walks
every byte offset and keys the collision map by byte address; `step =
be_width` walks only word-aligned offsets and keys it by word index, visiting
a quarter of the domain. The default is chosen by
`byte_granular = geom.total_bytes <= BYTE_SWEEP_BUDGET` (`1<<23`), because a
byte-granular map for the 8x8 is 67 108 864 entries x 8 B = 537 MB per level.
The report prints which was used.

By default: terapool **byte**-exhaustive, 4 194 304 probes per level; the 8x8
**word**-exhaustive, 16 777 216 probes per level. The gap is closed two ways.

1. Structurally. In `permute_addr` the low `ByteOffset` bits of the offset
   appear in exactly one term, ORed into bit 0 of the result; every other
   field is extracted with a shift of `o_bank_pos = byte_bits` or more, so
   nothing else can see them. Hence
   `addr(O) = (wordmap(O >> 2) << 2) | (O & 3)`, and word-granular injectivity
   implies byte-granular injectivity. `check_byte_lane_identity` tests that
   property over **all** lanes at sampled bases (1.4 M comparisons on the
   8x8) — exhaustive in the lane dimension, sampled in the base dimension.
2. Directly. `--byte-exhaustive` forces the byte sweep regardless of the
   budget. Run on the 8x8: **43 checks, 0 failed, 233 s**, four levels at
   67 108 864 byte probes each (268 435 456 total), every level covering all
   67 108 864 bytes of L1 exactly once, with per-probe alias equivalence and
   confinement. So the byte-level claim on the 8x8 does **not** rest on the
   lemma; the lemma only justifies the cheaper default used in CI.

## Specification findings

Places where §T2.1 as written is ambiguous, imprecise or would mislead an
implementer. Track C implements the RTL branch straight from that section, so
these are worth resolving in the document rather than in the RTL.

* **S1 — `word[31:14]` overstates the word field.** §"Canonical layout" writes
  the terapool layout as `word[31:14]`, 18 bits. Only `TCDMAddrMemWidth` bits
  are the word-in-bank index — 8 for terapool (`TCDMSizePerBank/BeWidth =
  1024/4`), and
  `hardware/src/mempool_tile_remote_req_router.sv:66` slices exactly that
  many. Bits [31:22] are outside the 4 MiB TCDM window. A `chunk_base` CSR
  sized from the document would be 10 bits too wide. The self-check pins the
  real width.
* **S2 — the 8x8 geometry in the task brief is wrong, and the config file is
  untracked.** `config/dse_s1_8x8_winner.mk` gives 64 groups and 4
  tiles/group as expected, but **64 banks/tile, not 32**: `num_cores=512`,
  `num_fus_per_core=4` (`n_ipu=n_fpu=4`, `config/tensorpool256_spatz4.mk`),
  `banking_factor=8` (`config/dse_s1_common.mk`) give `NumBanks=16384` over
  256 tiles. `l1_bank_size` is 4096 B, so 1024 words/bank and a 64 MiB L1 with
  a 26-bit address. 32 banks/tile is `config/monster_baseline.mk`. Separately,
  `config/dse_s1_8x8_winner.mk` and its `dse_s*` includes are **untracked** in
  the main checkout and therefore absent from the `wp3-vc` worktree; the
  numbers above were read from the main checkout by evaluating the makefiles.
* **S3 — `byte = O[1:0]` is hardcoded in the pseudo-code.** Every other field
  in §T2.1 is parameterised; the byte field is not. It should be
  `O[ByteOffset-1:0]` with `ByteOffset = clog2(DataWidth/8)`
  (`hardware/src/mempool_pkg.sv:64-66`). Harmless today (`DataWidth` is 32
  everywhere) but it is the one line an implementer would copy verbatim.
* **S4 — "the base mask is per axis" has two plausible wrong readings, and
  they fail differently.** §2.2 item 4 says the wrong form gives "an S x 1
  column". That matches masking the flat id **and** ORing one packed `2k`-bit
  `vc_group` field at the bottom of the group field — that reading both
  collides and escapes the bounding box. The other reading, keeping the
  per-axis field placement but deriving both bases from the masked flat id,
  **collides but never escapes**: `gx_i | vc_gx` always stays inside
  `[gx_i & ~(S-1), +S)`. A confinement SVA alone (§T2.3 item 3) therefore
  would **not** catch it. §T2.3 item 2's bijection check is the one that does.
  Worth saying explicitly in the gate.
* **S5 — the chunk extent is never named as a parameter.** §T2.1 makes
  `chunk_base` programmable and `{window-range, k}` fixed, but the chunk
  *extent* — the number of high offset bits, i.e. `chunk_row`'s width — is
  what invariant #5 aligns against and it is not in the rule table. It has to
  be either fixed per level at config time or programmable alongside
  `chunk_base`; the reference models it as `LevelRule.chunk_rows`, in
  word-planes. If it is programmable, the alignment assert must be re-checked
  on every reprogram, not only at elaboration.
* **S6 — window size versus chunk size is unresolved.** The rationale says the
  VC region is "reserved at max-VC size" and that a VC "uses up to its
  capacity (over-run aliases/wraps)". If the fixed window is larger than the
  programmed chunk, offsets above the chunk either wrap onto it (an alias, and
  therefore a silent bijection break within the window) or must be rejected.
  The reference takes the strict reading: the offset space **is** the chunk,
  and anything larger is out of range. Track C needs a decision, and if the
  answer is "wrap", the bijection claim has to be restated as "bijective on
  the chunk, wrapping above it".
* **S7 — §T2.1 lists three levels (global, 2x2, 1x1) but the geometry allows
  more.** On the 4x4 mesh the whole-chip level *is* 4x4, and on the 8x8 mesh
  there are four fractal levels. The reference supports every `k` up to
  `min(log2 num_x, log2 num_y)` and the self-check covers all of them. The
  useful invariant that fell out: the top level with `chunk_base = 0` is
  **exactly the identity map**, which is invariant #10's backward-compatibility
  requirement expressed on the permuter — worth an assert in the RTL.
* **S8 — the id encoding and the DMA descriptor are consistent, and the
  rationale's worked example is correct.** §"Cross-VC L1<->L1 DMA addressing"
  gives a 2x2 terapool VC as `{0,1,4,5}` with a 0x800 contiguous run, a 0x1000
  x-group stride and a 0x4000 word-plane stride. The reference reproduces all
  four numbers, and the generalised descriptor agrees with the permuter on
  every probe. No defect — recorded because the example is load-bearing for
  Track B and it is now mechanically checked.
* **S9 — "a 3rd mutually-exclusive branch" is not enough: the VC branch has to
  come FIRST, and §T2.1 does not say so.** The existing chain in
  `hardware/src/mempool_addr_scrambler.sv:75-87` is
  `if (seq swap) ... else if (EnableTileIdRemap && not_io_address && addr >=
  NumTiles*SeqMemSizePerTile) ...`. All three compute sites enable both stages
  — `EnableSeqInterleaveSwap(1'b1)`, `EnableTileIdRemap(TileIdRemap)` at
  `hardware/src/mempool_tile.sv:1117-1118`, `:1430-1431` and `:1748-1749`.
  After the Phase-2 address-map enlargement (roadmap §"Phase-2 drift check
  (2026-08-22)" item 2) the decoded TCDM window covers the VC windows too, so
  `not_io_address` (`mempool_addr_scrambler.sv:49`) becomes **true** for a
  VC-window address, and that address is above the sequential region. A VC
  branch appended as the last `else if` would therefore never fire on any
  config with `tile_id_remap=1`: the tile-id branch swallows it, silently, and
  only on those configs. Place the VC branch immediately after the sequential
  branch, or exclude the VC windows from the tile-id branch's condition.
* **S12 — `chunk_base` is a dead register once windows are max-sized, and
  removing it deletes invariant #5.** Invariant #5 requires
  `chunk_base % chunk_rows == 0` and capacity requires
  `chunk_base + chunk_rows <= words_per_bank`. A max-sized window means
  `chunk_rows == words_per_bank`, so together they admit exactly one value:
  **0**, at every level of both geometries (checked). The offset's
  `chunk_row` field is then already the full bank-row index, so every row is
  reachable from the offset alone.
  The clean restatement: **a level's window is the full L1 of its rectangle** —
  all `S^2` groups, all tiles, all banks, all bank rows. That also re-derives
  S11's top-level redundancy: at `k = log2 N` the rectangle is the chip, so its
  window is the whole L1, which is the global region.
  What this removes: the `chunk_base` register and its per-group broadcast
  (the rule table's only address-path programmable field, `{enable,
  chunk_base}` -> `{enable}`); **invariant #5**, which becomes vacuous;
  the third OR (`out.word = chunk_base | chunk_row` -> `out.word = chunk_row`);
  **S5** (the chunk extent is now the config constant `words_per_bank`);
  **S6** (window == aperture, so no reserved-but-unbacked region exists); and
  most of section T2.3 item 4 — of the nine illegal descriptors only "level
  above the mesh" survives, and that is elaboration-time.
  It also downgrades the worst failure mode. A misaligned `chunk_base` folded a
  *single buffer onto itself*, silent and only findable by exhaustive sweep.
  Without it the map is a bijection over the entire window, so a buffer can
  never fold; the worst a buggy allocator can do is overlap *two* buffers,
  which has ordinary symptoms.
  **What it gives up:** hardware-enforced separation between levels, including
  the stack floor at rows 0..31. That moves to the allocator, consistent with
  the standing owner position for the global region — "Region = reachability,
  allocator = ownership... Allocator-collision safety is a SW convention, NOT a
  hardware concern" (`docs/plan/wp2-wp4-design-rationale.md` section "VC
  address model", "Global region = FULL L1"). No isolation is actually lost:
  any core can already compute a global pointer into another level's rows.
  **What it locks in:** windows must be max-sized, so S11's `1.33x` floor
  becomes structural rather than a choice.
  **Verification consequence:** `check_exhaustive` runs `LevelRule(geom, k)`
  with `chunk_base=0, chunk_rows=words_per_bank` — exactly this design. The
  byte-exhaustive sweeps at every level on both geometries already prove the
  simplified permuter directly; the budgeted-chunk configurations were the
  extra generality, not the target.
* **S11 — the window map is never sized, and the pow2 `TCDMMask` idiom will
  not express it.** The rationale asserts "address space is plentiful"
  (`docs/plan/wp2-wp4-design-rationale.md:191`) without a scaling bound, and
  neither document fixes the window constants, so nothing stops an
  implementation padding every window to the L1 size. The cost of the choices:
  a level's max aperture is `S^2 x (per-group L1) = 4^k x G`, so the windows
  are a geometric series with ratio 4, dominated by the top level.
  - Padding each region to the L1 size costs `(levels+1) x L1` and is the only
    variant that fails to fit: 1536 MiB at 16x16 against the 1 GiB between
    `TCDMBaseAddr` and `PeripheralsBaseAddr` (`mempool_system.sv:42`).
  - Packing at natural alignment costs `2.33 x L1`.
  - Dropping the top-level window costs `1.33 x L1` at every scale. The top
    level is provably redundant: at `k = log2 N` there is one rectangle, the
    base is always `(0,0)`, and the permuter is the identity - the property
    `check_top_level_is_identity` verifies. A top-level allocation is a global
    pointer biased by `chunk_base`, and it loses nothing, because confinement
    and position independence are both vacuous when the rectangle is the whole
    chip and there is one instance. This is why roadmap section T2.1 lists
    three rules (global + 2x2 + 1x1) rather than four.
  `1.33 x L1` is a FLOOR, not a design point: budget-proofness at level `k`
  forces `window_k >= 4^k x G`, the windows must be disjoint, and invariant #10
  forces a full-`L1` global region. Beating it means giving up budget-proofness
  (cap levels below their max), the constant compare (programmable
  base/mask, and with it variable-`k` slicing on the critical path), or
  invariant #10.
  **The implementation trap:** `TCDMMask = ~(TCDMSize-1)`
  (`hardware/src/mempool_pkg.sv:812-813`), consumed by `not_io_address`
  (`hardware/src/mempool_addr_scrambler.sv:49`), only works for a power-of-two
  region. `1.33 x L1` never is, so a single mask rounds the decoded region up
  to `2 x L1` - which at 32x32 means `[0, 0x8000_0000)`, exactly where L2
  lives. The fix is implied by the fixed-window design anyway: replace the one
  mask with a per-window constant range compare and make `not_io_address` the
  OR of the region hits. Drift-check item 2 currently only says the region has
  to grow; it also has to stop being a single pow2 mask.
* **S10 — the module's `tile_id` is the flat `{group, tile}` field, not
  §T2.1's `tile`.** `mempool_addr_scrambler.sv:41` sets
  `TileIdBits = $clog2(NumTiles)` over the **global** tile count, and the
  stage-1 swap at line 76 moves that whole field. For terapool that is 8 bits
  landing at `[13:6]`, i.e. the canonical `tile[9:6]` and `group[13:10]`
  together. §T2.1's `tile` is only `clog2(NumTilesPerGroup)` wide. The VC
  branch needs `NumTilesPerGroup`, `NumX` and `NumY` as new parameters — none
  of them exists on the module today — and must not reuse `TileIdBits`.
