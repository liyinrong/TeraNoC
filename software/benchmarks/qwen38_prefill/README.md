# Qwen3.8 full-layer prefill development

The accepted plan is `teranoc/docs/plan/qwen38-full-layer-prefill.md` §§1–9
(shared project documentation, alongside the TeraNoC worktrees).

This directory contains complete FP16 attention and GDN prefill correctness
drivers, independent references, and a simulator harness. The guest currently
uses both reduced and real model geometry with bounded weight panels, bounded mixer
activation windows, bounded FFN intermediates and an L2-resident residual
stream: with a window bound set, no L1 buffer scales with the sequence. Real
geometry has passed S=64 and S=128 attention checks; S=2048/4096 on both
meshes, fully independent stage workers, and complete performance DSE remain
unfinished. The Spatz probe is a qualification component, not a full-layer result.

## Numerical reference

`reference.py` evaluates both complete layers, including both residuals and
SwiGLU FFN, with explicit FP16 tensor interfaces. Linear operations accumulate
in FP32; normalization and GDN state use FP32; the independent attention core
uses FP64 to isolate its algorithm from accelerator rounding. These are
synthetic, deterministic correctness inputs, not pretrained-model evaluation.
Effective RMSNorm scale vectors are stored directly (a checkpoint adapter must
convert the decoder's zero-centered weights using `1 + weight`).

Architecture source: the model revision pinned in the plan §2.
Equation cross-check: Hugging Face `modeling_qwen3_5.py`, classes
`Qwen3_5Attention`, `Qwen3_5RMSNorm`, `Qwen3_5RMSNormGated`, `Qwen3_5MLP`, and
functions `torch_recurrent_gated_delta_rule`, `l2norm`, `causal_conv1d_update`.
The NumPy implementation is independent of guest kernels. It is not a bit-exact
emulation of RedMulE accumulation or a claim of parity with every fused backend.
The separate `linear_f16_fma` companion models ascending-K FP16 FMA feedback
with X shift zero, as in `gvsoc-dev/pulp/pulp/light_redmule/light_redmule_v3.cpp:2148`.
Guest fixtures use this arithmetic companion and retain the independent
FP32-projection outputs/states and their differences in separate files. No
tolerance was widened to accommodate the original normalized-K discrepancy.

From the TeraNoC-dev root:

```sh
conda run -n gvsoc env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s software/benchmarks/qwen38_prefill -v
conda run -n teranoc env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python software/benchmarks/qwen38_prefill/fixtures.py /absolute/new/fixture \
  --kind attention --tokens 17
```

Use `--kind gdn` for GDN. The default geometry is small but retains six-way
GQA, three-way GDN head sharing, expanded attention width and partial RoPE.
`--geometry real` uses full model dimensions and requires substantial host
memory and compute. Generated packs contain little-endian raw tensors and a
JSON manifest with dtype, shape, seed and SHA-256. Existing packs are never
overwritten. A manifest is written only after all tensors are complete.
The opt-in 1 GiB memory backing has passed numerical checks at real geometry.

## Complete guest layers

```sh
conda run -n teranoc env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python software/benchmarks/qwen38_prefill/run_layers.py \
  --output /absolute/new/layer-results \
  --model-install /absolute/path/to/matching/install \
  --profiles dse_s1_winner --tokens 65 --kv-placements global replicated
```

Both layer kinds and both schedules run by default. `--kinds attention` or
`gdn` restricts the layer; `--schedules serialized` or `async` selects one.
The resident fixture generator accepts S=1..768 with D=256, I=896 and the
documented toy head geometry. These are complete layers at reduced dimensions,
not real Qwen3.8-27B performance runs. Weight panels are packed offline into
64-column panels. Physical token rows are padded to 64.
`--attention-kv-heads 2` adds a second KV head and six more query heads to
exercise GQA head mapping; it is a correctness variant, not model geometry.

By default each projection stages only two 64-column weight panels per buffer.
`--weight-panels N` sets that bound; `N=0` retains the whole-matrix control.
This driver keeps K intact in each GEMM. RedMulE GEMM can preload the prior
FP16 output as Y, so ascending-K chunks are a possible alternative; their
numerical equivalence and added output traffic still need direct qualification.
The async bounded schedule overlaps the next column-panel group's DMA with
current computation, then
joins all consumers before reusing a buffer. A partial final group is supported.
The FFN reuses gate/up intermediates in 64-token blocks (`--ffn-rows`).
Smaller blocks can lower engine occupancy and reload weights more often; these
are correctness settings, not a tuned production mapping.

`--adaptive-panels` uses the same two L1 weight buffers but fits more
independent output panels in each when the projection's reduction dimension
is shorter than the largest one. `--projection-rows` splits the output rows;
`--projection-columns` sets the packed output panel width. The historical
defaults are 64×64. Use 8×32 for the physical S1 tile and 16×64 for S2;
fixtures and the guest must select the same panel width. Each job
still performs its complete K reduction, so these controls leave FP16 feedback
order unchanged.

`--gateup-pipeline` is an opt-in intermediate schedule for async FFN execution.
It pairs gate/up panels, lets tile agents compute the current group while
sibling Spatz cores activate the preceding group, and prefetches the next
weights through iDMA. The current collective wait and `sync_all()` still join
each panel group; this option qualifies overlap inside a stage, not independent
workers across all layer stages. The coordinator-only DMA queue and directed
three-engine probe under this directory qualify the no-collective runtime path
needed for that later integration.
Both settings are opt-in and the stack gate remains mandatory.

`--gdn-double-chunk --gdn-chunk-rows 16 --gdn-recur-projection-overlap`
starts the next GDN chunk's first full-K QKV output-column group before the
current chunk updates its ordered FP32 recurrent state. The spare chunk slot
holds the next normalized input and the QKV result until the engine completes;
its raw input is retained for residual addition. The current chunk finishes
before the next chunk prepares convolution, so the four-row convolution tail
and every recurrent-state writer keep token order. Other QKV groups, Z and
control projections, gated output projection, and FFN still have broad joins.
This is a qualified partial overlap, not the independent-worker schedule in
`docs/plan/qwen38-full-prefill-pipeline.md` §5. It requires async staged
weights and remains opt-in while full-length throughput is unmeasured.
Adding `--gdn-triple-pipeline --coordinator-dma` reserves a third GDN slot:
group iDMA coordinators stage raw input for chunk n+2 while RedMulE owns the
first QKV group of n+1 and Spatz advances recurrence n. The next iteration
waits for that generation before normalizing n+2; the slot is never reused
while its DMA, RedMulE, or recurrent consumer remains live. The remaining
projection panels still join globally. Complete S=128 S2 4×4 and S1 8×8
triple-slot toy layers pass exact sampled hashes, and all four S=2048
capacity builds fit; see `docs/plan/qwen38-full-prefill-pipeline.md` §7.
On S1 8×8, `--gdn-chunk-rows 64 --projection-rows 8 --weight-panels 8`
exposes 64 first-group QKV jobs for 256 installed engines; it passes a complete
S=128 toy layer
and links at S=2048, but longer recurrence chunks may raise latency.
Externally backed runs fail if either the resolved GVSoC hardware contract or
the guest compile geometry is missing, as well as when they disagree.

The same-source `dse_s2_winner`, S=64, reduced-geometry grid in
`results/qwen38-concurrency-grid-20260923-034759/{r64_a0,r64_a1,r16_a0,r16_a1}/summary.json`
passes all eight complete-layer runs. For each layer, all four sampled tensor
hashes match exactly and each resolved model/ELF contract agrees:

| Projection rows | Adaptive panels | Attention cycles | GDN cycles |
| ---: | :---: | ---: | ---: |
| 64 | no | 392,296 | 768,155 |
| 64 | yes | 263,769 | 651,433 |
| 16 | no | 254,837 | 639,800 |
| 16 | yes | 213,452 | 605,507 |

These are relative mapping comparisons at reduced geometry. They do not
establish absolute full-model utilization or an S=2048/4096 winner.

### Bounded mixer activation windows

`--mixer-rows N` bounds every mixer activation to an `N`-token row window;
`N=0` keeps the whole padded sequence resident and is the control. Several
values may be given, and the first is the bitwise reference the others must
reproduce exactly — shrinking the window must not change one output, state or
cache element. Windows are multiples of 64 so that they hold whole RedMulE row
tiles and whole 64-key attention blocks.

Attention runs two window passes. The first projects, normalizes, rotates and
persists one token window of keys and values into the blocked L2 cache; only
that window's head-major rearrangement is resident. The second owns one query
window at a time: its `q`/gate projection, per-row online-softmax statistics
and FP32 unnormalized output stay with the owning engines for the whole causal
key sweep, so complete query-row and head ownership is preserved and no
reduction or rescaling crosses owners. A window reads only the key blocks its
own rows can see, so the reported `kv_requested_read_bytes` rises with the
number of windows: the same L2 block is re-read by each later window. That is
the measured cost of the bound, not a bandwidth result.

GDN runs one window pass. Projections, convolution, L2-normalized Q/K, decay,
beta, per-head context and the gating buffer are all window sized. The
four-tap causal convolution carries the previous window's last projected rows,
and the `[value head, state column]` owner advances the persistent FP32
recurrent state across windows in token order, so the recurrence is unchanged.

Mixer, key/value and FFN buffers share one arena: their stage lifetimes are
disjoint, so the peak is the largest stage rather than their sum. The
resolved bound is printed as `[qwen38_bounds]` and the runner requires that
line; a run that cannot state which window it used is not qualified. The same
line reports `context_export_cycles`, the part of the measured interval spent
writing the diagnostic context tensor out of the window buffer.

### Residual stream

The residual stream is L2 resident: `input_init` holds the layer input,
`result_mixer` the post-mixer residual and `result_output` the layer output.
Each mixer window stages its own input rows, adds them to its output
projection and writes the window back; each FFN block stages its residual
rows, adds the down projection and writes the output block. Only a
window-sized staging pair stays in L1. Attention checks `result_mixer`
against a golden tensor; GDN only uses the buffer. Unlike the diagnostic
context export, this traffic is part of the layer and is inside the measured
interval.

Each run records `l1_bytes`, the interleaved L1 the guest reserves. On
`dse_s1_winner` at the toy geometry the resident control grows with the
sequence (2 806 016 B at S=65, 4 034 816 B at S=257, 6 934 784 B at S=705)
while `--mixer-rows 64` stays at 2 396 416 B for all three. That is a bounded
working set at reduced geometry, not evidence that full model geometry fits.

### Per-core stack

Each core has `stack_size` (512) bytes at the head of the sequential L1 region
(`config/config.mk:48`, `software/runtime/arch.ld.c:23`) and it cannot grow
without moving the hardware `seq_mem_size`. Overflow is silent: the guest
keeps storing below its own region and the first symptom is an invalid access
inside `printf`. `stackdepth.py` bounds the worst-case depth over the ELF's
direct call graph and the runner refuses any binary that does not fit, before
simulating it. Stage functions are therefore `noinline` so that `main` does
not carry their frames, and fatal DMA reporting writes the console directly
instead of nesting a 208-byte `printf` chain under the projection path.

The whole-matrix control prefetches the next projection's weights and uses two
RedMulE output slots, copying a completed slot with Spatz while the next job
is in flight. RedMulE completion uses the event-unit wait. Stages still join at
explicit barriers; the attention QK/softmax/PV chain and GDN recurrence are not
yet overlapped across independent token chunks. `--delay-seed 7` injects bounded
delays at issue/reuse points. S=257 exercises more than two projection jobs on
some S1 engines, so the output slots are reused.

Each run checks four complete tensors. Attention checks final output, attention
context, packed K/V and the mixer residual; GDN checks final output, context,
convolution history and FP32 recurrent state. Check registrations are fenced.
Paired schedules request verbose host-side DPI output, save actual tensor bits
to raw files, and require bit-for-bit equality, including persistent states.
Checks happen after the measured interval. Missing descriptors or incomplete
element dumps fail the run even if EOC is zero.

### External tensor backing

`--external-data` writes `layer_data.elf` with data-only PT_LOAD segments and
small pointer declarations in the executable header. Inputs, weights and
references bypass C compilation. Result buffers remain executable BSS. This
reduces compiler/executable size; the total tensor storage is still required.
Every tensor has a fixed aligned address and the manifest records its extent
and sidecar SHA256. Run host tests in conda `gvsoc` for pyelftools.

This path uses the opt-in `qwen38_model` target. Build its exact target token
before running, using `GVSOC_EXTRA_MODULES` pointing to this directory and
`USE_GVRUN=1`. For example:
`qwen38_model:system_config=dse_s2_winner:virtual_cluster=true:qwen_l2_mib=512`.
Pass the sidecar through `--parameter=qwen_data_elf=/absolute/data.elf`, never
as part of the target token. The loader explicitly retains entry 0x80000000
and loads the executable before the sidecar. Overlapping ELF segments and
out-of-range sidecar segments are rejected before simulation.

Every `(system_config, virtual_cluster, qwen_l2_mib)` combination is a
*separate* prebuilt platform-tree library, and a missing one fails at launch
with `Missing platform_tree library for target ...` rather than falling back.
The install used in this milestone has `dse_s1_winner_8x8` and
`dse_s2_winner` at 1 GiB and `dse_s2_winner` at 512 MiB; the 4x4
`dse_s1_winner` at 1 GiB is **not** built. Check
`<model-install>/lib/libplatform_tree_qwen38_model_*.so` before choosing a
profile for a sidecar run.

`--l2-mib 1024` requires external data and a matching target build.
It increases backing capacity with the same L2 bank count and port widths,
and relocates ROM to 0xD0000000 in both model and guest build, preserving
the UART at 0xC0000000. Non-power-of-two L2 apertures are rejected because
the address scrambler permutes whole address-bit fields. This is an
explicit simulation capacity variant, not the canonical winner memory map.
The reduced test deliberately places tensors above the old 512-MiB boundary.
Full model activation streaming is implemented; S2048/4096 simulation and
numerical qualification remain outstanding.

### KV placement

Attention persists a blocked L2 cache: for every key block, all transposed K
heads precede all row-major V heads. The globally shared control reads each
block once into shared L1. Replicated mode reads the same L2 block independently
into each VC with ordinary DMA; it does not use multicast. One-group VCs are
the default; `--kv-level 1` selects 2x2-group VCs. All placement runs enable
virtual clusters in both ELF and model, including the global control. Build
the corresponding `teranoc_v2:system_config=...:virtual_cluster=true` platform
libraries before running. 8x8 defaults to replicated mode; 4x4 defaults to global.

When both placements are requested, their actual tensors must also match
bit-for-bit. Reported `kv_requested_read_bytes` counts requested KV payload,
not every L2 bus transaction; `kv_load_join_cycles` includes the load and join.
These diagnostics do not establish full-size L2 bandwidth sufficiency.
The qualification path fills every configured VC, including VCs without an
active query task in a small fixture; requested bytes include those copies.

The runner freezes config, guest/runtime sources and headers before the first
build. Results must live outside the repository. The original toolchain is
reused; the source snapshot and SHA-256 manifest are saved with the results.

## GVSoC numerical qualification

```sh
conda run -n teranoc env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python software/benchmarks/qwen38_prefill/run_softmax_probe.py \
  --output /absolute/new/results \
  --model-root /absolute/path/to/gvsoc-dev \
  --model-install /absolute/path/to/matching/install
```

The runner builds isolated ELFs/runtime objects for the four winner flavors,
then invokes GVSoC under `conda gvsoc`. Every core processes a distinct row.
Fixtures include equal scores, negative scores, late maxima and partial blocks.
Each key block occupies a separate tile with a 32-element-aligned physical
stride, finite negative score padding and zero probability padding. Check input
round trips as well as outputs, including the padding. Contiguous odd-sized
rows/blocks are not a qualified layout: initial experiments failed numerically.
Use `--columns 521 --block 131` to exercise multiple vector chunks inside a
key block at VLEN512. Numerical acceptance requires the exact expected DPI
element count, no errors, EOC0 and a successful simulator exit. Missing checks
are failures. Numerical tolerances are 0.001 absolute for FP16 probabilities
and 0.003 for FP32 statistics, fixed before the first model run.

The probe materializes probabilities solely for inspection. Full attention
will instead rescale its FP32 output accumulator and consume each probability
block with PV. The probe has no RedMulE work or asynchronous pipeline and its
cycles are not a layer throughput estimate. Result and reference buffers remain
alive until EOC; three DPI descriptors cover input, probabilities and statistics.

### Cross-revision reproducibility

Within one source revision every comparison the runner makes is bitwise:
bounds, schedules, KV placements and profiles all agree. Across revisions,
`-ffast-math` is the hazard. The bounded-window refactor moved the GDN S=257
result by one FP16 ULP in 4 of 61 440 context and 50 of 81 920 output
elements, and the cause was a re-association of the query scale: the compiler
had been folding the constant `1/sqrt(key_dim)` in ahead of the last Newton
step of `llm_rsqrt`. That multiply is now spelled out in inline assembly so it
cannot drift, and the source-faithful values supersede the archived ones. The
attribution runs and the ruled-out causes are in
`teranoc/results/qwen38-prefill-20260921/windows/README.md`. When a
cross-revision tensor comparison moves, look for a scalar FP chain the
compiler is free to re-associate before suspecting the algorithm. Only that
one site is pinned; every other scalar chain here can still be re-associated.
Removing `-ffast-math` from this translation unit would settle it globally but
would change every scalar FP result and invalidate all archived hashes, so it
is a decision to take deliberately.

### Bounded host-side checking

`--check-samples N` switches the guest from whole-tensor DPI checks to a
bounded set of `N` elements per tensor, taken as anchored contiguous runs of
`--check-run-length` (default 32). It exists because whole-tensor checking
cannot survive to full geometry for two independent reasons: the golden
tensors are 72-144 MiB of L2 the weights need, and producing them costs more
than anyone will wait for. One FFN down projection under the accelerator's
FP16 FMA arithmetic is 17 408 sequential passes over a tokens-by-5120 array --
**21 minutes measured**, for one projection of one fixture. The same
arithmetic at 2048 sampled points takes **0.5 s**
(`reference.linear_f16_fma_at`, bitwise identical to `linear_f16_fma` at those
coordinates and tested as such).

The runs are placed, not taken from the front. `sampling.py` gives every axis
its ends plus any boundary the fixture names -- the real/padded token split, a
head start, a key-block edge -- and takes both the head and the tail of every
chosen row; a seeded draw fills the rest. A run never crosses a row of the
last axis, so it is one contiguous copy. The chosen starts are saved beside
the fixture as `sample_runs_<tensor>.npy` and the manifest records per-tensor
coverage facts.

Two gathers, for two different reasons:

- The three tensors the layer genuinely produces -- output, the mixer
  residual, the KV cache, or the GDN state -- stay in L2 and are gathered
  after the measured interval closes, on one core.
- The mixer **context** is diagnostic: the layer consumes it in L1 and throws
  it away. Writing the whole tensor out so it could be checked put real DMA
  inside the measured interval and cost 24 MiB of L2 at full geometry. In
  sampled mode each window instead copies only the runs falling in its own
  rows straight out of the L1 window buffer, on one core rather than through
  collective DMA, whose per-transfer barrier costs far more than a few hundred
  bytes of payload. `result_context` does not exist.

  This copy is **scalar**, like the other three gathers, and that is not an
  oversight. It was a Spatz `vle16`/`vse16` copy until 2026-09-22, which was
  22x cheaper and **wrong**: it returned values from elsewhere in the window
  for about 18% of the elements. A padded sequence hid it -- at S=65, 63 of
  128 rows are zero padding, so a corrupt read landing there still returned
  the correct zero, and the path was archived as passing. S=128, the first
  fixture with no padding, failed immediately. Do not reintroduce the vector
  copy without understanding why it corrupts; it is the only copy in this
  driver that crosses from L1 into `.l2_bss`.

  Cost, measured on toy geometry at S=128: `context_export_cycles` is
  52 820-53 080, about 5.7% of a 920 365-cycle run. It is reported separately
  so it can be subtracted. Recovering the old 2 369 cycles means either
  diagnosing the vector copy or moving the gather out of the measured interval,
  which needs the window buffer to outlive its window.

The gather is verified rather than assumed: for the S=65 attention fixture
every gathered element is bit-identical to the corresponding element of an
independently produced whole-tensor dump, across output, context, the
block-major KV cache and the mixer residual. That covers the per-window L1
address arithmetic as well as the L2 gathers. Corrupting one sampled
reference value yields exactly one error in exactly that descriptor.

**Verify at an unpadded length too.** That S=65 check passed with a context
gather that was broken, because 63 of its 128 rows are zero padding and a
corrupt read landing there returns the correct zero. Any fixture whose
`tokens` is a multiple of 64 exercises the gather with nothing to hide behind;
S=128 is the cheap one. The separation that found it is one flag: running the
same fixture with `--check-samples 0` checks the whole tensors instead, so
"computed wrong" and "copied wrong" fall apart in a single 61-second run.

Sampled and whole-tensor runs are not comparable by tensor hash -- they dump
different element sets -- so each result records its `check_mode`,
`check_run_length` and per-tensor coverage.

**Bounded checking is weaker, and by a calculable amount.** The anchored runs
catch structural defects -- a wrong tile, a dropped window, an unzeroed pad --
because they sit where those show up. Against a defect touching a fraction
`f` of the tensor scattered at random, what counts is the number of
*independent sites*, which is `N / run_length`, not `N`: elements inside one
run are adjacent and fail together. The smallest such defect found 95 times in
100 is about `3 * run_length / N`:

| samples | run length | sites | smallest scattered defect found | host cost |
|---:|---:|---:|---:|---:|
| 512 | 32 | 16 | 19% | 0.1 s |
| 8 192 | 32 | 256 | 1.2% | 2.1 s |
| 32 768 | 32 | 1 024 | 0.3% | ~8 s |
| 131 072 | 32 | 4 096 | 0.07% | ~30 s |

The `-ffast-math` query-scale drift found in this milestone moved 4 elements
in 61 440, or 0.007%, and none of these budgets would reliably have seen it.
Host cost is seconds, so full-geometry runs should take a large budget;
whole-tensor checking stays the default at reduced geometry, where it is
affordable and complete.

### Bounded reference at full geometry

A bounded *check* is only half of it: at reduced geometry
`generate_layer_header` still evaluates the whole layer and indexes the
result, which at full geometry would be the hours it was meant to avoid.
`bounded_reference.py` evaluates the sampled coordinates instead. It is used
automatically whenever a `geometry` is passed (`--geometry real`), and can be
forced at reduced geometry with `bounded=True`.

What has to be computed in full is exactly what every row depends on:

| kind | in full | restricted to the sampled rows |
|---|---|---|
| attention | the K/V cache -- any query row attends to all earlier keys | Q/gate projection, the attention core, gating, output projection, FFN |
| GDN | projection, causal convolution, recurrence -- the state at token t carries 0..t | gated norm, z, output projection, FFN |

The restricted part is the same functions the whole-layer reference uses,
called on a row slice, so the result is bit-identical to evaluating
everything and indexing. `test_bounded_reference.py` asserts exactly that, in
two ways: sampled values equal whole-tensor values bit for bit for both
kinds, and the two generator paths emit a **byte-identical**
`layer_data.h` at reduced geometry.

Cost comes from two places. The number of *distinct token rows* the sample
touches drives the restricted part, and it is small: anchored runs of 32
elements give `samples / 32` runs per tensor and each run sits in one row.
The full part scales with S and is unavoidable. `chunk` (default 256 rows)
splits a projection's token axis so one reduction step stays in cache; it is
bit-identical either way and only moves the cost.

Measured on this host, real Qwen3.8 geometry, 2048 sampled elements per
tensor in runs of 32, `chunk=256`:

| kind | S | generation | peak host RSS | sidecar |
|---|---:|---:|---:|---:|
| attention | 128 | 28 s | 1.51 GiB | 712 MiB |
| GDN | 128 | 33 s | 1.55 GiB | 733 MiB |
| attention | 2048 | 4 min 8 s | 1.55 GiB | 731 MiB |
| GDN | 2048 | 6 min 18 s | 1.59 GiB | 752 MiB |

Both passes are included: the FP16 FMA companion the guest is checked
against, and the independent FP32 oracle whose distance from it the manifest
records as `arithmetic_delta_from_fp32_projection`. At S=2048 that distance is
at most 2.0e-3 on the attention output and 2.0e-3 on the GDN output, 4.7e-2 on
the KV cache and 4.7e-4 on the recurrent state.

`fixture_cache.py` generates each distinct fixture once and hard-links it
into every case that shares it, keyed on every parameter that changes an
emitted byte. `--fixture-cache DIR` on `run_layers.py` turns it on; without
it each case regenerates, which is fine at reduced geometry and not at full.

The sidecar base is no longer a constant. `layer_fixture.backing_base` places
it above the executable and the result BSS and below the aperture end, and
keeps the historical constants (`0x90000000` for 512 MiB, `0xa0010000` for
1 GiB) while they still satisfy both -- so reduced-geometry fixtures keep the
addresses their archives recorded, and full geometry, whose weights alone are
710 MiB, moves down.

## Full model geometry capacity probe

```sh
conda run -n teranoc env PYTHONPATH=software/benchmarks/qwen38_prefill \
  python software/benchmarks/qwen38_prefill/footprint.py \
  --output /absolute/new/probe --kind attention --tokens 2048 \
  --backing-base 0x90000000 --no-goldens
```

`footprint.py` emits a guest header for the real geometry with tensor
addresses but no data, then reports how much L2 the layer needs. Compiling
`layer_resident.c` against that header reports the interleaved L1 a profile
and bound reserve. It answers capacity only: nothing runs and nothing is
checked numerically.

## Engine activity dashboard

### Scheduling probes at real geometry

Append `--schedule-probe` to a normal real-geometry `run_layers.py` command
with `--tokens 2048` or `4096`, external 1 GiB backing, and a full-size cached
fixture. The runner creates separate `start` and `tail` cases by default;
`--probe-regions start` or `--probe-regions tail` selects one. The probe adds
the exact GVSoC target token `:redmule_timing_only=true`; build that
platform-tree library in an isolated model workdir before launching the probe.
The normal numerical target library does not satisfy this token. Each case keeps
the real sequence position, model widths, physical RedMulE tiles, full-K GEMM
descriptors, and modeled DMA traffic for each selected job. Within each
projection it executes the first two and final output-column panel groups;
the two adjacent groups reuse the weight/compute slots, and the final group
exercises drain and column tails. Attention likewise visits the first two and
final causal key blocks for each selected query window. It executes two aligned
FFN blocks, two adjacent attention mixer windows or one GDN mixer window.
One 64-row GDN window contains four 16-row chunks, enough for the three-slot
pipeline to turn over. With 64-row mixer windows and 128-row FFN blocks at
S=2048, the FFN covers rows 0–255 or 1792–2047; attention covers 0–127 or
1920–2047; GDN covers 0–63 or 1984–2047. FFN rows outside the selected mixer
output are seeded with finite deterministic values before timing. The paired
regions cover startup, steady turnover, distant causal addresses and final
drain without repeating every token.

The model's `redmule_timing_only` target parameter skips host FP16 MAC
evaluation while retaining RedMulE timing and memory transactions. The guest
therefore skips DPI checks, and `result.json` reports
`schedule_passed=true`, `numerical_checked=false`, zero checked elements and a
partial-layer scope. A successful probe establishes completion and model/ELF
configuration agreement, not tensor correctness or full-layer tokens/cycle.
Before the selected layer work, a separate capacity burst reuses the idle FFN
input arenas as real-size rows, stages full-K QKV or Q/gate weights, and issues
one independent output rectangle per installed RedMulE. It starts a bounded
iDMA transfer and one Spatz activation task per core while those jobs run.
This burst tests whether the machine can be filled with available M/N output
work and whether the three engines can overlap. It is explicitly separate
from the following mixer and FFN schedule: burst occupancy cannot make an
underfilled layer stage pass its own utilization gate.
`--probe-burst-only` runs just this capacity phase for a quick hardware-profile
check. It reports `integrated_gate_passed=null`; a normal `--schedule-probe`
run is still required to inspect attention/GDN and FFN stage scheduling.
The tight S1 4×4 profile reuses mixer and FFN L1 arenas because the separate
arenas overflow 16 MiB. The burst finishes before the mixer begins, so this
reuse is safe for the capacity test; it does not establish cross-stage FFN
overlap. The build-only results distinguish those two layouts.
The tail run seeds the omitted KV prefix or GDN history/recurrent state with
finite deterministic values before the timer starts. GDN probes update 32 of
128 recurrent K rows per selected token by default; `--probe-recurrent-k`
can change that limit. These partial states keep the dependency and traffic
schedule alive, but data-dependent numerical behavior is not validated. The
runner checks that
the selected mixer, GDN chunk or attention key, and FFN loops completed, and
that RedMulE, Spatz and iDMA all emitted activity. Its
`schedule_activity.burst_gate_passed` requires one observed job per RedMulE,
at least 80% occupancy for 1000 cycles, 90% unstalled time while busy,
iDMA/RedMulE overlap and a scheduled
vector task per core. `integrated_gate_passed` separately requires the mixer
and FFN stages to reach 80% RedMulE occupancy for 1000 cycles, overlap iDMA,
and show sampled Spatz work. A completed probe can fail either gate. Periodic
Spatz samples do not prove exact three-way hardware overlap.
The activity adapter labels its dashboard `layer_complete=false` even when
the probe finishes.
Keep full numerical runs for final correctness and throughput qualification.

### GDN ready queue and wide FFN

`--gdn-ready-queue` (guest `Q38_GDN_READY_QUEUE`) replaces the per-chunk GDN
projection with a supertile pipeline. The QKV, Z and A/B control projections
of the next `--gdn-project-rows` supertile form one list of full-K output
rectangles, grouped `--gdn-queue-panels` column panels at a time; a group
exposes `(rows/PR) * panels` jobs, i.e. 64 on S2 4×4 at 256 rows and four
panels. Tile agents keep up to two jobs queued per RedMulE (the HWPE queue
depth) and service them between recurrent tokens and in the prepare/gate
loops, while every core advances the current supertile's recurrence in token
order. The last engine to retire a group releases its weight slot for the
group two ahead and publishes a landing-buffer spill to a two-generation L2
ring through the coordinator DMA queue. Each 16-row recurrent chunk stages
its rows back with one collective DMA per group. The ring is guest L2 BSS, so
the fixture moves the data sidecar above it (`l2_scratch_bytes`). The queue
drains before the supertile's own 64-job output projection reuses the input
buffer and weight slots. The mode needs shared mixer/FFN arenas: the stages
still run one after the other, and the separate arenas do not fit P=4 on
S2 4×4.

`--ffn-wide` (guest `Q38_FFN_WIDE`) issues each gate/up panel group as one
job list, so gate and up rectangles run at the same time, and stages
`--ffn-down-panels` full-K down panels in the whole weight area. The down DMA
is single-buffered and exposed; in exchange, a 256-row FFN block on S2 4×4
offers 64 down jobs instead of 16. Both modes change only job-to-engine
assignment and buffering: every rectangle keeps its full-K ascending FP16
arithmetic.

`--ffn-down-slices 2` (guest `Q38_FFN_DOWN_SLICES`) runs each wide down
rectangle as two ordered K slices on one engine. GEMM reads its Y addend from
Z, so the second job continues the first job's FP16 accumulation. Each slice's
four panels fill half of the weight area, the next slice streams into the
other half, and the first slice is staged during the last gate/up group. The
FFN intermediate is then stored slice-major. Sibling Spatz cores add the
residual of each finished down group while the engines run the next one. In
both FFN loops the engines start before the coordinators issue the next
prefetch. RTL still has to be checked for this Y accumulation: the RTL
scheduler gates Y loading on an op-select bit, while the model always loads Y
for a GEMM.

`--attn-spread` (guest `Q38_ATTN_SPREAD`) keeps QK and PV on one engine per
head but moves each (head, row)'s masking, online-softmax update and
accumulation onto any core, with barriers between the four phases of a key
block. Without it the 24 head agents run those rows serially.
`--attn-reuse-norm` makes the interleaved query pass reuse the K/V pass's
normalized rows. With `--adaptive-panels` the attention projections stage six
K=5120 or five K=6144 panels per group, for 24 or 20 jobs.

`--redmule-matmul` starts every zero-addend job as MATMUL (Z = X*W) instead
of GEMM (Z = X*W + Y, Y read from Z), so no core zeroes a Z tile first; down
slice 1 keeps GEMM. `--ffn-dataflow` (guest `Q38_FFN_DATAFLOW`) replaces the
cluster barrier between wide gate/up groups with dependency counters. Engines
claim a group's rectangles dynamically, start once that group's weights have
landed (and, for up rectangles, once group g-2's activation freed the up
buffer), and copy a finished Z tile out while their next job runs. The last
engine to finish a group publishes the weights of group g+2, and of group G-2
the first down slice. Sibling cores activate a group as soon as all its tiles
are copied out. `--probe-lead-groups N` makes a probe execute N leading panel
groups, not two, before the final one, so group-level steady state is visible.

`--attn-supertile-rows T` (guest `Q38_ATTN_SUPERTILE`) normalizes T rows once,
runs their K/V, Q/gate and output projections as groups of one full-K
rectangle per RedMulE, and keeps the causal key sweeps per 64-row window. The
Q/gate, gated-context and output buffers grow to T rows in the mixer/FFN
union; at T=256 on S2 4x4 the guest links at 32,526,592 of 33,554,432 L1
bytes.

`--proj-dataflow` (guest `Q38_PROJ_DATAFLOW`) runs those supertile
projections with the FFN's dependency counters instead of a cluster barrier
per panel group. The group-local key sweep options build on supertiles:

- `--attn-local` (`Q38_ATTN_LOCAL`) deals the window's 16-row (head, row
  tile) owners round robin over the groups. Each owner's query, scores,
  probabilities, PV tile, accumulator and softmax state live in its group's
  level-0 VC arena; its QK/PV jobs run on that group's four engines, its
  softmax on the group's eight cores, and a key block only needs group
  barriers. RedMulE addresses the arena through the VC window, cores and
  Spatz through its global alias (`q38_la`).
- `--attn-rm-accum` (`Q38_ATTN_RM_ACCUM`) lets RedMulE accumulate the
  unnormalized context in FP16 (Z = P·V + Z, MATMUL for key block 0), so
  the per-block Spatz accumulate disappears and Spatz only rescales rows
  whose running maximum moved. This changes rounding, not the algorithm.
- `--attn-local-pipe` (`Q38_ATTN_LOCAL_PIPE`) software-pipelines the key
  loop: while Spatz runs softmax(n), RedMulE runs PV(n−1) and QK(n+1).
  Scores and probabilities alternate between two slots, and a four-buffer
  K/V ring keeps V(n−1) and K(n+1) resident while K/V(n+2) streams in.
- `--attn-rm-fold` (`Q38_ATTN_RM_FOLD`, needs the pipeline) bounds the FP16
  error: each block's PV starts fresh (MATMUL) into a group tile, and a
  16×16 identity GEMM on RedMulE folds it into the running context
  (Z = I·Zg + Z). Short FMA chains, not the accumulator width, set the
  error, so this matches an FP32 fold's precision without Spatz work. An
  FP32 fold on Spatz was tried and removed: ~10.8k cycles per folded block.
- `--attn-fused-softmax` (`Q38_ATTN_FUSED_SOFTMAX`) does each softmax row in
  one register pass. Its vector arithmetic is that of the unfused row, but
  the reduced layer differs by up to 1 ulp on rows whose maximum moves.

The guest aligns every attention stage field to 256 B and every arena field
to a 1 KB VC row. A misaligned DMA beat aborts the GVSoC DMA interleaver,
and a RedMulE operand row starting 64 B into a 128 B block reads half its
data from the neighbouring block. `[qwen38_key_phase]` lines in probe
builds report per-core cycle totals for each key-block phase and the
per-window prep and finish.

In a scheduling probe with the queue, the mixer covers two supertiles. The
first's recurrence overlaps the second's queued projections, and the guest's
`[qwen38_queue]` line reports that overlap window, queue group and job counts,
and core-0 phase cycles. The runner requires every queued group and job to
retire. `integrated_gate_passed` then judges the mixer on the overlap window
alone, because the serial prologue and output projections fill RedMulE
without any recurrence running beside them.
`spatz_busy_share_sampled_during_redmule_80` prorates the periodic Spatz
samples over the cycles with at least 80% of RedMulEs busy.

Status (2026-09-26): every queue and wide-FFN variant is bit-identical to the
unqueued control on the reduced S=120 layer
(`results/qwen38-gdn-ready-queue-toy-20260926/comparison.json`). At S=2048 on
S2 4×4, the wide FFN completes a scheduling probe; the GDN queue does not.
GVSoC aborts on an L1 bank-interconnect wide-retry assertion inside the
second 64-job queue group, on the numerical and timing-only models alike
(`results/qwen38-gdn-queue-model-abort-20260926.json`). The diagnostic
switches `--gdn-queue-depth`, `--gdn-no-group-sync`, `--gdn-spill-serial`,
`--gdn-retire-copy`, `--gdn-drain-sleep`, `--gdn-no-weight-prefetch`,
`--gdn-z-fixed`, `--probe-burst-rounds` and `--probe-burst-slot` isolate
parts of that pattern; none avoids the abort, while repeated burst-only
rounds do not trigger it. Defaults are depth 1, group-synchronous starts,
prefetch on and a single vector retire copy.


For an instrumented GVSoC run, set `RM_STALL_REPORT=1`,
`SPATZ_ACTIVITY_REPORT=1`, and `IDMA_ACTIVITY_REPORT=1` before the model is
constructed. `SPATZ_ACTIVITY_PERIOD` selects the vector sampling interval.
After the run, pass its `model.log` and `result.json` to
`engine_activity.py model.log --result result.json --telemetry engines.jsonl --json
engines.json`, then run the vendored TeraNoC
`sim_dashboard/generate.py --telemetry engines.jsonl --mesh 4x4 --complete
--out engines.html`. The iDMA records count backend slice payloads after their
acknowledgments and mark intervals with at least one slice outstanding. They
do not claim bank or NoC payload handshakes. A report without a completed
`[qwen38_layer]` timer is explicitly a prefix, not whole-layer utilization.

The vendored dashboard's full-run overview plots FPU-lane, MSHR and NoC
counter families, which this GVSoC engine capture does not emit. Use
`engine_timeline.py engines.jsonl --out engine-timeline.html` for a separate
offline RedMulE/Spatz/iDMA occupancy chart. RedMulE and iDMA intervals are
exact at their reported model boundaries; Spatz busy cycles are distributed
proportionally within each periodic sample and show sample coverage. The page
does not fill missing RTL counter families with zeros.
Older guests printed only the layer duration, not its absolute timer bounds;
their engine rows are tagged `phase: unknown`. In the vendored detail page,
select **All phases** or **Unclassified** rather than the default Benchmark
filter to see those rows. The engine-specific timeline includes every captured
model cycle and labels the startup span explicitly.

The probe exists because the host reference cannot produce full-size goldens
at a useful cost. `linear_f16_fma` models FP16 FMA feedback and is sequential
in K, so one full FFN down projection is 17408 passes over a tokens-by-5120
array. Full-size runs therefore need bounded host-side checking of sampled
elements against files, as the plan's §7 anticipates, rather than whole
golden tensors resident in L2. `--no-goldens` reports the L2 total under that
assumption; the header still declares the symbols so it compiles.

`weight_dims` and `vector_dims` in `reference.py` declare the fixture's tensor
shapes without materializing them, and `test_footprint.py` asserts they agree
with `make_fixture` and that the probe header declares the same guest
interface as the real generator.

## Design exploration

`design_space.py` emits 16 requested profiles: four winner/mesh combinations,
four/eight FPUs per Spatz and VLEN512/1024. Two cores/tile and four integer
units/core remain fixed. Banking factor compensates to retain 64 banks/tile.
`qwen38_model.py` is an opt-in GVSoC target adapter that asserts fixed data
memory/NoC geometry and emits the resolved contract. It must be built against
the current model before using it for measurements. Profiles are unqualified
until numerical checks pass.

Increasing FU count also increases local VLSU port count and instruction-cache
geometry in the current RTL/model. The emitted contract discloses both; do not
describe the resulting design as changing FP arithmetic alone.

On 8x8, start KV replication inside VCs using ordinary DMA reads from L2.
Measure whether L2 can feed replicas before pursuing multicast. Include a
globally shared KV control. Core-count exploration follows lane tuning only.
