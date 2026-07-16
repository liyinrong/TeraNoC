# Dense-LLM FFN benchmarks (`llm_ffn_prefill_f16`, `llm_ffn_decode_f16`)

Baremetal benchmarks for the **SwiGLU feed-forward block** of a dense
decoder-only LLM (Llama-3.1-8B recipe), targeting the TeraNoC RedMulE + Spatz
cluster. They are the first of the workloads specified in `dense_workload.md`
(the FFN block dominates both prefill FLOPs and decode weight traffic).

## What is computed

Per the Llama/Qwen recipe, one FFN block over `S` tokens of width `D` with
intermediate width `I`:

```
h   = x / sqrt(mean(x^2, axis=-1) + eps) * gamma   # RMSNorm        (Spatz RVV)
g   = h @ Wg                                        # gate_proj      (RedMulE)
u   = h @ Wu                                        # up_proj        (RedMulE)
a   = SiLU(g) * u,  SiLU(z) = z / (1 + exp(-z))     # SwiGLU gate    (Spatz RVV)
d   = a @ Wd                                        # down_proj      (RedMulE)
out = x + d                                         # residual       (Spatz RVV)
```

* **`llm_ffn_prefill_f16`** — large `S` (compute-bound, RedMulE array
  utilization is the metric).
* **`llm_ffn_decode_f16`** — small `S` (weight-bandwidth bound, AI ≈ 1; stresses
  the double-buffered weight streaming through TeraNoC).

The golden reference is computed in float32 by the data generator and checked at
the end with `mempool_check_dpi_f16` (tol 0.05).

## Engine mapping

| Stage | Engine | Notes |
|---|---|---|
| RMSNorm, SiLU·gate, residual | **Spatz RVV** (`mempool_llm_ffn_f16.h`) | native f16, f32-promoted scalars; row-/element-striped across all cores |
| gate / up / down GEMMs | **RedMulE** (`mempool_llm_ffn_pipeline_f16.h`) | one tile-local engine per RedMulE tile |

GEMMs are **row-split** across `min(NUM_REDMULE_TILES, S/PORT_WIDTH)` engines, so
the work scales automatically with the number of RedMulE tiles in the active
config. Weights live in **L2** and are streamed into L1 one panel at a time with
**double buffering** (the next panel's DMA overlaps the current panel's compute);
activations stay resident in L1.

## Configurability

* **Workload shape** is set in `software/data/gendata_params.hjson`
  (`S`, `D`, `I`, `PANEL_I`). All dimensions must be multiples of the RedMulE
  `PORT_WIDTH = REDMULE_H*(REDMULE_P+1)` (64 covers both tensorpool64 and
  tensorpool256). The default preset scales the Llama-3.1-8B `I/D ≈ 3.5` ratio
  down to fit the 16 MiB L2.
* **Arch config** is read from compile-time `-D`s (`NUM_REDMULE_TILES`,
  `REDMULE_H/P`, `NUM_CORES`, `RVV`), so the same source builds across configs
  with more/fewer RedMulE engines and any Spatz core count.
* **Precision**: FP16 today. The GEMM format is a single switch
  (`LLM_GEMM_FMT`, default `Float16`); an FP8 path drops in once the FP8
  golden-data generator is added.

## Build & run

These need a **Spatz + RedMulE** config and the **LLVM** toolchain, e.g.:

```bash
# build (reference config)
cd software/apps/baremetal
make llm_ffn_prefill_f16 config=tensorpool256_spatz4 COMPILER=llvm
make llm_ffn_decode_f16  config=tensorpool256_spatz4 COMPILER=llvm

# multi-shape sweep
python auto_build.py --plan build_plan_llm.json

# simulate (from hardware/)
cd ../../../hardware
make simc config=tensorpool256_spatz4 app=apps/baremetal/llm_ffn_prefill_f16
```

The smallest config that exercises both engines is `minpool_spatz4_redmule`
(4 cores, 4 RedMulE), useful for quick functional checks.

## Next workloads (per `dense_workload.md`)

Attention prefill/decode (QKV/O projections, RoPE, GQA `QK^T`/softmax/`scores·V`,
FlashAttention tiling) reuse this pipeline's RedMulE dispatch + double-buffer
scaffolding and the Spatz RVV kernels.
