# Welvet

**Welvet** is the AI engine: layers, numerical types, quants / k-quants, and backends (CPU tiled · Plan 9 SIMD · WebGPU).

| | |
|--|--|
| **Version** | **v1.1.0** |
| **Scorecard** | **100 / 100** pts (see [Version scorecard](#version-scorecard)) |

Engine is v1. Apps (`octo`), stubs, and NPU/C++ accel sit **outside** this board — they are sibling / later trees (`welvet.cpp`), not missing Welvet.

| Repo | Role |
|------|------|
| **[openfluke/welvet](https://github.com/openfluke/welvet)** (this tree) | **Engine only** — layers, quant, SIMD (Plan 9 `.s`), WebGPU, ENTITY, dispatch |
| **[openfluke/w2a](https://github.com/openfluke/w2a)** (`w2a/`) | Tests, CABI, docs, menus — **never** in engine packages |
| **[openfluke/octo](apps/octo/)** (`apps/octo/`) | Model shell — HF download, convert→ENTITY, quantize, run (Lucy successor) |

`loom/poly` is legacy reference only. Welvet is the rewrite.

### Architecture: Dense is the shared MatVec microkernel

Most transformer / CNN FLOPs are **weights × activations**. Welvet keeps **one** Dense stack for that (FormatNone×34 + 20 quants × CPU/SIMD/WebGPU). Layers whose expensive bit is GEMV **reuse** it; layer-specific ALU stays local.

| Kind | Examples | What runs where |
|------|----------|-----------------|
| **Native layer math** | Dense, RMSNorm, LayerNorm, Softmax, Embedding | Own fwd/bwd; norms + Softmax (incl. exotic) WebGPU; Softmax/SiLU SIMD; Embedding gather/scatter GPU |
| **Composite → Dense projs** | MHA (Q/K/V/O), SwiGLU (Gate/Up/Down), RNN/LSTM/Residual/Sequential | Projections = Dense children; MHA RoPE/attn fwd+bwd GPU (decoder gate); SwiGLU SiLU⊙ SIMD+GPU; RNN/LSTM fused GPU |
| **im2col → Dense** | CNN1/2/3 | Host im2col+Dense for quants/non-f32; **FormatNone f32** uses tiled WebGPU conv (no im2col) |

**This is intentional**, not a missing abstraction: one MatVec surface means one place for quant bugs, dtype wires, and backend parity. Separate native kernels pay off when the calc is **not** GEMV (fused attention, tiled CNN, Softmax/SiLU SIMD).

**Not “fully native” yet (honest):**
- Optional future: dedicated k/IQ Plan 9 `.s` beyond fused Go/int8 `DotKRow`/`DotIQRow` (peak cell = fused Dot*, already ✅)
- MHA exotic (cross/ALiBi/sigmoid/dropout/sliding) and CNN quant paths still host / im2col
- WebGPU device ALU is typically **f32** at the boundary (storage dtype narrows on upload)
- Suite honesty: `w2a/suites.StampBackendNote` / `AffinePackable` — no silent host counted as “WebGPU/SIMD done”

Remaining work: [`docs/loom_2_welvet_todolist.md`](../docs/loom_2_welvet_todolist.md).

### Tree layout

| Folder | Contains |
|--------|----------|
| (top) | `core`, `weights`, `quant`, `simd`, `webgpu`, `tiling`, `architecture`, `layers/`, `lucy/`, `fusedgpu/` |
| `runtime/` | `forward`, `backward`, `training`, `step` |
| `systems/` | `dna`, `evolution`, `tween`, `tanhi`, `telemetry` |
| `model/` | `transformer`, `entity`, `tokenizer`, `sampling`, `hf` |
| `apps/` | `octo`, `flux2`, `mosstts` (not on the v1 board) |
| `stub/` | future: `accel`, `donate`, `fountain`, `hardware`, `memory`, `seed`, `serialization` (not on the v1 board) |
| `w2a/`, `tools/` | harness (not engine) |


**Status: v1.1.0.** Scorecard still **100/100** (v1.0 board complete). This minor packs
**CamSync** (inter-cameral / cross-layer same-shape weight blend), the full
[`training_modes.md`](training_modes.md) equations sheet, Lucy **Lean** density
(Acc keep ≥95% then smallest/fastest), a DotF64 generic wire fix, and feature-book §70.
NPU/Metal/QNN are not scored here.

| Legend | Meaning | Pts credit |
|--------|---------|------------|
| ✅ | **Implemented** — layer/runtime/stub API works; w2a suite passes (full timed matrix for Dense / transformer / CNN-RNN / §5 extended) | **100%** of row weight |
| 🚧 | **Partial** — works but lighter coverage, inflate-not-fused SIMD, or host ALU on GPU/SIMD path | **50%** of row weight |
| ⬜ | **Not started** — stub `doc.go` only, hard-error everywhere, or peak fused kernel explicitly missing | **0%** |

---

## Version scorecard

**Formula:** `version = 0.{round(earned)}` until 100 → **v1.0**. Patch tags
(v1.0.1–v1.0.3) and minor tags (**v1.1.0**, …) ship engine deltas without moving the
board. Weights sum to **100**. This tag is **v1.1.0** (feature pack on the full v1.0 board).

| # | Section | Wt | How scored today | Earned |
|--:|---------|---:|------------------|-------:|
| 1 | **Foundation** — layout, rules, `core`, `weights`, `quant`, `simd`, `webgpu` base, `tiling` | 15 | all ✅ | **15.0** |
| 2 | **Dense MatVec microkernel** — FormatNone×34 + quants × backends, train/grad; fused Dense SIMD for all quants | 15 | all ✅ | **15.0** |
| 3 | **Transformer stack** — MHA, SwiGLU, RMSNorm, LayerNorm, Softmax, Embedding, Sequential, Residual, seqmix | 14 | all ✅ | **14.0** |
| 4 | **CNN / RNN / LSTM** — full timed 34×20×3 matrices; tiled-conv / recurrence shaders | 6 | all ✅ | **6.0** |
| 5 | **Extended layers** — GDN, ConvT1–3, Mamba, KMeans, Parallel, Metacognition | 7 | all ✅ (full timed matrix + train grids; GDN truncated BPTT) | **7.0** |
| 6 | **Runtime + architecture** — volumetric grid, forward, backward, training, step | 8 | all ✅ | **8.0** |
| 7 | **Systems** — dna, evolution, tween, tanhi, telemetry | 5 | all ✅ | **5.0** |
| 8 | **Model / IO** — tokenizer, entity, transformer, sampling, hf | 8 | all ✅ | **8.0** |
| 9 | **Training credit** — StepBP, Tween*, TweenSplit, HeadProxy, Linear, FastProxy, Sparse, Alt, cameral BranchModes | 8 | all ✅ | **8.0** |
| 10 | **Peak fused / no host ALU** — fused k/IQ Dot*, MHA attn/RoPE GPU fwd+bwd, LN/SwiGLU/Softmax/Embedding/RNN/LSTM/CNN tiled GPU, Softmax SIMD; `fusedgpu` decoder fuse | 14 | all ✅ | **14.0** |
| | **Total → v1.0** | **100** | | **100.0** |

**v1.0 readout:** engine credit + MatVec + volumetric step + GPU fuse are in. **v1.0.1:** `lucy.BuildLPD` + `TrainStackCE`. **v1.0.2:** Step\* systolic pipe + Step\* credit family + `ShortTrainMode`. **v1.0.3:** Dense FormatNone SIMD forward (~4.3× geo-mean; Go 24 / C++ 9 / Rust 1 on the deep suite). **v1.1.0:** **CamSync** / `BlendStores` (α 1%→100%, Groups, Cross same-shape pairs, after sample/step/pulse); `training_modes.md` (all 29 named modes); Lucy **Lean** density (`LPDLeanKeep`, LeanChamp / LeanByArch); DotF64 generic wire fix; feature book §70 CamSync. **Not on this board:** `apps/octo`, `stub/*`, **NPU / Metal / QNN**. FastProxy can match/beat StepBP **Acc** on sine/copy toys; Sparse wins Lucy **Score** via Avail. Do not write “Sparse is better backprop.”

Detail tables below still list per-feature ✅/🚧/⬜; they feed honesty, but **only this scorecard sets the version number**.

---

## Snapshot (honest)

Status rollup — version points live in the [scorecard](#version-scorecard) only.

| Area | Status |
|------|--------|
| Engine layout (one feature → one folder) | ✅ |
| Rules: no engine tests / no fallbacks / no hardcoded float32 / no QAT / train keeps storage truth | ✅ |
| `core` types (34 dtypes, Tensor\[T\], activations, backends) | ✅ |
| `weights` FormatNone × 34 stream pack/MatVec + in-dtype ApplySGD (no retained f32 master) | ✅ |
| `quant` Pack/Unpack/MatVec all 20 formats (CPU) | ✅ |
| `simd` Plan 9 kernels linked (amd64/arm64) | ✅ |
| webgpu | Dense GEMV/GEMVT (incl. Affine resident) + RMS/LN/Softmax exotic + SwiGLU + MHA attn/RoPE fwd+bwd + tiled CNN + Embedding + RNN/LSTM | ✅ |
| **Dense** FormatNone × 34 × CPU/SIMD/WebGPU fwd+bwd | ✅ |
| **Dense** all 20 quants — WebGPU fwd+bwd | ✅ |
| **Dense** k/IQ/Affine SIMD (group Dot* + scales; no F32 inflate) | ✅ |
| `architecture/` volumetric grid (cells, hops, remote links) | ✅ |
| `runtime/forward/` / `backward` / `training` — Dense…Residual + ConvT1–3 + Parallel + KMeans + Mamba + Metacognition + GDN | ✅ |
| Cross-Numeric Train (w2a Step) — Op kinds × weight dtype × act host (smoke ~735 / full ~10.7k) | ✅ |
| ConvT / Parallel / KMeans / Mamba / Metacognition / GDN — full timed matrix + train grids (GDN truncated BPTT) | ✅ |
| Model IO / transformer / entity / tokenizer / hf | ✅ |
| Training credit (StepBP / TweenSplit / FastProxy / Sparse / …) | ✅ |
| Cameral `.entity` (`WriteCameralFile` / `LoadCameral`) | ✅ |
| CamSync — soft/hard cam weight avg + cross-layer same-shape pairs | ✅ |
| `fusedgpu/` decoder fuse (WebGPU + optional Android Vulkan) | ✅ |
| `apps/octo/` interactive model shell (download / convert / chat) | 🚧 off-board |
| `stub/` seed · serialization · hardware · memory · fountain · donate | 🚧 off-board |
| `stub/accel/` (NPU/Metal/QNN — not Go Welvet; later `welvet.cpp`) | ⬜ off-board |
| Full v1 matrix (every cell peak-fused, no host ALU) | 🚧 (peak fused ✅; nested Sequential/Residual still open) |

Validate live:
```bash
cd w2a && go test ./tests/dense -v
cd w2a && go test ./tests/mha -v
cd w2a && go test ./tests/swiglu -v
cd w2a && go test ./tests/rmsnorm -v
cd w2a && go test ./tests/layernorm -v
cd w2a && go test ./tests/cnn1 -v
cd w2a && go test ./tests/cnn2 -v
cd w2a && go test ./tests/cnn3 -v
cd w2a && go test ./tests/rnn -v
cd w2a && go test ./tests/lstm -v
cd w2a && go test ./tests/embedding -v
cd w2a && go test ./tests/softmax -v
cd w2a && go test ./tests/sequential -v
cd w2a && go test ./tests/residual -v
cd w2a && go test ./tests/gdn -v
cd w2a && go test ./tests/mamba -v
cd w2a && go test ./tests/convt1 -v
cd w2a && go test ./tests/convt2 -v
cd w2a && go test ./tests/convt3 -v
cd w2a && go test ./tests/kmeans -v
cd w2a && go test ./tests/parallel -v
cd w2a && go test ./tests/metacognition -v
cd w2a && go test ./tests/dense -run LinearGradIn -v
cd w2a && go test ./tests/entity -run Cameral -v
cd w2a && go test ./tests/parallel -run 'Credit|AllCredit|AllStack|Combine|HemiN|Sparse|ParseTrain|RequiresGrid|WrappedCNN|Inherit|TrainMSE' -v
```

---

## Non-negotiable rules

1. **No testing code in the engine tree** — all checks in `w2a/`.
2. **No fallbacks** — missing path → hard error (no SIMD→Go, no fake GPU).
3. **Nothing hardcoded to float32** — APIs are `Tensor[T]` / generics. Host wires are `WireF32` / `WireF64` / `WireI8` via `weights.SelectWire` (float64 & integers are **not** forced through f32). WebGPU WGSL ALU is f32 on typical adapters — narrowing happens only at the device boundary.
4. **No QAT** — `DType` + `QuantFormat` are storage truth.
5. **Train keeps storage truth** — `weights.ApplySGD` updates FormatNone in native lanes (or unpack→update→re-Pack for packed formats). No retained float32 master beside storage (`RetainsF32Master` only for FormatNone+float32).
6. **One poly feature → one folder.**
7. **v1.0 = scorecard 100/100** (this board). Patch/minor tags (v1.0.1…v1.0.3, **v1.1.0**, …) add measuring/API without moving the board. Apps, stubs, and NPU are not scored.

---

## Numerical type vs k-quant (read this)

Weight storage is **two independent axes**. They are not the same thing, and **neither implies an FP32 master tensor**.

| Axis | Field | What it is | Examples |
|------|--------|------------|----------|
| **Numerical type** | `core.DType` | How each element is encoded when **not** using a packed quant scheme | `float32`, `float16`, `int4`, `binary`, `nf4`, … (34 total) |
| **Quant / k-quant** | `quant.Format` | Block / group packing layout (ggml-style Q*, K-quants, IQ, Ternary/BinaryPacked, …) | `Q4_0`, `Q5_K`, `IQ2_XXS`, `BinaryPacked`, … |

**`FormatNone` (“none” in logs)** means: *no packed quant layout* — weights are stored as the chosen **DType** native bytes.  
It does **not** mean “FP32 master” or “untyped.”

| Label you see | Actual storage |
|---------------|----------------|
| `none/float32` | FormatNone + float32 payload |
| `none/float16` | FormatNone + float16 bytes (not an f32 master) |
| `none/int4` | FormatNone + int4 native pack |
| `none/binary` | FormatNone + `DTypeBinary` (native 1-bit dtype) |
| `Q4_0` / `Q5_K` / … | Packed quant blob (`Format != none`); DType is incidental for pack paths |
| `BinaryPacked` | Packed 1-bit **quant** layout — different from `none/binary` |

**Rules of thumb**

1. Pick **either** a native dtype under FormatNone **or** a packed quant format — that pair *is* storage truth.
2. Convert hubs through a temporary f32 scratch, then **re-encodes and drops it**. Nothing keeps a parallel master.
3. **SGD** uses the same rule: FormatNone → in-dtype `ApplySGD`; packed → short-lived unpack scratch → update → re-Pack. Weight dtype ⊥ activation `Tensor[T]` (Cross-Numeric Train in w2a Step).
4. w2a / Octo lines like `none/float16 → Q4_0` mean: decode current storage → scratch → encode destination (lossy when leaving high precision).

Public Dense demotion + full W×A showcase (charts/PDF): [down-the-dem](https://github.com/openfluke/down-the-dem).

Tables below list the 34 dtypes and 20 formats.

---

## Axes (what “done” means per feature)

For each layer / op, every cell must work:

| Axis | Count | Values |
|------|------:|--------|
| Backend | 3 | CPU tiled (SC+MC) · Plan 9 SIMD · WebGPU |
| DType | 34 | `0…33` — numerical types table below |
| Quant | 20 | `None` (= FormatNone + DType) + classic + k-quant + IQ + Ternary/BinaryPacked |
| Pass | 2 | forward **and** backward (where trainable) |

**No cell may silently substitute another cell.**

---

## DTypes (`core.DType`) — 34 numerical types

Element / native storage types. Used with **`FormatNone`**. Dense coverage today:

| # | DType | CPU tiled | SIMD | WebGPU | Notes |
|--:|-------|:---------:|:----:|:------:|-------|
| 0 | Float64 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device f64→f32 | |
| 1 | Float32 | ✅ | ✅ Master+DotTile | ✅ FP32 WGSL | |
| 2 | Float16 | ✅ | ✅ F16C+DotTile | ✅ native decode | no Wire cache |
| 3 | BFloat16 | ✅ | ✅ packed+DotTile | ✅ native decode | |
| 4 | FP8E4M3 | ✅ native codec | ✅ packed+DotTile | ✅ native decode | real E4M3 |
| 5 | FP8E5M2 | ✅ native codec | ✅ packed+DotTile | ✅ native decode | real E5M2 |
| 6 | Int64 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device | |
| 7 | Int32 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device | |
| 8 | Int16 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device | |
| 9 | Int8 | ✅ | ✅ DotI8 | ✅ on-device I8 | |
| 10 | Uint64 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 11 | Uint32 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 12 | Uint16 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 13 | Uint8 | ✅ | ✅ affine+DotTile | ✅ on-device affine | |
| 14 | Int4 | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 15 | Uint4 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 16 | FP4 | ✅ native E2M1 | ✅ packed+DotTile | ✅ native decode | |
| 17 | Int2 | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 18 | Uint2 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 19 | Ternary | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 20 | Binary | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 21 | Int | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device | Go native width |
| 22 | Uint | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 23 | Uintptr | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 24 | Complex64 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ real-part GEMV | |
| 25 | Complex128 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ real-part GEMV | |
| 26 | NF4 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device table | QLoRA |
| 27 | FP6 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device signed-6 | |
| 28 | Int6 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device signed-6 | |
| 29 | Uint6 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 30 | Int5 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device signed-5 | |
| 31 | Uint5 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 32 | Int3 | ✅ | ✅ DecodeRowF64+DotTileF64 | ✅ on-device signed-3 | |
| 33 | Uint3 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |

**SIMD:** no `GPUWireF32` / `WireF64` full-matrix cache — Master / DecodeRow / packed native → DotTile.  
**WebGPU:** all 34 FormatNone dtypes on-device fwd+GEMVT + DenseDW.  
**✅** = dtype-specific path end-to-end for that backend.

---

## Quant formats (`quant.Format`) — 20

Packed layouts (Q* / K / IQ / …). Row **None** = FormatNone: use a **DType** from the table above — not “FP32 master.”

CPU Pack/Unpack/MatVec/MatVecT vs Dense SIMD / WebGPU:

| Format | CPU pack+MatVec | Dense SIMD | Dense WebGPU |
|--------|:---------------:|:----------:|:------------:|
| None | ✅ (via `weights` + DType) | ✅ FormatNone native/stream | ✅ all 34 fwd+GEMVT |
| Q8_0 | ✅ | ✅ fused DotI8×scale | ✅ on-device Q8 GEMV (in%32) |
| Q4_0 | ✅ | ✅ fused DotQ4_0 fwd | ✅ on-device Q4 GEMV (in%32) |
| Q4_1 | ✅ | ✅ fused DotQ4_1 | ✅ on-device Q4_1 |
| Q5_0 | ✅ | ✅ fused DotQ5 | ✅ on-device Q5 |
| Q5_1 | ✅ | ✅ fused DotQ5_1 | ✅ on-device Q5 |
| Q2_K | ✅ | ✅ fused group DotKRow + scales/mins | ✅ on-device k GEMV |
| Q3_K | ✅ | ✅ fused group DotKRow + scales/mins | ✅ on-device k GEMV |
| Q4_K | ✅ | ✅ fused group DotKRow + scales/mins | ✅ on-device k GEMV |
| Q5_K | ✅ | ✅ fused group DotKRow + scales/mins | ✅ on-device k GEMV |
| Q6_K | ✅ | ✅ fused group DotKRow + scales | ✅ on-device k GEMV |
| IQ1_S | ✅ | ✅ fused DotIQRow + scales | ✅ on-device IQ GEMV |
| IQ2_XXS | ✅ | ✅ fused DotIQRow + scales | ✅ on-device IQ GEMV |
| IQ2_XS | ✅ | ✅ fused DotIQRow + scales | ✅ on-device IQ GEMV |
| IQ3_XXS | ✅ | ✅ fused DotIQRow + scales | ✅ on-device IQ GEMV |
| IQ3_S | ✅ | ✅ fused DotIQRow + scales | ✅ on-device IQ GEMV |
| IQ4_NL | ✅ | ✅ fused DotIQRow + NL grid | ✅ on-device IQ GEMV |
| IQ4_XS | ✅ | ✅ fused DotIQRow + scales | ✅ on-device IQ GEMV |
| TernaryPacked | ✅ | ✅ BitNet code-dot SIMD | ✅ on-device ternary GEMV |
| BinaryPacked | ✅ | ✅ bit-fused DotBinaryWord | ✅ on-device binary GEMV |
| AffinePacked | ✅ | ✅ fused Affine packed code-dot | ✅ resident Affine GEMV + GEMVT |

Legend for this table:
- ✅ = fused / native packed path for that backend (no per-call full-matrix unpack; k/IQ/Affine SIMD = once-project Int8QS + scales, not F32 inflate)
- Peak cell = fused k/IQ `DotKRow`/`DotIQRow` (no F32 inflate); optional dedicated Plan 9 `.s` is future polish, not a scorecard blocker

---

## Backends

| Backend | Status | Requirement |
|---------|:------:|-------------|
| CPU tiled | ✅ | SC+MC; `weights.MatVec` / `MatVecT` stream native + packed |
| Plan 9 SIMD | ✅ | amd64 AVX2+FMA / arm64 NEON; unsupported arch → hard error |
| WebGPU | ✅ | Real device; FormatNone+quant GEMV/GEMVT + DenseDW; no host fake-GPU |

---

## Package feature board

Row **Wt** is the share of that package inside its scorecard section (not additive across the whole README — see [Version scorecard](#version-scorecard) for the 100-pt total).

### Core / infra — scorecard §1 (15 pts) + Dense §2 (15 pts) share

| Package | Features | Wt | Status |
|---------|----------|---:|:------:|
| `core/` | 34 DTypes, `Numeric`, `Tensor[T]`, activations, Layer/Network, Backend enum | 3 | ✅ |
| `weights/` | FormatNone pack/stream MatVec (f64 acc), SelectWire F32/F64/I8, DecodeRow(F64), ApplySGD (in-dtype / re-Pack; no retained f32 master) | 3 | ✅ |
| `quant/` | All 20 formats Pack/Unpack/MatVec/MatVecT | 3 | ✅ |
| `simd/` | DotTile, DotI8/U8, DotQ4_0, Saxpy, BitNet helpers (amd64/arm64 `.s`) | 3 | ✅ |
| `webgpu/` | Dense GEMV/GEMVT/DenseDW + `norm` / `softmax` / `swiglu_fuse` shaders | 2 | ✅ |
| `tiling/` | Tile size / SC / MC / GPU workgroup caps | 1 | ✅ |
| `layers/dense/` | Shared MatVec microkernel; FormatNone×34 + quants × 3 backends; grad verify | 15 | ✅† |

† Dense package is ✅ for API/suites and fused SIMD for all 20 quants (scorecard §2 → 15.0). Fused k/IQ Dot* is the §10 peak cell (optional dedicated `.s` is future polish).

### Runtime / architecture — scorecard §6 (8 pts)

| Package | Features | Wt | Status |
|---------|----------|---:|:------:|
| `architecture/` | Volumetric grid, cells, hops, remote links, Op bind | 2 | ✅ |
| `runtime/forward/` | Grid walk; Dense…Residual + ConvT + Parallel + KMeans + Mamba + Metacognition + GDN | 2 | ✅ |
| `runtime/backward/` | Reverse tape; same layer set | 2 | ✅ |
| `runtime/training/` | MSE + SGD; ApplyGradSGD → store ApplySGD for same layer set; storage-truth train | 1 | ✅ |
| `runtime/step/` | Discrete-time volumetric step mesh — Forward/Backward/ApplyTween/StepMesh; Cross-Numeric Train (kinds × weight dtype × act host) | 1 | ✅ |
| `lucy/` | Live Acc / SoftAcc / Avail / Score + `BuildLPD` density board (consciousness radar, LPD, gold/trap) | — | ✅ |

New hosts should import `github.com/openfluke/welvet/lucy` and call `Finalize` + `BuildLPD`. Tide dash/PDF only *display* that board. Score uses **hard Acc**, not SoftAcc.

```
Score = Throughput × Availability × Acc / 10_000
Q     = geomean(RelAcc, RelThru, RelAvail) vs learner peaks
LPD   = Q × shrink vs Acc-champ RAM   (0 unless RelAcc ≥ 70%)
```

Consciousness radar = Acc/Thru/Avail keep. Memory density radar = those × shrink (traps at origin). See `lucy/README.md`.

### Training credit — scorecard §9 (8 pts)

Full mode list + equations: **[`training_modes.md`](training_modes.md)**.

Sandwich `TrainStackMSE` / `TrainStackCE` / `OpenSplitTape`. Step\* and non-Step of the same family share one update on Stack (no Grid). Step\* is the 1D systolic pipe (`IsLineStep` / `TrainLine`; fill ticks do not update). Mesh\* needs volumetric placement. Classification hosts call **CE** (softmax − one-hot); regression hosts keep MSE. Credit modes are the same walk. Tables use `ShortTrainMode` (`[T]` Tween, `[S]` Split, `[FP]` FastProxy, `[L]` Linear, `[HP]` HeadProxy); persistence still uses `String()`.

| Mode | Credit | Status |
|------|--------|:------:|
| `NormalBP` / `StepBP` / `MeshBP` | Real chain rule (`BackwardStack` + SGD) | ✅ |
| `Tween` / `StepTween` / `MeshTween` | Broadcast \(P(g_y)\), half LR | ✅ |
| `TweenChain` / `StepTweenChain` / `MeshTweenChain` | Chain rule (= BP on Stack) | ✅ |
| `TweenSplit` / `StepTweenSplit` | \(g_i=\frac1N P(g_y)\) | ✅ |
| `TweenAlt` / `StepTweenAlt` | Split then Tween, `AltTimes` | ✅ |
| `TweenSplitHeadProxy` / `StepTweenSplitHeadProxy` | Head \(J^\top g_y\); hidden `dW` only | ✅ |
| `TweenSplitLinear` / `StepTweenSplitLinear` | Affine \(W^\top\) walk, skip act′ | ✅ |
| `TweenSplitFastProxy` / `StepTweenSplitFastProxy` | \(g_{\mathrm{proxy}}=W_{\mathrm{head}}^\top g_y\) (no act′); hidden `dW` only | ✅ |
| `TweenSplitHeadProxyAsync` / `StepTweenSplitHeadProxyAsync` | Hidden uses proxy from sample \(T-1\) | ✅ |
| `TweenSplitSparse` / `StepTweenSplitSparse` | Head + one rotating hidden leaf | ✅ |
| `TweenSplitLinearCache` / `StepTweenSplitLinearCache` | Cached Linear walk (dead on sine freq switch; kept for A/B) | ✅ |
| `MeshTweenSplit` / `MeshTweenAlt` / `MeshTweenSplitFastProxy` / `MeshTweenSplitSparse` | Grid-scheduled Split/Alt (family collapse on Stack) | ✅ |
| Cameral `BranchModes` / `HemispheresFrom` / `CombineAdd` | n=1 one mid Op; n=2/3 Bi/Tri | ✅ |
| **CamSync** (`CamSyncConfig` / `BlendStores`) | Soft/hard inter-cameral weight avg (α 1%→100%); groups; cross-layer same-shape pairs; after sample/step/pulse | ✅ |
| Dense `LinearGradIn` / `GradWOnly` | SIMD \(W^\top\) and `dW`-only kernels | ✅ |
| Cameral `.entity` | `WriteCameralFile` / `LoadCameral` (modes + weights) | ✅ |
| **test49** | All 29 named modes × 1³/2³/3³ origin smoke (one live cell; rest disabled) × Parallel + Bicameral + poly kinds (`w2a` `Test49AllTrainModesCubes`) | ✅ |

Rivaling backprop = matched **hard Acc** vs StepBP, not Lucy Score. Sparse Score is Avail (skip-GEMV). FastProxy is the Acc rival on sine/copy toys.

### CamSync — inter-cameral / cross-layer weight blend

Cams can diverge (Mix `BranchModes`, different inits). **CamSync** optionally pulls matching weight stores back together without collapsing the graph into one Dense:

| Knob | What it does |
|------|----------------|
| `Alpha` | Soft pull \(w_i ← (1-α)·w_i + α·\mathrm{mean}\) — `0.01` = 1%, `1.0` = hard average |
| `Groups` | Within one Parallel: all cams, or cliques e.g. `{{0,1},{2,3}}` |
| `Cross` | Explicit pairs via `SyncEndpoint{StackIdx, Branch, Store}` — same layer across cams, or **same-shaped** stores on different Stack children |
| `When` | `SyncAfterSample` / `SyncAfterStep` / `SyncAfterPulse` / `SyncManual` (+ `SyncNow`) |

**Shape rule:** blend only when `Rows×Cols` match. Mesh size does **not** matter — a cam on layer 1 of a `1×2×1` cell can sync to the same-shaped layer on cam 3 of a `2×4×5` (or any other layout) if both stores resolve and match. Today every link is **bidirectional** mean-blend; one-way teacher→student is not in yet. Separate Stack instances need a host-side `weights.BlendStores` call.

API: `parallel.CamSyncConfig`, `SetCamSync`, `SyncNow` / `Pulse` / `MaybeSync` · `weights.BlendStores` / `StoreCosine`. Hosts: `ok/cam_sync`.

### Layers (full w2a timed matrix + train grids; peak fused ALU in §10)

**§3 Transformer stack (14 pts)** · **§4 CNN/RNN/LSTM (6 pts)** · **§5 Extended (7 pts)**

| Package | Features | Wt | Section | Status |
|---------|----------|---:|---------|:------:|
| `layers/dense/` | Shared MatVec microkernel; FormatNone×34 + quants × 3 backends; packed SIMD/GPU; grad verify | — | §2 | ✅ |
| `layers/mha/` | Policy Mask/Pos/Mode; Dense proj coverage; attn ALU host; full timed matrix + train grids | 3 | §3 | ✅ |
| `layers/swiglu/` | Gate/Up/Down via Dense; WebGPU SiLU⊙ fuse (fwd); full timed matrix + train grids | 2 | §3 | ✅ |
| `layers/seqmix/` | Sequence-mixer kinds (attention / SSM / linear / conv) — contract only | 1 | §3 | ✅ |
| `layers/rmsnorm/` | Native RMS; WebGPU fwd+bwd; full timed matrix + train grids | 2 | §3 | ✅ |
| `layers/layernorm/` | Native LN; WebGPU fwd / bwd host; full timed matrix + train grids | 2 | §3 | ✅ |
| `layers/embedding/` | Token gather/scatter; full timed matrix + train grids | 1 | §3 | ✅ |
| `layers/softmax/` | All kinds CPU/SIMD; std/grid/hierarchical WebGPU; full timed matrix | 1 | §3 | ✅ |
| `layers/sequential/` | Sequential compose — Dense children + mixed SwiGLU/RMSNorm/LayerNorm via `NewFromOps`; full timed matrix + train grids | 1 | §3 | ✅ |
| `layers/residual/` | Residual y=F(x)+x — Dense or mixed F via `NewFromOps`; Parallel-as-F is `ResidualGraft`; full timed matrix | 1 | §3 | ✅ |
| `layers/cnn1/` | Conv1d im2col→Dense + FormatNone f32 tiled WebGPU; full timed matrix | 1 | §4 | ✅ |
| `layers/cnn2/` | Conv2d im2col→Dense + FormatNone f32 tiled WebGPU; full timed matrix | 1 | §4 | ✅ |
| `layers/cnn3/` | Conv3d im2col→Dense + FormatNone f32 tiled WebGPU; full timed matrix | 1 | §4 | ✅ |
| `layers/rnn/` | Vanilla tanh RNN; IH/HH via Dense; full timed matrix + train grids | 1.5 | §4 | ✅ |
| `layers/lstm/` | LSTM i/f/g/o via Dense; full timed matrix + train grids | 1.5 | §4 | ✅ |
| `layers/gdn/` | Gated DeltaNet; `Exec` CPU/SIMD/WebGPU; truncated BPTT; full timed matrix + train grids | 1.5 | §5 | ✅ |
| `layers/mamba/` | SSM selective scan; Dense projs; full timed matrix + train grids | 1 | §5 | ✅ |
| `layers/convt1/` | Transposed conv1d; host scatter+Proj; full timed matrix + train grids | 0.7 | §5 | ✅ |
| `layers/convt2/` | Transposed conv2d; host scatter+Proj; full timed matrix + train grids | 0.7 | §5 | ✅ |
| `layers/convt3/` | Transposed conv3d; host scatter+Proj; full timed matrix + train grids | 0.6 | §5 | ✅ |
| `layers/kmeans/` | Soft k-means; Centers via Dense; full timed matrix + train grids | 0.5 | §5 | ✅ |
| `layers/parallel/` | MoE concat/add/avg/filter; Dense branches; Split/FastProxy/Sparse credit; **CamSync** inter-cameral / cross-layer weight blend; full timed matrix + train grids | 1 | §5 | ✅ |
| `layers/metacognition/` | Observed Dense + rules; full timed matrix + train grids | 1 | §5 | ✅ |

### Dense detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| FormatNone × 34 dtypes — forward | ✅ | ✅ | ✅ |
| FormatNone × 34 dtypes — backward | ✅ | ✅ | ✅ native GEMVT + DenseDW |
| All 20 quants — forward | ✅ | ✅ all fused (Q*/k/IQ/Affine/BitNet; no F32 inflate for k/IQ/Affine) | ✅ on-device (all formats) |
| All 20 quants — backward | ✅ | ✅ packed MatVecT + Saxpy | ✅ GEMVT all formats + DenseDW |
| Fused k/IQ Dot* (no F32 inflate); optional dedicated `.s` | — | ✅ fused Go/int8 | — |
| SC + MC tiling | ✅ CPU SC+MC | ✅ row-parallel MC (`gemv_parallel`); CPU SC tile schedule 🚧 | ✅ workgroup caps |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Grad verify (CPU↔SIMD↔GPU + finite-diff) | ✅ | ✅ | ✅ |
| Train (fwd+MSE+bwd+SGD) FormatNone×34 + all quants | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |

### MHA detail (attention seqmix — transformers + diffusion ready)

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Mask: causal / bidirectional / sliding window / Prefix-LM / custom | ✅ | ✅ | ✅ |
| Pos: RoPE / none / ALiBi / RoPE+ALiBi | ✅ | ✅ | ✅ |
| Mode: self + cross (`ForwardWithContext`) | ✅ | ✅ | ✅ |
| GQA / MQA (`NumKVHeads`) + optional QK-RMSNorm | ✅ | ✅ | ✅ |
| Presets: Decoder / Encoder / Diffusion self+cross / PrefixLM / Local / ALiBi | ✅ | ✅ | ✅ |
| Q/K/V/O FormatNone × 34 — fwd+bwd | ✅ Dense projs | ✅ Dense projs | ✅ Dense projs |
| Q/K/V/O all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Attention / RoPE ALU | ✅ host | ✅ host (Enabled gate) | ✅ on-device (decoder gate) / host else |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| On-device attention / RoPE shaders | ⬜ | ⬜ | ✅ fwd+bwd (causal/bi; SoftmaxStandard; no train-drop) |
| SoftmaxSigmoid / train Dropout | ✅ | ✅ | ✅ host (GPU attn path skips when active) |

Non-attention mixers (Mamba/SSM, linear attn, Hyena) are **not** forks of `layers/mha/` — they land under `seqmix.Kind*` in their own packages.

### SwiGLU detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| SiLU(gate) ⊙ up → down | ✅ host | ✅ `simd.SiluMul*` | ✅ `webgpu.SwiGLUFuse` fwd+bwd |
| Gate/Up/Down FormatNone × 34 — fwd+bwd | ✅ Dense projs | ✅ Dense projs | ✅ Dense projs |
| Gate/Up/Down all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused SiLU⊙ shader / SIMD SiLU | ✅ | ✅ | ✅ fwd+bwd fuse |

### RMSNorm detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Per-token RMS + γ (eps=1e-6) | ✅ | ✅ DotTile Σx² + `RMSNormScaleF32` | ✅ `webgpu.RMSNorm` fwd+bwd |
| γ FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| γ all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Full SIMD scale (not just DotTile stats) | ✅ | ✅ | n/a |

### LayerNorm detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Per-token mean+var + γ/β (eps=1e-5) | ✅ | ✅ DotTile Σx/Σx² + `LayerNormScaleF32` | ✅ `webgpu.LayerNorm` fwd+bwd |
| γ+β FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| γ+β all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| On-device LayerNorm bwd + full SIMD scale | ✅ | ✅ | ✅ bwd |

### CNN1 detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Conv1d [B,C,L] + im2col → Dense GEMV | ✅ | ✅ via Dense | ✅ via Dense GEMV |
| Weights FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Weights all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device Conv1d shader (no im2col host) | ⬜ im2col | ⬜ im2col | ✅ FormatNone f32 tiled |

### CNN2 detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Conv2d [B,C,H,W] + im2col → Dense GEMV | ✅ | ✅ via Dense | ✅ via Dense GEMV |
| Weights FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Weights all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device Conv2d shader (no im2col host) | ⬜ im2col | ⬜ im2col | ✅ FormatNone f32 tiled |

### CNN3 detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Conv3d [B,C,D,H,W] + im2col → Dense GEMV | ✅ | ✅ via Dense | ✅ via Dense GEMV |
| Weights FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Weights all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device Conv3d shader (no im2col host) | ⬜ im2col | ⬜ im2col | ✅ FormatNone f32 tiled |

### RNN detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Vanilla tanh RNN [B,T,In]→[B,T,Hid]; BPTT | ✅ | ✅ via Dense | ✅ FormatNone f32 fused; else Dense |
| W_ih / W_hh FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| W_ih / W_hh all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device RNN recurrence shader | ⬜ | ⬜ | ✅ FormatNone f32 fwd+bwd |

### LSTM detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| LSTM [B,T,In]→[B,T,Hid]; i/f/g/o + BPTT | ✅ | ✅ via Dense | ✅ FormatNone f32 fused; else Dense |
| Gate W_ih/W_hh FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Gate W_ih/W_hh all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device LSTM recurrence shader | ⬜ | ⬜ | ✅ FormatNone f32 fwd+bwd |

### Embedding detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Gather [B,T]→[B,T,E]; scatter dW; gradIn=0 | ✅ | ✅ host gather | ✅ on-device gather/scatter |
| Table FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Table all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device embedding gather/scatter shader | ⬜ | ⬜ | ✅ |

### Softmax detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Weightless Softmax […,C]; max-subtract + Jacobian×1/T | ✅ | ✅ `simd.SoftmaxF32` | ✅ `webgpu.Softmax` / `SoftmaxEx` |
| KindStandard (last-axis) + KindGrid + Temperature | ✅ | ✅ | ✅ |
| No weight store — dtype/quant harness axes exercise ALU only | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 (ALU cells) | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Sparsemax / Entmax / Gumbel / Masked | ✅ all kinds | ✅ all kinds | ✅ `SoftmaxEx` on-device |
| Softmax SIMD kernel (not host ALU) | ✅ | ✅ | n/a |

### Sequential detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Dense→Dense chain in one cell (not grid hops) | ✅ | ✅ via Dense | ✅ via Dense |
| Child weights FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Child weights all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Nested non-Dense children (SwiGLU / RMSNorm / LayerNorm via `NewFromOps`) | ✅ | ✅ | ✅ |

### Residual detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| y = F(x) + x; F = Dense Dim→Dim (Depth≥1) | ✅ | ✅ via Dense | ✅ via Dense |
| Skip grad: gradIn = ∂F/∂x + ∂L/∂y | ✅ | ✅ | ✅ |
| F weights FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| F weights all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Nested non-Dense F (SwiGLU / RMSNorm / LayerNorm) / Parallel `ResidualGraft` | ✅ | ✅ | ✅ |

### Model / IO — scorecard §8 (8 pts)

| Package | Features | Wt | Status | Earned |
|---------|----------|---:|:------:|-------:|
| `model/tokenizer/` | BPE / HF tokenizers | 1.5 | ✅ | 1.5 |
| `model/entity/` | `.entity` Open/Inspect/Write + PackFromHF; cameral `WriteCameralFile` / `LoadCameral` (Stack, BranchModes, FastProxy); F32/F16/BF16/F64 LoadBlob | 2 | ✅ | 2.0 |
| `model/transformer/` | Decoder generate, KV cache, LM head; TopK/temp/greedy GenOptions | 2.5 | ✅ | 2.5 |
| `model/sampling/` | ArgMax, SampleTopK, penalties, BanIDs, chat sanitize | 1 | ✅ | 1.0 |
| `model/hf/` | InspectSnapshot + DetectArchitecture + safetensors/MLX loaders | 1 | ✅ | 1.0 |
| | **§8 subtotal** | **8** | | **8.0** |

### Systems — scorecard §7 (5 pts)

| Package | Features | Wt | Status |
|---------|----------|---:|:------:|
| `systems/dna/` | Topology DNA — all implemented Ops + GDN blobs; FlattenF32 across dtype×quant | 1 | ✅ |
| `systems/evolution/` | DNA splice + NEAT — clones all implemented Ops; dtype/quant preserved via SetFromF32 | 1 | ✅ |
| `systems/tween/` | Target prop — BackendSIMD DotTile/Saxpy chain-rule; Hebbian Saxpy + DotTile budgets; all weighted Ops | 1 | ✅ |
| `systems/tanhi/` | UDP HUD telemetry — all implemented Ops × dtype/quant via FlattenOp | 1 | ✅ |
| `systems/telemetry/` | Structural blueprint — all implemented Ops (+ meta estimates) | 1 | ✅ |

### Off-board (not v1 pts) — apps, stubs, NPU

Octo is a nested shell repo. Stubs are future. NPU/Metal/QNN are 3rd-party C++ — not a Welvet Go backend. Peak fused **is** scored (§10).

| Package | Features | Status |
|---------|----------|:------:|
| `apps/octo/` | Interactive model shell (download / convert / chat) | 🚧 |
| `stub/seed/` · `serialization/` · `hardware/` · `memory/` · `fountain/` · `donate/` | Future engine surfaces | 🚧 |
| `stub/accel/` | Intel NPU / Qualcomm / Apple Metal / QNN — later `welvet.cpp` | ⬜ |
| `fusedgpu/` | Q4_0 Engine + BinaryG128 HybridEngine; Android Vulkan optional (`WELVET_NATIVE_VK`) | ✅ (in §10) |

### Harness (not engine — does not count toward v1 pts)

| Package | Features | Status |
|---------|----------|:------:|
| `w2a/` | Interactive menu: **22 layer suites** + DNA/evolution/tween/step/seed/serialization/helpers; transformer stack has full 34×20×3 timed matrix | 🚧 |

---

## SIMD kernels on disk

| Kernel family | amd64 | arm64 | Wired into Dense |
|---------------|:-----:|:-----:|:----------------:|
| DotTile f32→f64 acc | ✅ | ✅ | ✅ FormatNone wire / lowp tiles |
| DotI8 / DotU8 | ✅ | ✅ | ✅ Int8 / Uint8 fwd |
| DotQ4_0 / Rows4 | ✅ | ✅ | ✅ Q4_0 fwd + packed bwd |
| Saxpy f32→f64 | ✅ | ✅ | ✅ FormatNone bwd |
| BitNet ternary / packed / TL1 | ✅ | ✅ | ✅ TernaryPacked / BinaryPacked |
| F16C cvtF16x8 + DotTile | ✅ amd64 | ✅ decode+DotTile | ✅ Float16 packed (no Wire cache) |

---

## Layer API contract

```go
// T is any core.Numeric — never assume float32
dense.Forward[T](layer, input) / dense.Backward[T](...)
mha.Forward[T](layer, input) / mha.Backward[T](...)  // input [batch,seq,d] or [seq,d]
ForwardCPUTiled[T] / ForwardSIMD[T] / ForwardWebGPU[T]
weights.New[T](...) / weights.MatVec[T](...) / weights.MatVecT[T](...)
```

Dispatcher: `core.ExecConfig.Backend` ∈ {`BackendCPUTiled`, `BackendSIMD`, `BackendWebGPU`}.

---

## Build & validate

```bash
# Engine only (no tests in welvet packages)
cd welvet && go build ./...

# Validation + timings
cd w2a
go run .                 # interactive
go test ./tests/dense -v # FormatNone timed matrix + gap census
go test ./tests/mha -v   # causal+RoPE+GQA; same coverage axes as Dense
go test ./tests/swiglu -v # SiLU-gated FFN; same coverage axes as Dense
go test ./tests/rmsnorm -v # RMSNorm γ; same coverage axes as Dense
go test ./tests/layernorm -v # LayerNorm γ+β; same coverage axes as Dense
go test ./tests/cnn1 -v    # Conv1d im2col→Dense; same coverage axes as Dense
go test ./tests/cnn2 -v    # Conv2d im2col→Dense; same coverage axes as Dense
go test ./tests/cnn3 -v    # Conv3d im2col→Dense; same coverage axes as Dense
go test ./tests/rnn -v     # vanilla tanh RNN; same coverage axes as Dense
go test ./tests/lstm -v    # LSTM i/f/g/o; same coverage axes as Dense
go test ./tests/embedding -v # token gather/scatter; same coverage axes as Dense
go test ./tests/softmax -v   # weightless Softmax; ALU harness (no weight store)
go test ./tests/sequential -v # Dense→Dense Sequential compose; same coverage axes as Dense
go test ./tests/residual -v  # Residual y=F(x)+x; same coverage axes as Dense
go test ./tests/gdn -v       # Gated DeltaNet; Exec + truncated BPTT; full timed matrix
go test ./tests/mamba -v     # SSM selective scan; full timed matrix
go test ./tests/convt1 -v    # ConvTranspose1d; full timed matrix
go test ./tests/convt2 -v    # ConvTranspose2d; full timed matrix
go test ./tests/convt3 -v    # ConvTranspose3d; full timed matrix
go test ./tests/kmeans -v    # Soft k-means; full timed matrix
go test ./tests/parallel -v  # MoE Parallel; full timed matrix
go test ./tests/metacognition -v # Observed Dense + rules; full timed matrix
```

Docs: `w2a/docs/`.

---

## Philosophy

Welvet is the fabric where **any AI op** can run on **any quant** at **any precision** on **any of the three backends**, with tiling and Plan 9 SIMD as first-class. Training credit (FastProxy / Sparse / …) is a first-class axis on a sandwich. NPU is not a fourth Go backend.

If something is hard, we **implement it** or **fail loudly**. We do not paper over gaps.
