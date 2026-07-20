# Welvet

**Welvet** is the AI engine: layers, numerical types, quants / k-quants, and backends (CPU tiled · Plan 9 SIMD · WebGPU). **Pre-v1** — the full dtype×quant×backend matrix is not green everywhere; see honesty notes below.

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
| **Native layer math** | Dense, RMSNorm, LayerNorm, Softmax, Embedding | Own fwd/bwd; norms/softmax have real WebGPU shaders; SIMD may still be DotTile+host scale or host ALU |
| **Composite → Dense projs** | MHA (Q/K/V/O), SwiGLU (Gate/Up/Down), RNN/LSTM/Residual/Sequential | Projections = Dense children (`syncProjExec`); attn / SiLU / recurrence ALU separate |
| **im2col → Dense** | CNN1/2/3 | Host im2col, then Dense GEMV (intentional; tiled conv shaders still ⬜) |

**This is intentional**, not a missing abstraction: one MatVec surface means one place for quant bugs, dtype wires, and backend parity. Separate native kernels pay off when the calc is **not** GEMV (fused attention, tiled CNN, Softmax/SiLU SIMD, true fused k-quant asm).

**Not “fully native” yet (honest):**
- k/IQ/**AffinePacked** SIMD often = **inflate-once F32Cache + DotTile** (not true fused k-quant `.s`)
- MHA attn, SwiGLU SiLU⊙ (SIMD), Softmax/Embedding under SIMD, CNN im2col = **host ALU**
- WebGPU device ALU is typically **f32** at the boundary (storage dtype narrows on upload)
- Suite honesty: `w2a/suites.StampBackendNote` / `AffinePackable` — no silent host counted as “WebGPU/SIMD done”

Remaining work: [`docs/loom_2_welvet_todolist.md`](../docs/loom_2_welvet_todolist.md).

### Tree layout

| Folder | Contains |
|--------|----------|
| (top) | `core`, `weights`, `quant`, `simd`, `webgpu`, `tiling`, `architecture`, `layers/` |
| `runtime/` | `forward`, `backward`, `training`, `step` |
| `systems/` | `dna`, `evolution`, `tween`, `tanhi`, `telemetry` |
| `model/` | `transformer`, `entity`, `tokenizer`, `sampling`, `hf` |
| `apps/` | `octo`, `flux2`, `mosstts` |
| `stub/` | future: `accel`, `donate`, `fountain`, `hardware`, `memory`, `seed`, `serialization` |
| `w2a/`, `tools/` | harness (not engine) |


**Status: pre-v1.** v1 ships only when every row below is ✅.

| Legend | Meaning |
|--------|---------|
| ✅ | **Implemented** — layer/runtime/stub API works; w2a suite passes (full timed matrix for transformer stack; smoke+census for ConvT/Parallel/Mamba/etc.) |
| 🚧 | **Partial** — works but lighter coverage, inflate-not-fused SIMD, or host ALU on GPU/SIMD path |
| ⬜ | **Not started** — stub `doc.go` only, hard-error everywhere, or peak fused kernel explicitly missing |

---

## Snapshot (honest)

| Area | Status |
|------|--------|
| Engine layout (one feature → one folder) | ✅ |
| Rules: no engine tests / no fallbacks / no hardcoded float32 / no QAT | ✅ |
| `core` types (34 dtypes, Tensor\[T\], activations, backends) | ✅ |
| `weights` FormatNone × 34 stream pack/MatVec | ✅ |
| `quant` Pack/Unpack/MatVec all 20 formats (CPU) | ✅ |
| `simd` Plan 9 kernels linked (amd64/arm64) | ✅ |
| webgpu | Dense GEMV family + RMSNorm/Softmax/LayerNorm-fwd/SwiGLU-fuse shaders; attn/tiled-CNN ⬜ | ✅ |
| **Dense** FormatNone × 34 × CPU/SIMD/WebGPU fwd+bwd | ✅ |
| **Dense** all 20 quants — WebGPU fwd+bwd | ✅ |
| **Dense** k/IQ/Affine SIMD (inflate+DotTile, not fused `.s`) | 🚧 |
| `architecture/` volumetric grid (cells, hops, remote links) | ✅ |
| `runtime/forward/` / `backward` / `training` — Dense…Residual + ConvT1–3 + Parallel + KMeans + Mamba + Metacognition + GDN | ✅ |
| ConvT / Parallel / KMeans / Mamba / Metacognition / GDN — lighter w2a suites (smoke+census, not full 34×20 timed matrix) | 🚧 |
| Model IO / transformer / entity / tokenizer / hf | 🚧 |
| `apps/octo/` interactive model shell (download / convert / chat) | 🚧 |
| `stub/` seed · serialization · hardware · memory · fountain · donate | 🚧 |
| `stub/accel/` (NPU/Metal/QNN plugins) | ⬜ |
| Full v1 matrix (every cell peak-fused, no host ALU) | ⬜ |

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
```

---

## Non-negotiable rules

1. **No testing code in the engine tree** — all checks in `w2a/`.
2. **No fallbacks** — missing path → hard error (no SIMD→Go, no fake GPU).
3. **Nothing hardcoded to float32** — APIs are `Tensor[T]` / generics. Host wires are `WireF32` / `WireF64` / `WireI8` via `weights.SelectWire` (float64 & integers are **not** forced through f32). WebGPU WGSL ALU is f32 on typical adapters — narrowing happens only at the device boundary.
4. **No QAT** — `DType` + `QuantFormat` are storage truth.
5. **One poly feature → one folder.**
6. **v1 = checklist all ✅.**

---

## Axes (what “done” means per feature)

For each layer / op, every cell must work:

| Axis | Count | Values |
|------|------:|--------|
| Backend | 3 | CPU tiled (SC+MC) · Plan 9 SIMD · WebGPU |
| DType | 34 | `0…33` — table below |
| Quant | 20 | `None` + classic + k-quant + IQ + Ternary/Binary |
| Pass | 2 | forward **and** backward (where trainable) |

**No cell may silently substitute another cell.**

---

## DTypes (`core.DType`) — 34

Storage / weight element types. Dense **FormatNone** coverage today:

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

CPU Pack/Unpack/MatVec/MatVecT vs Dense SIMD / WebGPU:

| Format | CPU pack+MatVec | Dense SIMD | Dense WebGPU |
|--------|:---------------:|:----------:|:------------:|
| None | ✅ (via `weights`) | ✅ FormatNone packed/stream | ✅ all 34 fwd+GEMVT |
| Q8_0 | ✅ | ✅ fused DotI8×scale | ✅ on-device Q8 GEMV (in%32) |
| Q4_0 | ✅ | ✅ fused DotQ4_0 fwd | ✅ on-device Q4 GEMV (in%32) |
| Q4_1 | ✅ | ✅ fused DotQ4_1 | ✅ on-device Q4_1 |
| Q5_0 | ✅ | ✅ fused DotQ5 | ✅ on-device Q5 |
| Q5_1 | ✅ | ✅ fused DotQ5_1 | ✅ on-device Q5 |
| Q2_K | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device k GEMV |
| Q3_K | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device k GEMV |
| Q4_K | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device k GEMV |
| Q5_K | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device k GEMV |
| Q6_K | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device k GEMV |
| IQ1_S | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device IQ GEMV |
| IQ2_XXS | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device IQ GEMV |
| IQ2_XS | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device IQ GEMV |
| IQ3_XXS | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device IQ GEMV |
| IQ3_S | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device IQ GEMV |
| IQ4_NL | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device IQ GEMV |
| IQ4_XS | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ on-device IQ GEMV |
| TernaryPacked | ✅ | ✅ BitNet code-dot SIMD | ✅ on-device ternary GEMV |
| BinaryPacked | ✅ | ✅ bit-fused DotBinaryWord | ✅ on-device binary GEMV |
| AffinePacked | ✅ | 🚧 inflate-once F32Cache+DotTile | ✅ resident Affine GEMV |

Legend for this table:
- ✅ = fused / native packed path for that backend (no per-call full-matrix unpack)
- 🚧 = works via **once-inflated F32Cache + DotTile** (or f32 SSBO stage on GPU) — correct, not peak fused asm
- AffinePacked SIMD falls back to native `matVecAffine` when inflate is refused (size cap)

---

## Backends

| Backend | Status | Requirement |
|---------|:------:|-------------|
| CPU tiled | ✅ | SC+MC; `weights.MatVec` / `MatVecT` stream native + packed |
| Plan 9 SIMD | ✅ | amd64 AVX2+FMA / arm64 NEON; unsupported arch → hard error |
| WebGPU | ✅ | Real device; FormatNone+quant GEMV/GEMVT + DenseDW; no host fake-GPU |

---

## Package feature board

### Core / infra

| Package | Features | Status |
|---------|----------|:------:|
| `core/` | 34 DTypes, `Numeric`, `Tensor[T]`, activations, Layer/Network, Backend enum | ✅ |
| `weights/` | FormatNone pack/stream MatVec (f64 acc), SelectWire F32/F64/I8, DecodeRow(F64) | ✅ |
| `quant/` | All 20 formats Pack/Unpack/MatVec/MatVecT | ✅ |
| `simd/` | DotTile, DotI8/U8, DotQ4_0, Saxpy, BitNet helpers (amd64/arm64 `.s`) | ✅ |
| `webgpu/` | Dense GEMV/GEMVT/DenseDW + `norm` / `softmax` / `swiglu_fuse` shaders | ✅ |
| `tiling/` | Tile size / SC / MC / GPU workgroup caps | ✅ |
| `layers/dense/` | Shared MatVec microkernel; FormatNone×34 + quants × 3 backends; grad verify | ✅ |
| `layers/mha/` | Causal+RoPE+GQA; Q/K/V/O via Dense; attn ALU host; full w2a timed matrix | ✅ |
| `layers/swiglu/` | Gate/Up/Down via Dense; WebGPU SiLU⊙ fuse (fwd); full w2a timed matrix | ✅ |
| `layers/rmsnorm/` | Native RMS; WebGPU fwd+bwd shaders; full w2a timed matrix | ✅ |
| `layers/layernorm/` | Native LN; WebGPU fwd / bwd host; full w2a timed matrix | ✅ |
| `layers/cnn1/` | Conv1d im2col→Dense; full w2a timed matrix; tiled conv shader ⬜ | ✅ |
| `layers/cnn2/` | Conv2d im2col→Dense; full w2a timed matrix; tiled conv shader ⬜ | ✅ |
| `layers/cnn3/` | Conv3d im2col→Dense; full w2a timed matrix; tiled conv shader ⬜ | ✅ |
| `layers/rnn/` | Vanilla tanh RNN; IH/HH via Dense; full w2a timed matrix | ✅ |
| `layers/lstm/` | LSTM gates via Dense; full w2a timed matrix | ✅ |
| `layers/embedding/` | Token gather/scatter; full w2a timed matrix; fused GPU gather ⬜ | ✅ |
| `layers/softmax/` | All kinds on CPU/SIMD; std family on WebGPU; full w2a timed matrix | ✅ |
| `layers/sequential/` | Dense→Dense Sequential compose; full w2a timed matrix | ✅ |
| `layers/residual/` | Residual y=F(x)+x (Dense F); full w2a timed matrix; heterogeneous F ⬜ | ✅ |
| `layers/gdn/` | Gated DeltaNet; runtime+SGD+w2a suite; truncated BPTT; no `Core.Exec` grid dispatch | 🚧 |
| `layers/convt1/` | Transposed conv1d; runtime wired; smoke+census suite | 🚧 |
| `layers/convt2/` | Transposed conv2d; runtime wired; smoke+census suite | 🚧 |
| `layers/convt3/` | Transposed conv3d; runtime wired; smoke+census suite | 🚧 |
| `layers/mamba/` | SSM selective scan; runtime wired; smoke+census suite | 🚧 |
| `layers/kmeans/` | Soft k-means; runtime wired; smoke+census suite | 🚧 |
| `layers/parallel/` | MoE concat/add/avg/filter; runtime wired; smoke+census suite | 🚧 |
| `layers/metacognition/` | Observed Dense + rules; runtime wired; smoke+census suite | 🚧 |
| `architecture/` | Volumetric grid, cells, hops, remote links, Op bind | ✅ |
| `runtime/forward/` | Grid walk; Dense…Residual + ConvT + Parallel + KMeans + Mamba + Metacognition + GDN | ✅ |
| `runtime/backward/` | Reverse tape; same layer set | ✅ |
| `runtime/training/` | MSE + SGD; ApplyGradSGD for same layer set | ✅ |

### Layers (transformer stack = full w2a timed matrix; others = smoke+census)

| Package | Features | Status |
|---------|----------|:------:|
| `layers/dense/` | Shared MatVec microkernel; FormatNone×34 + quants × 3 backends; packed SIMD/GPU; grad verify | ✅ |
| `layers/mha/` | Policy Mask/Pos/Mode; Dense proj coverage; attn ALU host; full timed matrix + train grids | ✅ |
| `layers/swiglu/` | Gate/Up/Down via Dense; WebGPU SiLU⊙ fuse (fwd); full timed matrix + train grids | ✅ |
| `layers/seqmix/` | Sequence-mixer kinds (attention / SSM / linear / conv) — contract only | ✅ |
| `layers/rmsnorm/` | Native RMS; WebGPU fwd+bwd; full timed matrix + train grids | ✅ |
| `layers/layernorm/` | Native LN; WebGPU fwd / bwd host; full timed matrix + train grids | ✅ |
| `layers/cnn1/` | Conv1d im2col→Dense; full timed matrix; tiled conv shader ⬜ | ✅ |
| `layers/cnn2/` | Conv2d im2col→Dense; full timed matrix; tiled conv shader ⬜ | ✅ |
| `layers/cnn3/` | Conv3d im2col→Dense; full timed matrix; tiled conv shader ⬜ | ✅ |
| `layers/rnn/` | Vanilla tanh RNN; IH/HH via Dense; full timed matrix + train grids | ✅ |
| `layers/lstm/` | LSTM i/f/g/o via Dense; full timed matrix + train grids | ✅ |
| `layers/embedding/` | Token gather/scatter; full timed matrix + train grids | ✅ |
| `layers/softmax/` | All kinds CPU/SIMD; std/grid/hierarchical WebGPU; full timed matrix | ✅ |
| `layers/sequential/` | Dense→Dense Sequential compose; full timed matrix + train grids | ✅ |
| `layers/residual/` | Residual y=F(x)+x (Dense F); full timed matrix; heterogeneous F ⬜ | ✅ |
| `layers/gdn/` | Gated DeltaNet; runtime+SGD; w2a suite; truncated BPTT; grid `Exec` ⬜ | 🚧 |
| `layers/mamba/` | SSM selective scan; runtime wired; smoke+census w2a | 🚧 |
| `layers/convt1/` | Transposed conv1d; runtime wired; smoke+census w2a | 🚧 |
| `layers/convt2/` | Transposed conv2d; runtime wired; smoke+census w2a | 🚧 |
| `layers/convt3/` | Transposed conv3d; runtime wired; smoke+census w2a | 🚧 |
| `layers/kmeans/` | Soft k-means; runtime wired; smoke+census w2a | 🚧 |
| `layers/parallel/` | MoE concat/add/avg/filter; runtime wired; smoke+census w2a | 🚧 |
| `layers/metacognition/` | Observed Dense + rules; runtime wired; smoke+census w2a | 🚧 |

### Dense detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| FormatNone × 34 dtypes — forward | ✅ | ✅ | ✅ |
| FormatNone × 34 dtypes — backward | ✅ | ✅ | ✅ native GEMVT + DenseDW |
| All 20 quants — forward | ✅ | ✅ Q4/Q8/BitNet fused; k/IQ/Affine = F32Cache+DotTile | ✅ on-device (all formats) |
| All 20 quants — backward | ✅ | ✅ packed MatVecT + Saxpy | ✅ GEMVT all formats + DenseDW |
| True peak-fused k/IQ/Affine SIMD (no F32Cache) | — | ⬜ | — |
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
| Attention / RoPE ALU | ✅ host | ✅ host (Enabled gate) | ✅ host (proj on-device) |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| On-device attention / RoPE shaders | ⬜ | ⬜ | ⬜ |
| SoftmaxSigmoid / train Dropout | ⬜ hard-error | ⬜ | ⬜ |

Non-attention mixers (Mamba/SSM, linear attn, Hyena) are **not** forks of `layers/mha/` — they land under `seqmix.Kind*` in their own packages.

### SwiGLU detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| SiLU(gate) ⊙ up → down | ✅ host | ✅ host | ✅ `webgpu.SwiGLUFuse` (fwd); bwd combine host |
| Gate/Up/Down FormatNone × 34 — fwd+bwd | ✅ Dense projs | ✅ Dense projs | ✅ Dense projs |
| Gate/Up/Down all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused SiLU⊙ shader / SIMD SiLU | ⬜ SIMD | ⬜ | ✅ fwd fuse; ⬜ bwd fuse |

### RMSNorm detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Per-token RMS + γ (eps=1e-6) | ✅ | ✅ DotTile Σx²; scale host | ✅ `webgpu.RMSNorm` fwd+bwd |
| γ FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| γ all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Full SIMD scale (not just DotTile stats) | ⬜ | ⬜ | n/a |

### LayerNorm detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Per-token mean+var + γ/β (eps=1e-5) | ✅ | ✅ DotTile Σx/Σx²; scale host | ✅ `webgpu.LayerNorm` fwd; bwd host |
| γ+β FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| γ+β all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| On-device LayerNorm bwd + full SIMD scale | ⬜ | ⬜ | ⬜ bwd |

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
| Fused on-device Conv1d shader (no im2col host) | ⬜ | ⬜ | ⬜ |

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
| Fused on-device Conv2d shader (no im2col host) | ⬜ | ⬜ | ⬜ |

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
| Fused on-device Conv3d shader (no im2col host) | ⬜ | ⬜ | ⬜ |

### RNN detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Vanilla tanh RNN [B,T,In]→[B,T,Hid]; BPTT | ✅ | ✅ via Dense | ✅ device required; host ALU |
| W_ih / W_hh FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| W_ih / W_hh all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device RNN recurrence shader | ⬜ | ⬜ | ⬜ |

### LSTM detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| LSTM [B,T,In]→[B,T,Hid]; i/f/g/o + BPTT | ✅ | ✅ via Dense | ✅ device required; host ALU |
| Gate W_ih/W_hh FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Gate W_ih/W_hh all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device LSTM recurrence shader | ⬜ | ⬜ | ⬜ |

### Embedding detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Gather [B,T]→[B,T,E]; scatter dW; gradIn=0 | ✅ | ✅ host gather | ✅ device required; host ALU |
| Table FormatNone × 34 — fwd+bwd | ✅ | ✅ | ✅ |
| Table all 20 quants — fwd+bwd | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Fused on-device embedding gather/scatter shader | ⬜ | ⬜ | ⬜ |

### Softmax detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| Weightless Softmax […,C]; max-subtract + Jacobian×1/T | ✅ | ✅ host ALU (Enabled gate) | ✅ `webgpu.Softmax` std/temp/grid/hierarchical |
| KindStandard (last-axis) + KindGrid + Temperature | ✅ | ✅ | ✅ |
| No weight store — dtype/quant harness axes exercise ALU only | ✅ | ✅ | ✅ |
| Activation `Tensor[T]` × all 15 `core.Numeric` kinds | ✅ | ✅ | ✅ |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Gap census 34×20×3 (ALU cells) | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × FormatNone×34 × backends | ✅ | ✅ | ✅ |
| Train volumetric 1³/2³/3³ × all 20 quants × backends | ✅ | ✅ | ✅ |
| Sparsemax / Entmax / Gumbel / Masked | ✅ all kinds | ✅ all kinds (host ALU) | ⬜ hard-error (no silent host) |
| Softmax Plan 9 SIMD kernel (not host ALU) | ⬜ | ⬜ | n/a |

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
| Nested non-Dense children (Softmax/Residual/…) | ⬜ | ⬜ | ⬜ |

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
| Nested non-Dense F / Parallel residual graft | ⬜ | ⬜ | ⬜ |

### Model / IO / runtime

| Package | Features | Status |
|---------|----------|:------:|
| `entity/` | `.entity` native checkpoints | 🚧 |
| `model/transformer/` | Decoder generate, KV cache, LM head (all quants) | 🚧 |
| `model/sampling/` | TopK, greedy, penalties | 🚧 |
| `model/tokenizer/` | BPE / HF tokenizers | ✅ |
| `model/hf/` | HuggingFace → native packs | 🚧 |
| `stub/seed/` | Seed manifests / infinite init / He / mixed grids | 🚧 |
| `stub/serialization/` | ENTITY encode/decode / native I/O | 🚧 |

### Systems

| Package | Features | Status |
|---------|----------|:------:|
| `stub/accel/` | Intel NPU / Qualcomm / Apple Metal / … | ⬜ |
| `stub/hardware/` | Host probes / audit | 🚧 |
| `stub/memory/` | Footprint / VRAM accounting | 🚧 |
| `stub/fountain/` | Fountain codes + neural recover | 🚧 |
| `stub/donate/` | LAN donate-compute protocol (infer stub-echo) | 🚧 |
| `systems/tanhi/` | UDP HUD telemetry — all implemented Ops × dtype/quant via FlattenOp | ✅ |
| `systems/dna/` | Topology DNA — all implemented Ops + GDN blobs; FlattenF32 across dtype×quant | ✅ |
| `systems/evolution/` | DNA splice + NEAT — clones all implemented Ops; dtype/quant preserved via SetFromF32 | ✅ |
| `systems/telemetry/` | Structural blueprint — all implemented Ops (+ meta estimates) | ✅ |
| `systems/tween/` | Target prop — BackendSIMD DotTile/Saxpy chain-rule; Hebbian Saxpy + DotTile budgets; all weighted Ops | ✅ |
| `runtime/step/` | Discrete-time volumetric step mesh — Forward/Backward/ApplyTween; all Ops × dtype × quant × CPU/SIMD | ✅ |

### Harness (not engine)

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
```

Docs: `w2a/docs/`.

---

## Philosophy

Welvet is the fabric where **any AI op** can run on **any quant** at **any precision** on **any of the three backends**, with tiling and Plan 9 SIMD as first-class.

If something is hard, we **implement it** or **fail loudly**. We do not paper over gaps.

**v1 ships when this README’s feature board is all ✅.**
