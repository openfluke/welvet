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
| ✅ | Done — real path, no silent fallback |
| 🚧 | Partial — works with known gaps / wire-format bridges |
| ⬜ | Not started (stub `doc.go` only, or hard-error everywhere) |

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
| webgpu | Dense GEMV family + RMSNorm/Softmax/LayerNorm-fwd/SwiGLU-fuse shaders; no host fake-GPU | 🚧 |
| **Dense** FormatNone × 34 × CPU/SIMD/WebGPU fwd+bwd | ✅ |
| **Dense** all 20 quants × SIMD/WebGPU (k/IQ/Affine SIMD = inflate+DotTile 🚧) | 🚧 |
| `architecture/` volumetric grid (cells, hops, remote links) | ✅ |
| `runtime/forward/` / `runtime/backward/` volumetric Dense + MHA + SwiGLU + RMSNorm + LayerNorm + CNN1–3 + RNN + LSTM + Embedding + Softmax + Sequential + Residual walk | ✅ |
| `runtime/training/` SGD on volumetric tape (Dense + MHA + SwiGLU + RMSNorm + LayerNorm + CNN1–3 + RNN + LSTM + Embedding + Softmax + Sequential + Residual) | ✅ |
| Remaining layers (parallel, …) | ⬜ |
| Model IO / transformer / entity / tokenizer / hf | 🚧 |
| `apps/octo/` interactive model shell (download / convert / chat) | 🚧 |
| Accel / donate / fountain / … | ⬜ |
| Full v1 matrix | ⬜ |

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
| Q4_1 | ✅ | ✅ block decode+DotTile | ✅ on-device Q4_1 |
| Q5_0 | ✅ | ✅ block decode+DotTile | ✅ on-device Q5 |
| Q5_1 | ✅ | ✅ block decode+DotTile | ✅ on-device Q5 |
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
| `core/` | 34 DTypes, `Numeric`, `Tensor[T]`, activations, Layer/Network, Backend enum | 🚧 |
| `weights/` | FormatNone pack/stream MatVec (f64 acc), SelectWire F32/F64/I8, DecodeRow(F64) | 🚧 |
| `quant/` | All 20 formats Pack/Unpack/MatVec/MatVecT | 🚧 |
| `simd/` | DotTile, DotI8/U8, DotQ4_0, Saxpy, BitNet helpers (amd64/arm64 `.s`) | 🚧 |
| `webgpu/` | Dense GEMV/GEMVT/DenseDW + `norm` / `softmax` / `swiglu_fuse` shaders | 🚧 |
| `tiling/` | Tile size / SC / MC / GPU workgroup caps | ✅ |
| `layers/dense/` | Shared MatVec microkernel; FormatNone×34 + quants × 3 backends; grad verify | ✅ |
| `layers/mha/` | Causal+RoPE+GQA; Q/K/V/O via Dense; attn ALU host; train grids | 🚧 |
| `layers/swiglu/` | Gate/Up/Down via Dense; SiLU⊙ host (SIMD) / device fuse (WebGPU fwd) | 🚧 |
| `layers/rmsnorm/` | Native RMS; DotTile SIMD; WebGPU fwd+bwd shaders; γ stores | 🚧 |
| `layers/layernorm/` | Native LN; DotTile SIMD; WebGPU fwd / bwd host; γ+β stores | 🚧 |
| `layers/cnn1/` | Conv1d im2col→Dense; no tiled conv shader yet | 🚧 |
| `layers/cnn2/` | Conv2d im2col→Dense; no tiled conv shader yet | 🚧 |
| `layers/cnn3/` | Conv3d im2col→Dense; no tiled conv shader yet | 🚧 |
| `layers/rnn/` | Vanilla tanh RNN; IH/HH via Dense; recurrence ALU host on GPU path | 🚧 |
| `layers/lstm/` | LSTM gates via Dense; recurrence ALU host on GPU path | 🚧 |
| `layers/embedding/` | Token gather/scatter; SIMD/WebGPU still host ALU | 🚧 |
| `layers/softmax/` | Softmax ALU; WebGPU std family; SIMD host | 🚧 |
| `layers/sequential/` | Dense→Dense Sequential compose | ✅ |
| `layers/residual/` | Residual y=F(x)+x (Dense F); heterogeneous residual ⬜ | 🚧 |
| `architecture/` | Volumetric grid, cells, hops, remote links, Op bind | ✅ |
| `runtime/forward/` | Grid walk z→y→x→l; Dense … Sequential + Residual dispatch | ✅ |
| `runtime/backward/` | Reverse tape over Dense … Sequential + Residual | ✅ |
| `runtime/training/` | MSE + SGD; ApplyGradSGD for Dense … Sequential / Residual | ✅ |

### Layers (each needs CPU + SIMD + WebGPU × all dtype × all quant × fwd/bwd)

Status here = **layer API + suite coverage**. GEMV-shaped work reuses Dense; 🚧 means host ALU or missing fused kernel remains (see Architecture).

| Package | Features | Status |
|---------|----------|:------:|
| `layers/dense/` | Shared MatVec microkernel; FormatNone×34 + quants × 3 backends; packed SIMD/GPU; grad verify | ✅ |
| `layers/mha/` | Policy Mask/Pos/Mode; Dense proj coverage; attn ALU host; no on-device attn yet | 🚧 |
| `layers/swiglu/` | Gate/Up/Down via Dense; WebGPU SiLU⊙ fuse (fwd); SIMD SiLU host | 🚧 |
| `layers/seqmix/` | Sequence-mixer kinds (attention / SSM / linear / conv) — contract only | ✅ |
| `layers/mamba/` | SSM / Mamba (KindSSM) | ⬜ |
| `layers/rmsnorm/` | Native RMS; WebGPU fwd+bwd; SIMD DotTile+host scale | 🚧 |
| `layers/layernorm/` | Native LN; WebGPU fwd / bwd host; SIMD DotTile+host scale | 🚧 |
| `layers/cnn1/` | Conv1d im2col→Dense; tiled conv shader ⬜ | 🚧 |
| `layers/cnn2/` | Conv2d im2col→Dense; tiled conv shader ⬜ | 🚧 |
| `layers/cnn3/` | Conv3d im2col→Dense; tiled conv shader ⬜ | 🚧 |
| `layers/rnn/` | Vanilla tanh RNN; IH/HH via Dense; recurrence ALU host on GPU | 🚧 |
| `layers/lstm/` | LSTM i/f/g/o via Dense; recurrence ALU host on GPU | 🚧 |
| `layers/embedding/` | Token gather/scatter; SIMD/WebGPU still host ALU | 🚧 |
| `layers/softmax/` | Softmax ALU; WebGPU std family; SIMD host; exotic kinds host/error | 🚧 |
| `layers/sequential/` | Dense→Dense Sequential compose | ✅ |
| `layers/residual/` | Residual y=F(x)+x (Dense F); heterogeneous residual ⬜ | 🚧 |
| `layers/convt1/` | 1D transposed conv | ⬜ |
| `layers/convt2/` | 2D transposed conv | ⬜ |
| `layers/convt3/` | 3D transposed conv | ⬜ |
| `layers/kmeans/` | K-means | ⬜ |
| `layers/parallel/` | Parallel compose | ⬜ |
| `layers/metacognition/` | Metacognition | ⬜ |

### Dense detail

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| FormatNone × 34 dtypes — forward | ✅ | ✅ | ✅ |
| FormatNone × 34 dtypes — backward | ✅ | ✅ | ✅ native GEMVT + DenseDW |
| All 20 quants — forward | ✅ | 🚧 Q4/Q8/BitNet fused; k/IQ/Affine = F32Cache+DotTile | ✅ on-device (all formats) |
| All 20 quants — backward | ✅ | ✅ packed MatVecT + Saxpy | ✅ GEMVT all formats + DenseDW |
| True packed dtype/quant kernels (no f32 wire) | ✅ MatVec stream | 🚧 (see quant table) | ✅ |
| SC + MC tiling | ✅ | 🚧 | ✅ workgroup caps |
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
| Sparsemax / Entmax / Gumbel / Masked on WebGPU | host helpers | host | ⬜ hard-error (no silent host) |
| Softmax Plan 9 SIMD kernel | ⬜ | ⬜ | n/a |

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
| `stub/seed/` | Seed manifests / infinite init | ⬜ |
| `stub/serialization/` | Bit-perfect native I/O | ⬜ |

### Systems

| Package | Features | Status |
|---------|----------|:------:|
| `stub/accel/` | Intel NPU / Qualcomm / Apple Metal / … | ⬜ |
| `stub/hardware/` | Host probes | ⬜ |
| `stub/memory/` | Footprint / VRAM accounting | ⬜ |
| `stub/fountain/` | Fountain codes | ⬜ |
| `stub/donate/` | LAN donate-compute | ⬜ |
| `systems/tanhi/` | UDP HUD telemetry — all implemented Ops × dtype/quant via FlattenOp | ✅ |
| `systems/dna/` | Topology DNA — all implemented Ops + GDN blobs; FlattenF32 across dtype×quant | ✅ |
| `systems/evolution/` | DNA splice + NEAT — clones all implemented Ops; dtype/quant preserved via SetFromF32 | ✅ |
| `systems/telemetry/` | Structural blueprint — all implemented Ops (+ meta estimates) | ✅ |
| `systems/tween/` | Target prop — BackendSIMD DotTile/Saxpy chain-rule; Hebbian Saxpy + DotTile budgets; all weighted Ops | ✅ |
| `runtime/step/` | Discrete-time volumetric step mesh — Forward/Backward/ApplyTween; all Ops × dtype × quant × CPU/SIMD | ✅ |

### Harness (not engine)

| Package | Features | Status |
|---------|----------|:------:|
| `w2a/` | Interactive menu: layer suites + dna/evolution/tween/step with **14 layers × 34 dtypes × 21 quants × CPU/SIMD** full census; timed matrices | 🚧 |

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
