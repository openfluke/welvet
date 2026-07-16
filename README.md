# Welvet

**Welvet** is the AI engine: every layer, every numerical type, every quant / k-quant, every backend — native execution, no compromises.

| Repo | Role |
|------|------|
| **[openfluke/welvet](https://github.com/openfluke/welvet)** (this tree) | **Engine only** — layers, quant, SIMD (Plan 9 `.s`), WebGPU, ENTITY, dispatch |
| **[openfluke/w2a](https://github.com/openfluke/w2a)** (`w2a/`) | Tests, CABI, docs, menus — **never** in engine packages |

`loom/poly` is legacy reference only. Welvet is the rewrite.

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
| webgpu | Real device; all FormatNone + all quant fwd; GEMVT; DenseDW | ✅ |
| **Dense** FormatNone × 34 × CPU/SIMD/WebGPU fwd+bwd | 🚧 |
| **Dense** block-quant × SIMD/WebGPU (all 20 formats on-device fwd+bwd) | ✅ |
| `architecture/` volumetric grid (cells, hops, remote links) | ✅ |
| All other layers | ⬜ |
| Model IO / transformer / entity / tokenizer / hf | ⬜ |
| Accel / donate / fountain / dna / … | ⬜ |
| Full v1 matrix | ⬜ |

Validate live: `cd w2a && go test ./tests/dense -v` (timed FormatNone matrix + gap census).

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
| 0 | Float64 | ✅ | ✅ WireF64 | ✅ on-device f64→f32 | SIMD DotTileF64 |
| 1 | Float32 | ✅ | ✅ | ✅ FP32 WGSL | |
| 2 | Float16 | ✅ | ✅ F16C+DotTile | ✅ native decode | no Wire cache |
| 3 | BFloat16 | ✅ | ✅ packed+DotTile | ✅ native decode | |
| 4 | FP8E4M3 | ✅ native codec | ✅ packed+DotTile | ✅ native decode | real E4M3 |
| 5 | FP8E5M2 | ✅ native codec | ✅ packed+DotTile | ✅ native decode | real E5M2 |
| 6 | Int64 | ✅ | 🚧 WireF64 | ✅ on-device | |
| 7 | Int32 | ✅ | 🚧 WireF64 | ✅ on-device | |
| 8 | Int16 | ✅ | 🚧 WireF64 | ✅ on-device | |
| 9 | Int8 | ✅ | ✅ DotI8 | ✅ on-device I8 | |
| 10 | Uint64 | ✅ | 🚧 WireF64 | ✅ on-device affine | |
| 11 | Uint32 | ✅ | 🚧 WireF64 | ✅ on-device affine | |
| 12 | Uint16 | ✅ | 🚧 WireF64 | ✅ on-device affine | |
| 13 | Uint8 | ✅ | ✅ affine+DotTile | ✅ on-device affine | |
| 14 | Int4 | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 15 | Uint4 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 16 | FP4 | ✅ native E2M1 | ✅ packed+DotTile | ✅ native decode | |
| 17 | Int2 | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 18 | Uint2 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 19 | Ternary | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 20 | Binary | ✅ | ✅ expand→DotI8 | ✅ expand→I8 GEMV | |
| 21 | Int | ✅ | 🚧 WireF64 | ✅ on-device | Go native width |
| 22 | Uint | ✅ | 🚧 WireF64 | ✅ on-device affine | |
| 23 | Uintptr | ✅ | 🚧 WireF64 | ✅ on-device affine | |
| 24 | Complex64 | ✅ | 🚧 WireF64 | ✅ real-part GEMV | |
| 25 | Complex128 | ✅ | 🚧 WireF64 | ✅ real-part GEMV | |
| 26 | NF4 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device table | QLoRA |
| 27 | FP6 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device signed-6 | |
| 28 | Int6 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device signed-6 | |
| 29 | Uint6 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 30 | Int5 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device signed-5 | |
| 31 | Uint5 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |
| 32 | Int3 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device signed-3 | |
| 33 | Uint3 | ✅ | ✅ DecodeRow+DotTile | ✅ on-device affine | |

**SIMD:** Float16 uses F16C convert + AVX2/NEON DotTile; FP8/FP4/BF16 packed decode→DotTile (no `GPUWireF32` cache). Wide ints still WireF64.  
**WebGPU:** all 34 FormatNone dtypes on-device (Native / Ext / I8 / U8 / FP32 shaders).  
**✅** = dtype-specific path; 🚧 = host widen wire still used for that backend.

---

## Quant formats (`quant.Format`) — 20

CPU Pack/Unpack/MatVec/MatVecT vs Dense SIMD / WebGPU:

| Format | CPU pack+MatVec | Dense SIMD | Dense WebGPU |
|--------|:---------------:|:----------:|:------------:|
| None | ✅ (via `weights`) | 🚧 FormatNone matrix (+ F16C/FP8 packed) | ✅ all 34 dtypes on-device |
| Q8_0 | ✅ | ✅ fused DotI8×scale | ✅ on-device Q8 GEMV (in%32) |
| Q4_0 | ✅ | ✅ fused DotQ4_0 fwd | ✅ on-device Q4 GEMV (in%32) |
| Q4_1 | ✅ | ✅ block decode+DotTile | ✅ on-device Q4_1 |
| Q5_0 | ✅ | ✅ block decode+DotTile | ✅ on-device Q5 |
| Q5_1 | ✅ | ✅ block decode+DotTile | ✅ on-device Q5 |
| Q2_K | ✅ | ✅ group decode+DotTile | ✅ on-device k GEMV |
| Q3_K | ✅ | ✅ group decode+DotTile | ✅ on-device k GEMV |
| Q4_K | ✅ | ✅ group decode+DotTile | ✅ on-device k GEMV |
| Q5_K | ✅ | ✅ group decode+DotTile | ✅ on-device k GEMV |
| Q6_K | ✅ | ✅ group decode+DotTile | ✅ on-device k GEMV |
| IQ1_S | ✅ | ✅ group decode+DotTile | ✅ on-device IQ GEMV |
| IQ2_XXS | ✅ | ✅ group decode+DotTile | ✅ on-device IQ GEMV |
| IQ2_XS | ✅ | ✅ group decode+DotTile | ✅ on-device IQ GEMV |
| IQ3_XXS | ✅ | ✅ group decode+DotTile | ✅ on-device IQ GEMV |
| IQ3_S | ✅ | ✅ group decode+DotTile | ✅ on-device IQ GEMV |
| IQ4_NL | ✅ | ✅ group decode+DotTile | ✅ on-device IQ GEMV |
| IQ4_XS | ✅ | ✅ group decode+DotTile | ✅ on-device IQ GEMV |
| TernaryPacked | ✅ | ✅ BitNet code-dot SIMD | ✅ on-device ternary GEMV |
| BinaryPacked | ✅ | ✅ bit-fused DotBinaryWord | ✅ on-device binary GEMV |

✅ for a quant×backend cell = **fused** packed kernel (no full-matrix host unpack). 🚧 = functional via f32 SSBO stage.

---

## Backends

| Backend | Status | Requirement |
|---------|:------:|-------------|
| CPU tiled | 🚧 | SC+MC; stream native dtype; block quants via `quant.MatVec` |
| Plan 9 SIMD | 🚧 | amd64 AVX2+FMA / arm64 NEON; unsupported arch → hard error |
| WebGPU | 🚧 | Real `openfluke/webgpu` device; no host fake-GPU |

---

## Package feature board

### Core / infra

| Package | Features | Status |
|---------|----------|:------:|
| `core/` | 34 DTypes, `Numeric`, `Tensor[T]`, activations, Layer/Network, Backend enum | 🚧 |
| `weights/` | FormatNone pack/stream MatVec (f64 acc), SelectWire F32/F64/I8, DecodeRow(F64) | 🚧 |
| `quant/` | All 20 formats Pack/Unpack/MatVec/MatVecT | 🚧 |
| `simd/` | DotTile, DotI8/U8, DotQ4_0, Saxpy, BitNet helpers (amd64/arm64 `.s`) | 🚧 |
| `webgpu/` | All FormatNone + all quant GEMV/GEMVT + DenseDW | ✅ |
| `tiling/` | Tile size / SC / MC / GPU workgroup caps | ✅ |
| `dense/` | FormatNone×34 + all quants × 3 backends; packed fwd/bwd; grad verify | 🚧 |
| `architecture/` | Volumetric grid, cells, hops, remote links | ✅ |
| `forward/` | Global forward dispatch | ⬜ |
| `backward/` | Global backward dispatch | ⬜ |
| `training/` | Optimizers, graphs, native train | ⬜ |

### Layers (each needs CPU + SIMD + WebGPU × all dtype × all quant × fwd/bwd)

| Package | Features | Status |
|---------|----------|:------:|
| `dense/` | FormatNone×34 + all quants × 3 backends; Q4/Q8/Ternary/Binary packed; packed SIMD bwd; grad verify | 🚧 |
| `mha/` | Multi-head attention | ⬜ |
| `swiglu/` | SwiGLU FFN | ⬜ |
| `rmsnorm/` | RMSNorm | ⬜ |
| `layernorm/` | LayerNorm | ⬜ |
| `cnn1/` | 1D convolution | ⬜ |
| `cnn2/` | 2D convolution | ⬜ |
| `cnn3/` | 3D convolution | ⬜ |
| `convt1/` | 1D transposed conv | ⬜ |
| `convt2/` | 2D transposed conv | ⬜ |
| `convt3/` | 3D transposed conv | ⬜ |
| `rnn/` | RNN | ⬜ |
| `lstm/` | LSTM | ⬜ |
| `embedding/` | Embedding | ⬜ |
| `kmeans/` | K-means | ⬜ |
| `softmax/` | Softmax variants | ⬜ |
| `parallel/` | Parallel compose | ⬜ |
| `sequential/` | Sequential compose | ⬜ |
| `residual/` | Residual | ⬜ |
| `metacognition/` | Metacognition | ⬜ |

### Dense detail (only layer with real coverage today)

| Feature | CPU | SIMD | WebGPU |
|---------|:---:|:----:|:------:|
| FormatNone × 34 dtypes — forward | ✅ | 🚧 | ✅ on-device (all 34) |
| FormatNone × 34 dtypes — backward | ✅ | 🚧 | 🚧 GEMVT f32/stage + DenseDW |
| All 20 quants — forward | ✅ | ✅ block/bit fused | ✅ on-device (all formats) |
| All 20 quants — backward | ✅ | ✅ packed MatVecT + Saxpy | ✅ GEMVT all formats + DenseDW |
| True packed dtype/quant kernels (no f32 wire) | ⬜ | ✅ f16/fp8/fp4 packed | ✅ FormatNone+quant |
| SC + MC tiling | ✅ | 🚧 | ✅ workgroup caps |
| Timed FormatNone + quant matrices in `w2a` | ✅ | ✅ | ✅ |
| Grad verify (CPU↔SIMD↔GPU + finite-diff) | ✅ | ✅ | ✅ |

### Model / IO / runtime

| Package | Features | Status |
|---------|----------|:------:|
| `entity/` | `.entity` native checkpoints | ⬜ |
| `transformer/` | Decoder generate, KV cache, LM head (all quants) | ⬜ |
| `sampling/` | TopK, greedy, penalties | ⬜ |
| `tokenizer/` | BPE / HF tokenizers | ⬜ |
| `hf/` | HuggingFace → native packs | ⬜ |
| `seed/` | Seed manifests / infinite init | ⬜ |
| `serialization/` | Bit-perfect native I/O | ⬜ |

### Systems

| Package | Features | Status |
|---------|----------|:------:|
| `accel/` | Intel NPU / Qualcomm / Apple Metal / … | ⬜ |
| `hardware/` | Host probes | ⬜ |
| `memory/` | Footprint / VRAM accounting | ⬜ |
| `fountain/` | Fountain codes | ⬜ |
| `donate/` | LAN donate-compute | ⬜ |
| `tanhi/` | UDP telemetry | ⬜ |
| `dna/` | Topology DNA | ⬜ |
| `evolution/` | Evolution | ⬜ |
| `telemetry/` | Telemetry | ⬜ |
| `tween/` | Tween / misc | ⬜ |

### Harness (not engine)

| Package | Features | Status |
|---------|----------|:------:|
| `w2a/` | Interactive menu, dense suite, timed matrix, gap census, docs | 🚧 |

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
```

Docs: `w2a/docs/`.

---

## Philosophy

Welvet is the fabric where **any AI op** can run on **any quant** at **any precision** on **any of the three backends**, with tiling and Plan 9 SIMD as first-class.

If something is hard, we **implement it** or **fail loudly**. We do not paper over gaps.

**v1 ships when this README’s feature board is all ✅.**
