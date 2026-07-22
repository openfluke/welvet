# qwentts — native Go Qwen3-TTS-12Hz (CustomVoice)

A pure Go/Welvet implementation of **Qwen3-TTS-12Hz CustomVoice** (0.6B first).
No Python runtime. Loads the HuggingFace snapshot directly (safetensors + JSON
config + BPE tokenizer) and synthesizes 24 kHz mono PCM.

The public API mirrors `apps/mosstts`:

```go
p, err := qwentts.LoadPipeline("/path/to/qwen3-tts-12hz-customvoice-0.6b")
if err != nil { log.Fatal(err) }

// write a WAV
path, err := p.SpeakToFile("Hello from Octo.", "./out", qwentts.SpeakOpts{
    Speaker:  "Ryan",
    Language: "English", // or "Auto"
    DoSample: true,
    Seed:     42,
})

// or get raw samples
samples, sr, ch, err := p.Speak("Hello.", qwentts.SpeakOpts{Speaker: "Ryan"})
```

## Snapshot layout (manual download)

Download is **not** performed by this package. Fetch the snapshot via the Octo
hub (or `huggingface-cli download`) into a manual-download directory, e.g.:

```
qwen3-tts-12hz-customvoice-0.6b/
├── config.json
├── model.safetensors            # talker.* + talker.code_predictor.*
├── vocab.json
├── merges.txt
└── speech_tokenizer/
    ├── config.json
    └── model.safetensors        # decoder.* (SplitRVQ + pre_transformer + upsample + Snake)
```

Point `LoadPipeline` at the directory that contains `config.json`.

## What runs

| Stage | Status |
|-------|--------|
| Config parsing (talker / code_predictor / speech_tokenizer) | ✅ |
| BPE text tokenizer (vocab.json + merges.txt, GPT-2 byte-level) | ✅ |
| Talker Qwen3 LLM (KV-cache, RMSNorm, Q/K-norm, GQA, SwiGLU, RoPE) | ✅ |
| Code predictor (5-layer MTP, 15 groups) | ✅ |
| CustomVoice generate loop (speaker/language/codec special tokens) | ✅ |
| SplitResidualVQ decode + pre_conv + pre_transformer + ConvNeXt upsample + SnakeBeta decoder → 24 kHz | ✅ |
| Voice clone / voice design / instruct (0.6B ignores instruct) | ❌ (not implemented — returns clear errors) |
| SIMD fuse (`simd_fuse`) | ✅ SIMD GEMV everywhere + SIMD attention dots |
| GPU fuse (`gpu_fuse`) | ✅ resident FP32 talker decode — one WebGPU submit per token |

Positional encoding uses standard sequential RoPE. For pure text/audio token
streams the MRoPE sections collapse to 1-D positions, so this matches the
reference; true multi-axis MRoPE is not required for CustomVoice.

## Fusion modes: `simd_fuse` vs `gpu_fuse`

Both modes are selected via `SpeakOpts{FuseSIMD, FuseGPU}` (the Octo menu
exposes them as **SIMD fuse** and **GPU fuse**). `FuseGPU` implies `FuseSIMD`
(SIMD is the host fallback for anything not on the GPU).

### `simd_fuse` (`FuseSIMD: true`)

Pure-CPU path. Every dense projection uses the SIMD GEMV kernel and attention
uses SIMD dot products. No WebGPU device required. Works on any GOARCH that has
SIMD kernels (falls back to scalar host otherwise).

### `gpu_fuse` (`FuseGPU: true`) — the real win

The talker (the autoregressive Qwen3 LLM, run ~300× per utterance) executes as
a **true fuse**: all layer weights and the KV cache stay resident in VRAM and a
whole decode token — every layer's `RMSNorm → Q/K/V GEMV → head-RMS → RoPE →
WriteKV → GQA attention → O GEMV → residual → RMSNorm → gate/up GEMV → SwiGLU →
down GEMV → residual`, then the final `RMSNorm` — runs in **one command encoder
and one `Submit`**, with a single readback of the last hidden state. The decode
position is a GPU-resident counter advanced on-device each token.

This is fundamentally different from the old "GPU fuse", which only offloaded
each large `Linear` as an isolated sticky GEMV (`DenseGEMVF32Resident`). That
sticky per-GEMV path incurred a submit + readback **per matrix**, so per-token
latency was dominated by dispatch/readback overhead. The resident fuse pays that
cost once per token instead of ~10× per layer.

Details:

- **Prefill** is processed by feeding each prompt token through the same
  one-submit decode step sequentially (path B). Causal attention with a growing
  KV cache makes this mathematically identical to a batched prefill, and TTS
  prompts are short relative to the ~300 decode frames, so this is cheap.
- **`maxSeq` is capped at 2048** (the attention-scores workgroup array bound);
  `headDim` must be ≤ 128. Qwen3-TTS-0.6B (headDim 128, GQA 16/8) fits.
- The non-fused talker heads (`text_projection`, `codec_head`) and the code
  predictor + decoder keep SIMD (with optional sticky FP32 GEMV) — the talker is
  where the autoregressive cost lives.
- Enabling GPU fuse builds the resident engine in `Talker.SetFuse`/pipeline
  `ApplyFuse` and prints `qwen GPU fuse: talker decode (one submit/token,
  maxSeq=N)`. `CloseGPU` tears the fuse down and clears sticky VRAM.
- If the WebGPU device or fuse init fails, the talker transparently falls back
  to the SIMD host path.

## Files

- `config.go` — parse `config.json` + nested `talker_config` / `code_predictor_config` + `speech_tokenizer/config.json`
- `tokenize.go` — GPT-2 byte-level BPE (vocab.json + merges.txt)
- `common.go` — Linear/embed loaders, RMSNorm, LayerNorm, SiLU/GELU, softmax, RoPE cache
- `talker.go` — Qwen3 talker: embeddings, text_projection, codec_head, KV cache, host+fuse forward
- `fuse_engine.go` — `talkerFuse`: resident FP32 GPU decode engine (one submit/token)
- `fuse_shaders.go` — WGSL for the talker fuse (GEMV, RMSNorm, head-RMS, RoPE, WriteKV, GQA attn, SwiGLU, IncPos)
- `code_predictor.go` — 5-layer MTP predicting codebooks 1..15
- `sample.go` — temperature / top-k / top-p / repetition-penalty sampling
- `codec_decode.go` — speech-tokenizer decoder (SplitRVQ → PCM)
- `generate.go` — CustomVoice prompt construction + autoregressive frame loop
- `pipeline.go` / `wav.go` — Octo façade + 16-bit PCM WAV writer
- `_ref/` — reference Python (`modeling_qwen3_tts.py`, `tokenizer_v2.py`, `inference.py`) and weight-key lists

## Build

```
cd welvet && go build ./apps/qwentts/
```
