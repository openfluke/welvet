# wav2vec2 (CTC ASR)

Native Welvet runner for [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h)
(~94M params, English CTC).

## Status

- FP32 greedy CTC transcription from 16 kHz mono WAV
- HF snapshot load (`config.json` + `vocab.json` + `model.safetensors`)
- Matches Hugging Face reference on the bundled smoke clip
- CPU Go GEMM/conv (parallelized); not yet on Dense/SIMD/WebGPU microkernels

## Quick start

```bash
# once: fetch weights
mkdir -p .cache/wav2vec2-base-960h && cd .cache/wav2vec2-base-960h
curl -LO https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json
curl -LO https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/vocab.json
curl -LO https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/preprocessor_config.json
curl -LO https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors

# transcribe
cd ../..
go run ./cmd/wav2vec2_transcribe .cache/wav2vec2-base-960h /path/to/audio.wav
```

## Entity

```bash
# via Octo tested models [7] → facebook/wav2vec2-base-960h
# or:
go run ./cmd/...  # PackFromHF on snapshot → octo_entities/facebook--wav2vec2-base-960h.entity
```

`octo transcribe` prefers the `.entity` when present.
## Test

```bash
WELVET_WAV2VEC2_DIR=$PWD/.cache/wav2vec2-base-960h \
  go test ./w2a/tests/wav2vec2 -count=1 -timeout 180s -v
```
