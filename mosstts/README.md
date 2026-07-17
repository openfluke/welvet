# MOSS-TTS-Nano Welvet layout

Self-host a converted checkpoint (Welvet does not load `pytorch_model.bin` directly).

## Convert

```bash
# needs: pip install torch safetensors sentencepiece numpy
python3 octo/tools/moss_convert/convert.py \
  /path/to/MOSS-TTS-Nano-100M \
  ./octo_hub/models--YOU--moss-tts-nano-100m-welvet/snapshots/manual-download
```

Also download the audio tokenizer (already safetensors):

```bash
# via Octo [2] (now allows .bin / .model) or huggingface-cli
# OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano → octo_hub/models--OpenMOSS-Team--MOSS-Audio-Tokenizer-Nano/snapshots/manual-download
```

Optional: copy/symlink tokenizer into the TTS snapshot as `audio_tokenizer/`.

Expected TTS snapshot files after convert:

| File | Role |
|------|------|
| `model.safetensors` | Dense F32 AR weights |
| `vocab_pieces.json` | SentencePiece dump for Go encode |
| `config.json` / `tokenizer.model` | Copied from upstream |

## Runtime

Octo **[9] Generate speech** or:

```bash
./octo speak "Hello from Octo." [--ref ref.wav] [--frames 300] [--seed 42] [--greedy] [--simd|--no-simd] [--gpu]
```

Menu **[9]** prompts for SIMD fuse and GPU fuse. GPU warms sticky FP32 GEMV weights (~0.5 GiB for Nano); short prompts often prefer SIMD-only due to PCIe readback.
