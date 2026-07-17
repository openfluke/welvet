// Package flux2 implements a Flux2 Klein (MMDiT) image-generation foundation for Welvet.
//
// Target checkpoint: prism-ml/bonsai-image-binary-4B-mlx-1bit
// Architecture: Flux2Transformer2DModel — double-stream transformer_blocks then
// single-stream single_transformer_blocks, FlowMatchEulerDiscreteScheduler, and
// AutoencoderKLFlux2 VAE decode (LoadVAEFromDir → Decode).
//
// Quantized linears use MLX AffineQuantized 1-bit g128 (U32 weights + BF16 scales/biases)
// via welvet/quant BinaryG128. Skip patterns (proj_out, embedders, time_*, norm_out,
// *_modulation*) stay dense BF16. VAE weights stay BF16/F16 dense.
//
// Text prompts: EncodePrompt is stubbed (MLX AffineQuantized 4-bit g64 Qwen3 not ready);
// LoadPromptEmbeds accepts .npy / raw float32 [S,7680] for transformer smoke tests.
// Klein extraction (when implemented): hidden_states layers (9,18,27) → concat to 7680.
//
// Reference: huggingface/diffusers v0.37.1 transformer_flux2.py / pipeline_flux2_klein.py /
// autoencoder_kl_flux2.py.
package flux2
