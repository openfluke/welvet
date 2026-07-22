package fusedgpu

// Shader / fuse hard limits. Keep these in sync with the WGSL workgroup arrays
// in shaders.go / hybrid_shaders.go (scores: array<f32, N>, qv: array<f32, H>).
const (
	// AttnScoresMaxSeq is the WGSL workgroup `scores` array length used by both
	// the dense Q4 fuse attn shader and the hybrid BinaryG128 attn shader.
	// KV length above this will write past the workgroup buffer.
	AttnScoresMaxSeq = 2048

	// HybridAttnMaxHeadDim is the hybrid attn `qv` workgroup array length.
	HybridAttnMaxHeadDim = 256

	// DenseFusedDefaultMaxSeq is the VRAM-safe default KV window for the dense
	// Q4 fused path (weights + scratch are heavier per-token than BinaryG128).
	DenseFusedDefaultMaxSeq = 256

	// DefaultMaxSeq is used when a model/entity does not declare a context length.
	DefaultMaxSeq = 2048

	// HostMaxSeqCap is a practical upper bound for CPU / host KV paths on laptops.
	HostMaxSeqCap = 8192
)

// ClampAttnMaxSeq returns desired clamped into [1, AttnScoresMaxSeq].
func ClampAttnMaxSeq(desired int) int {
	if desired <= 0 {
		desired = DefaultMaxSeq
	}
	if desired > AttnScoresMaxSeq {
		return AttnScoresMaxSeq
	}
	return desired
}

// ClampDenseFusedMaxSeq returns desired clamped into [1, AttnScoresMaxSeq],
// preferring DenseFusedDefaultMaxSeq when desired is unset.
func ClampDenseFusedMaxSeq(desired int) int {
	if desired <= 0 {
		desired = DenseFusedDefaultMaxSeq
	}
	if desired > AttnScoresMaxSeq {
		return AttnScoresMaxSeq
	}
	return desired
}

// ClampHostMaxSeq clamps a host/CPU context length to HostMaxSeqCap.
func ClampHostMaxSeq(desired int) int {
	if desired <= 0 {
		desired = DefaultMaxSeq
	}
	if desired > HostMaxSeqCap {
		return HostMaxSeqCap
	}
	return desired
}

// EstimateHybridKVBytes estimates K+V cache bytes for full_attention layers
// at the given maxSeq (float32 caches).
func EstimateHybridKVBytes(numFullAttnLayers, numKVHeads, headDim, maxSeq int) int64 {
	if numFullAttnLayers <= 0 || numKVHeads <= 0 || headDim <= 0 || maxSeq <= 0 {
		return 0
	}
	// 2 caches (K,V) × layers × heads × seq × dim × 4 bytes
	return int64(numFullAttnLayers) * int64(numKVHeads) * int64(headDim) * int64(maxSeq) * 8
}

// ClampMaxSeqForKVBudget lowers maxSeq so EstimateHybridKVBytes stays under budget.
// budgetBytes <= 0 means "no budget" (only shader clamp applies).
func ClampMaxSeqForKVBudget(desired, numFullAttnLayers, numKVHeads, headDim int, budgetBytes int64) int {
	maxSeq := ClampAttnMaxSeq(desired)
	if budgetBytes <= 0 || numFullAttnLayers <= 0 || numKVHeads <= 0 || headDim <= 0 {
		return maxSeq
	}
	for maxSeq > 64 {
		if EstimateHybridKVBytes(numFullAttnLayers, numKVHeads, headDim, maxSeq) <= budgetBytes {
			return maxSeq
		}
		maxSeq /= 2
	}
	return maxSeq
}
