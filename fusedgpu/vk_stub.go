//go:build !birdkit_native_vk

package fusedgpu

import "errors"

// ErrNotAvailable is returned when native Vulkan hybrid is unavailable.
// Experimental Android path: go build -tags 'cshared birdkit_native_vk' and WELVET_NATIVE_VK=1.
var ErrNotAvailable = errors.New("fusedgpu: native Vulkan hybrid unavailable (build -tags birdkit_native_vk on Android; WELVET_NATIVE_VK=1)")

// NativeVKAvailable reports whether the native Vulkan hybrid path can be used.
func NativeVKAvailable() bool { return false }

// NewHybridVKFromSpec builds a HybridVKEngine from a HybridSpec.
func NewHybridVKFromSpec(spec *HybridSpec) (*HybridVKEngine, error) {
	_ = spec
	return nil, ErrNotAvailable
}

type hybridVKEngine struct {
	pos         int
	maxSeq      int
	adapterName string
}

func (e *hybridVKEngine) release() {}
func (e *hybridVKEngine) resetState() error {
	return ErrNotAvailable
}
func (e *hybridVKEngine) appendTokens([]uint32) ([]float32, error) {
	return nil, ErrNotAvailable
}
func (e *hybridVKEngine) prefillSample([]uint32) (uint32, error) { return 0, ErrNotAvailable }
func (e *hybridVKEngine) stepTokenSample(uint32) (uint32, error) { return 0, ErrNotAvailable }
func (e *hybridVKEngine) decodeChunkSample(int) ([]uint32, error) {
	return nil, ErrNotAvailable
}
func (e *hybridVKEngine) estimateVRAM() uint64 { return 0 }
