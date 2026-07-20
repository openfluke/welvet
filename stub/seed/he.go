package seed

import (
	"encoding/binary"
	"hash/fnv"
	"math"

	"github.com/openfluke/welvet/weights"
)

// InitFloat32He fills weights with He-init from seed (bit-compatible with loom).
func InitFloat32He(weights []float32, inputSize int, seed uint64) {
	if len(weights) == 0 {
		return
	}
	if inputSize <= 0 {
		inputSize = 1
	}
	rng := New(seed)
	stddev := float32(math.Sqrt(2.0 / float64(inputSize)))
	for i := range weights {
		weights[i] = float32(rng.NormFloat64()) * stddev
	}
}

// InitFloat32HeSeeded is the loom alias.
func InitFloat32HeSeeded(w []float32, inputSize int, seed uint64) {
	InitFloat32He(w, inputSize, seed)
}

// InitStoreHe He-inits a Store master via Flatten/SetFromF32.
func InitStoreHe(s *weights.Store, inputSize int, seed uint64) error {
	if s == nil {
		return nil
	}
	n := s.Rows * s.Cols
	w := make([]float32, n)
	InitFloat32He(w, inputSize, seed)
	return s.SetFromF32(w)
}

// FingerprintF32 is FNV-1a over little-endian float32 bits.
func FingerprintF32(w []float32) uint64 {
	h := fnv.New64a()
	var buf [4]byte
	for _, v := range w {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

// StoreFingerprint hashes FlattenF32 of a store.
func StoreFingerprint(s *weights.Store) uint64 {
	if s == nil {
		return 0
	}
	w, err := s.FlattenF32()
	if err != nil || w == nil {
		return 0
	}
	return FingerprintF32(w)
}
