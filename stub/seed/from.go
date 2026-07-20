package seed

import (
	"encoding/binary"
	"hash/fnv"
	"math"
)

const goldenRatio = 0x9e3779b97f4a7c15

// From mixes arbitrary inputs into a single uint64 seed. The same inputs in
// the same order always yield the same seed.
//
// Supported types: string, []byte, bool, and all signed/unsigned integer types.
func From(parts ...any) uint64 {
	var h uint64 = goldenRatio
	for i, p := range parts {
		h = mix(h, uint64(i))
		h = mixValue(h, p)
	}
	return splitmix64(h)
}

func mix(h, v uint64) uint64 {
	h ^= v + goldenRatio + (h << 6) + (h >> 2)
	return h
}

func mixValue(h uint64, v any) uint64 {
	switch x := v.(type) {
	case string:
		return mixBytes(h, []byte(x))
	case []byte:
		return mixBytes(h, x)
	case bool:
		if x {
			return mix(h, 1)
		}
		return mix(h, 0)
	case int:
		return mix(h, uint64(x))
	case int8:
		return mix(h, uint64(uint8(x)))
	case int16:
		return mix(h, uint64(uint16(x)))
	case int32:
		return mix(h, uint64(uint32(x)))
	case int64:
		return mix(h, uint64(x))
	case uint:
		return mix(h, uint64(x))
	case uint8:
		return mix(h, uint64(x))
	case uint16:
		return mix(h, uint64(x))
	case uint32:
		return mix(h, uint64(x))
	case uint64:
		return mix(h, x)
	case float32:
		return mix(h, uint64(math.Float32bits(x)))
	case float64:
		return mix(h, math.Float64bits(x))
	default:
		panic("seed: unsupported seed input type")
	}
}

func mixBytes(h uint64, b []byte) uint64 {
	hasher := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], h)
	_, _ = hasher.Write(buf[:])
	_, _ = hasher.Write(b)
	return hasher.Sum64()
}

// SeedFrom is the loom/poly alias for From.
func SeedFrom(parts ...any) uint64 { return From(parts...) }

// DeriveLayer derives a per-layer weight seed from init seed, index, and path.
func DeriveLayer(initSeed uint64, layerIndex int, path string) uint64 {
	return From(initSeed, layerIndex, path)
}

// DeriveLayerSeed is the loom alias for DeriveLayer.
func DeriveLayerSeed(initSeed uint64, layerIndex int, path string) uint64 {
	return DeriveLayer(initSeed, layerIndex, path)
}
