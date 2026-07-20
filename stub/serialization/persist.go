package serialization

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/weights"
)

// SerializeGrid converts a Grid into indented JSON (SerializeNetwork alias).
func SerializeGrid(g *architecture.Grid) ([]byte, error) {
	spec, err := BuildSpec(g)
	if err != nil {
		return nil, err
	}
	return json.MarshalIndent(spec, "", "  ")
}

// SerializeNetwork is the loom alias.
func SerializeNetwork(g *architecture.Grid) ([]byte, error) {
	return SerializeGrid(g)
}

// BuildSpec walks the grid and snapshots all Ops as storage-truth stores.
func BuildSpec(g *architecture.Grid) (*NetworkSpec, error) {
	if g == nil {
		return nil, fmt.Errorf("serialization: nil grid")
	}
	spec := &NetworkSpec{
		ID:            "network",
		Depth:         g.Depth,
		Rows:          g.Rows,
		Cols:          g.Cols,
		LayersPerCell: g.LayersPerCell,
		Layers:        make([]LayerSpec, 0, g.StackLayerCount()),
	}
	for _, c := range g.HopOrder() {
		cell := g.At(c.Z, c.Y, c.X, c.L)
		if cell == nil || cell.Op == nil {
			continue
		}
		ls, err := encodeCell(c, cell)
		if err != nil {
			return nil, fmt.Errorf("cell %v: %w", c, err)
		}
		spec.Layers = append(spec.Layers, ls)
	}
	return spec, nil
}

// DeserializeGrid rebuilds a Grid from SerializeGrid JSON.
func DeserializeGrid(data []byte) (*architecture.Grid, error) {
	var spec NetworkSpec
	if err := json.Unmarshal(data, &spec); err != nil {
		return nil, err
	}
	return GridFromSpec(&spec)
}

// DeserializeNetwork is the loom alias.
func DeserializeNetwork(data []byte) (*architecture.Grid, error) {
	return DeserializeGrid(data)
}

// GridFromSpec materializes cells from a NetworkSpec.
func GridFromSpec(spec *NetworkSpec) (*architecture.Grid, error) {
	if spec == nil {
		return nil, fmt.Errorf("serialization: nil spec")
	}
	g := architecture.NewGrid(spec.Depth, spec.Rows, spec.Cols, spec.LayersPerCell)
	for _, ls := range spec.Layers {
		if err := placeFromSpec(g, ls); err != nil {
			return nil, fmt.Errorf("layer (%d,%d,%d,%d): %w", ls.Z, ls.Y, ls.X, ls.L, err)
		}
	}
	return g, nil
}

// EncodeNativeStore packs FormatNone storage-truth bytes from a store.
func EncodeNativeStore(s *weights.Store) (raw []byte, scale float32, err error) {
	snap, err := weights.TakeSnapshot(s)
	if err != nil {
		return nil, 0, err
	}
	return snap.Raw, snap.Scale, nil
}

// EncodeNativeWeightsRaw encodes float32 LE (FormatNone float32 path).
func EncodeNativeWeightsRaw(data []float32) []byte {
	return weights.EncodeF32LE(data)
}

// EncodeF32LE writes little-endian float32 bytes.
func EncodeF32LE(w []float32) []byte {
	return weights.EncodeF32LE(w)
}

// DecodeNativeF32 reads LE float32 weights.
func DecodeNativeF32(raw []byte, n int) ([]float32, error) {
	return weights.DecodeF32LE(raw, n)
}

// NativeWeightsEqual compares two f32 slices by on-disk encoding.
func NativeWeightsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	return string(EncodeF32LE(a)) == string(EncodeF32LE(b))
}

// StripWeights clears weight payloads from a spec (ENTITY topology header).
func StripWeights(spec *NetworkSpec) {
	if spec == nil {
		return
	}
	for i := range spec.Layers {
		stripLayer(&spec.Layers[i])
	}
}

func stripLayer(ls *LayerSpec) {
	ls.Weights = ""
	ls.Native = false
	ls.Scale = 0
	ls.Stores = nil
	ls.Extras = nil
}

// EncodeStoreBlobB64 is a test helper: snapshot → base64.
func EncodeStoreBlobB64(s *weights.Store) (string, error) {
	b, err := encodeStore("w", s)
	if err != nil {
		return "", err
	}
	return b.Data, nil
}

// MustB64 decodes base64 or panics (tests).
func MustB64(s string) []byte {
	raw, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return raw
}
