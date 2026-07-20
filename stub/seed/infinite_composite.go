package seed

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/quant"
)

// partF32 materializes a leaf "dense" part manifest (He baseline + overrides) to flat f32.
func partF32(m *InfiniteLayer, name string) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("seed: missing part %q", name)
	}
	l, err := BuildDenseFromInfinite(m)
	if err != nil {
		return nil, fmt.Errorf("seed: part %q: %w", name, err)
	}
	return l.Weights.FlattenF32()
}

func manifestFromDenseNamed(op *dense.Layer, layerSeed uint64, path string) (*InfiniteLayer, error) {
	seed := DeriveLayer(layerSeed, 0, path)
	return ManifestFromDense(op, seed)
}

// ManifestFromMHA builds an infinite manifest for Q/K/V/O projections.
func ManifestFromMHA(op *mha.Layer, layerSeed uint64) (*InfiniteLayer, error) {
	if op == nil {
		return nil, fmt.Errorf("seed: nil mha")
	}
	q, err := manifestFromDenseNamed(op.Q, layerSeed, "mha.q")
	if err != nil {
		return nil, err
	}
	k, err := manifestFromDenseNamed(op.K, layerSeed, "mha.k")
	if err != nil {
		return nil, err
	}
	v, err := manifestFromDenseNamed(op.V, layerSeed, "mha.v")
	if err != nil {
		return nil, err
	}
	o, err := manifestFromDenseNamed(op.O, layerSeed, "mha.o")
	if err != nil {
		return nil, err
	}
	return &InfiniteLayer{
		Format:    infiniteLayerFormat,
		Kind:      "mha",
		DType:     "Float32",
		LayerSeed: layerSeed,
		Parts:     map[string]*InfiniteLayer{"q": q, "k": k, "v": v, "o": o},
	}, nil
}

// BuildMHAFromInfinite materializes MHA from a manifest; cfg supplies the
// head-count / masking geometry that a raw weight matrix cannot recover.
func BuildMHAFromInfinite(m *InfiniteLayer, cfg mha.Config) (*mha.Layer, error) {
	if m == nil || m.Kind != "mha" {
		return nil, fmt.Errorf("seed: want kind=mha, got %v", kindOf(m))
	}
	qW, err := partF32(m.Parts["q"], "q")
	if err != nil {
		return nil, err
	}
	kW, err := partF32(m.Parts["k"], "k")
	if err != nil {
		return nil, err
	}
	vW, err := partF32(m.Parts["v"], "v")
	if err != nil {
		return nil, err
	}
	oW, err := partF32(m.Parts["o"], "o")
	if err != nil {
		return nil, err
	}
	return mha.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, qW, kW, vW, oW)
}

// ManifestFromSwiGLU builds an infinite manifest for Gate/Up/Down projections.
func ManifestFromSwiGLU(op *swiglu.Layer, layerSeed uint64) (*InfiniteLayer, error) {
	if op == nil {
		return nil, fmt.Errorf("seed: nil swiglu")
	}
	gate, err := manifestFromDenseNamed(op.Gate, layerSeed, "swiglu.gate")
	if err != nil {
		return nil, err
	}
	up, err := manifestFromDenseNamed(op.Up, layerSeed, "swiglu.up")
	if err != nil {
		return nil, err
	}
	down, err := manifestFromDenseNamed(op.Down, layerSeed, "swiglu.down")
	if err != nil {
		return nil, err
	}
	return &InfiniteLayer{
		Format:    infiniteLayerFormat,
		Kind:      "swiglu",
		DType:     "Float32",
		LayerSeed: layerSeed,
		Parts:     map[string]*InfiniteLayer{"gate": gate, "up": up, "down": down},
	}, nil
}

// BuildSwiGLUFromInfinite materializes SwiGLU from a manifest.
func BuildSwiGLUFromInfinite(m *InfiniteLayer, cfg swiglu.Config) (*swiglu.Layer, error) {
	if m == nil || m.Kind != "swiglu" {
		return nil, fmt.Errorf("seed: want kind=swiglu, got %v", kindOf(m))
	}
	gateW, err := partF32(m.Parts["gate"], "gate")
	if err != nil {
		return nil, err
	}
	upW, err := partF32(m.Parts["up"], "up")
	if err != nil {
		return nil, err
	}
	downW, err := partF32(m.Parts["down"], "down")
	if err != nil {
		return nil, err
	}
	return swiglu.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, gateW, upW, downW)
}

// ManifestFromCNN1 builds an infinite manifest for the Conv1d projection.
func ManifestFromCNN1(op *cnn1.Layer, layerSeed uint64) (*InfiniteLayer, error) {
	if op == nil {
		return nil, fmt.Errorf("seed: nil cnn1")
	}
	proj, err := manifestFromDenseNamed(op.Proj, layerSeed, "cnn1.proj")
	if err != nil {
		return nil, err
	}
	return &InfiniteLayer{
		Format:    infiniteLayerFormat,
		Kind:      "cnn1",
		DType:     "Float32",
		LayerSeed: layerSeed,
		Parts:     map[string]*InfiniteLayer{"proj": proj},
	}, nil
}

// BuildCNN1FromInfinite materializes CNN1 from a manifest.
func BuildCNN1FromInfinite(m *InfiniteLayer, cfg cnn1.Config) (*cnn1.Layer, error) {
	if m == nil || m.Kind != "cnn1" {
		return nil, fmt.Errorf("seed: want kind=cnn1, got %v", kindOf(m))
	}
	w, err := partF32(m.Parts["proj"], "proj")
	if err != nil {
		return nil, err
	}
	return cnn1.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, w)
}

// ManifestFromCNN2 builds an infinite manifest for the Conv2d projection.
func ManifestFromCNN2(op *cnn2.Layer, layerSeed uint64) (*InfiniteLayer, error) {
	if op == nil {
		return nil, fmt.Errorf("seed: nil cnn2")
	}
	proj, err := manifestFromDenseNamed(op.Proj, layerSeed, "cnn2.proj")
	if err != nil {
		return nil, err
	}
	return &InfiniteLayer{
		Format:    infiniteLayerFormat,
		Kind:      "cnn2",
		DType:     "Float32",
		LayerSeed: layerSeed,
		Parts:     map[string]*InfiniteLayer{"proj": proj},
	}, nil
}

// BuildCNN2FromInfinite materializes CNN2 from a manifest.
func BuildCNN2FromInfinite(m *InfiniteLayer, cfg cnn2.Config) (*cnn2.Layer, error) {
	if m == nil || m.Kind != "cnn2" {
		return nil, fmt.Errorf("seed: want kind=cnn2, got %v", kindOf(m))
	}
	w, err := partF32(m.Parts["proj"], "proj")
	if err != nil {
		return nil, err
	}
	return cnn2.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, w)
}

// ManifestFromCNN3 builds an infinite manifest for the Conv3d projection.
func ManifestFromCNN3(op *cnn3.Layer, layerSeed uint64) (*InfiniteLayer, error) {
	if op == nil {
		return nil, fmt.Errorf("seed: nil cnn3")
	}
	proj, err := manifestFromDenseNamed(op.Proj, layerSeed, "cnn3.proj")
	if err != nil {
		return nil, err
	}
	return &InfiniteLayer{
		Format:    infiniteLayerFormat,
		Kind:      "cnn3",
		DType:     "Float32",
		LayerSeed: layerSeed,
		Parts:     map[string]*InfiniteLayer{"proj": proj},
	}, nil
}

// BuildCNN3FromInfinite materializes CNN3 from a manifest.
func BuildCNN3FromInfinite(m *InfiniteLayer, cfg cnn3.Config) (*cnn3.Layer, error) {
	if m == nil || m.Kind != "cnn3" {
		return nil, fmt.Errorf("seed: want kind=cnn3, got %v", kindOf(m))
	}
	w, err := partF32(m.Parts["proj"], "proj")
	if err != nil {
		return nil, err
	}
	return cnn3.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, w)
}

func kindOf(m *InfiniteLayer) string {
	if m == nil {
		return "<nil>"
	}
	return m.Kind
}
