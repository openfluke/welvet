package seed

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
)

const infiniteLayerFormat = "welvet-infinite-layer-v2"

// DefaultDenseWeightChunk is the tile size for matrix weight overrides [out, in].
var DefaultDenseWeightChunk = []int{8, 8}

// WeightChunkOverride is one weight tile that differs from He-init(layer_seed).
type WeightChunkOverride struct {
	At      []int  `json:"at"`
	Shape   []int  `json:"shape"`
	Payload []byte `json:"payload"`
}

// InfiniteLayer is a procedural layer as root seed + optional sparse diffs.
//
// Kind "dense" is a leaf (In/Out/Overrides describe one weight matrix).
// Composite kinds ("mha", "swiglu", "cnn1", "cnn2", "cnn3") carry no direct
// weight matrix of their own — Parts holds one leaf "dense" InfiniteLayer per
// projection (e.g. mha: q/k/v/o; swiglu: gate/up/down; cnnN: proj).
type InfiniteLayer struct {
	Format    string                    `json:"format"`
	Kind      string                    `json:"kind"`
	DType     string                    `json:"dtype"`
	LayerSeed uint64                    `json:"layer_seed"`
	WeightFP  uint64                    `json:"weight_fp"`
	ChunkSize []int                     `json:"chunk_size,omitempty"`
	Overrides []WeightChunkOverride     `json:"overrides,omitempty"`
	In        int                       `json:"in,omitempty"`
	Out       int                       `json:"out,omitempty"`
	Parts     map[string]*InfiniteLayer `json:"parts,omitempty"`
}

// IsComposite reports whether m carries Parts (mha/swiglu/cnnN) rather than a
// direct weight matrix (dense).
func (m *InfiniteLayer) IsComposite() bool {
	return m != nil && len(m.Parts) > 0
}

// OverrideCount returns sparse chunk count.
func (m *InfiniteLayer) OverrideCount() int {
	if m == nil {
		return 0
	}
	return len(m.Overrides)
}

// BuildDenseFromInfinite materializes a Dense layer from an infinite manifest.
func BuildDenseFromInfinite(m *InfiniteLayer) (*dense.Layer, error) {
	if m == nil || m.In <= 0 || m.Out <= 0 {
		return nil, fmt.Errorf("seed: infinite need In/Out")
	}
	w := make([]float32, m.In*m.Out)
	InitFloat32He(w, m.In, m.LayerSeed)
	if err := applyOverrides(w, m.Out, m.In, m.Overrides); err != nil {
		return nil, err
	}
	fp := FingerprintF32(w)
	if m.WeightFP != 0 && fp != m.WeightFP {
		return nil, fmt.Errorf("seed: weight_fp mismatch got 0x%x want 0x%x", fp, m.WeightFP)
	}
	return dense.NewConfigured(m.In, m.Out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
}

// ManifestFromDense builds an infinite manifest (He baseline + sparse diffs).
func ManifestFromDense(op *dense.Layer, layerSeed uint64) (*InfiniteLayer, error) {
	if op == nil || op.Weights == nil {
		return nil, fmt.Errorf("seed: nil dense")
	}
	out, in := op.Weights.Rows, op.Weights.Cols
	actual, err := op.Weights.FlattenF32()
	if err != nil {
		return nil, err
	}
	baseline := make([]float32, in*out)
	InitFloat32He(baseline, in, layerSeed)
	cs := append([]int(nil), DefaultDenseWeightChunk...)
	overrides, err := diffChunks(baseline, actual, out, in, cs)
	if err != nil {
		return nil, err
	}
	return &InfiniteLayer{
		Format:    infiniteLayerFormat,
		Kind:      "dense",
		DType:     "Float32",
		LayerSeed: layerSeed,
		WeightFP:  FingerprintF32(actual),
		ChunkSize: cs,
		Overrides: overrides,
		In:        in,
		Out:       out,
	}, nil
}

// MarshalInfinite JSON-encodes.
func MarshalInfinite(m *InfiniteLayer) ([]byte, error) {
	return json.MarshalIndent(m, "", "  ")
}

// ParseInfinite decodes JSON.
func ParseInfinite(data []byte) (*InfiniteLayer, error) {
	var m InfiniteLayer
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

func diffChunks(base, actual []float32, rows, cols int, chunk []int) ([]WeightChunkOverride, error) {
	chR, chC := chunk[0], chunk[1]
	var out []WeightChunkOverride
	for r0 := 0; r0 < rows; r0 += chR {
		for c0 := 0; c0 < cols; c0 += chC {
			rN := chR
			if r0+rN > rows {
				rN = rows - r0
			}
			cN := chC
			if c0+cN > cols {
				cN = cols - c0
			}
			diff := false
			tile := make([]float32, rN*cN)
			for rr := 0; rr < rN; rr++ {
				for cc := 0; cc < cN; cc++ {
					i := (r0+rr)*cols + (c0 + cc)
					tile[rr*cN+cc] = actual[i]
					if actual[i] != base[i] {
						diff = true
					}
				}
			}
			if !diff {
				continue
			}
			payload, err := compressF32(tile)
			if err != nil {
				return nil, err
			}
			out = append(out, WeightChunkOverride{
				At: []int{r0, c0}, Shape: []int{rN, cN}, Payload: payload,
			})
		}
	}
	return out, nil
}

func applyOverrides(w []float32, rows, cols int, overrides []WeightChunkOverride) error {
	for _, o := range overrides {
		if len(o.At) < 2 || len(o.Shape) < 2 {
			return fmt.Errorf("seed: bad override")
		}
		r0, c0 := o.At[0], o.At[1]
		rN, cN := o.Shape[0], o.Shape[1]
		tile, err := decompressF32(o.Payload, rN*cN)
		if err != nil {
			return err
		}
		for rr := 0; rr < rN; rr++ {
			for cc := 0; cc < cN; cc++ {
				w[(r0+rr)*cols+(c0+cc)] = tile[rr*cN+cc]
			}
		}
	}
	return nil
}

func compressF32(w []float32) ([]byte, error) {
	raw := make([]byte, len(w)*4)
	for i, v := range w {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	var buf bytes.Buffer
	zw, err := flate.NewWriter(&buf, flate.DefaultCompression)
	if err != nil {
		return nil, err
	}
	if _, err := zw.Write(raw); err != nil {
		return nil, err
	}
	if err := zw.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func decompressF32(payload []byte, n int) ([]float32, error) {
	zr := flate.NewReader(bytes.NewReader(payload))
	defer zr.Close()
	raw, err := io.ReadAll(zr)
	if err != nil {
		return nil, err
	}
	if len(raw) < n*4 {
		return nil, fmt.Errorf("seed: short payload")
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out, nil
}
