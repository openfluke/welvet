package gdn

import (
	"fmt"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/seqmix"
	"github.com/openfluke/welvet/quant"
)

// Validate checks Config geometry.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("gdn: nil config")
	}
	if c.HiddenSize <= 0 || c.NumKeyHeads <= 0 || c.NumValueHeads <= 0 {
		return fmt.Errorf("gdn: need positive HiddenSize/NumKeyHeads/NumValueHeads")
	}
	if c.KeyHeadDim <= 0 || c.ValueHeadDim <= 0 {
		return fmt.Errorf("gdn: need positive head dims")
	}
	if c.NumValueHeads%c.NumKeyHeads != 0 {
		return fmt.Errorf("gdn: NumValueHeads must divide by NumKeyHeads")
	}
	if c.ConvKernel < 1 {
		c.ConvKernel = 4
	}
	if c.Eps <= 0 {
		c.Eps = 1e-6
	}
	return nil
}

// Kind returns seqmix.KindLinearAttn.
func (l *Layer) Kind() seqmix.Kind { return seqmix.KindLinearAttn }

// CoreMeta returns a volumetric Layer header for Place.
func (l *Layer) CoreMeta() core.Layer {
	h := 0
	if l != nil {
		h = l.Cfg.HiddenSize
	}
	return core.Layer{
		Type:         core.LayerGDN,
		DType:        core.DTypeFloat32,
		Activation:   core.ActivationLinear,
		InputHeight:  h,
		OutputHeight: h,
		TileSize:     32,
		MultiCore:    true,
	}
}

// NewConfigured builds GDN from FormatNone float32 projection matrices.
// inQKV [convDim×H], inZ [valDim×H], inB/inA [numV×H], out [H×valDim].
func NewConfigured(cfg Config, inQKV, inZ, inB, inA, outW, convW, aLog, dtBias, gamma []float32) (*Layer, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	h := cfg.HiddenSize
	cd, vd, nh := cfg.convDim(), cfg.valueDim(), cfg.NumValueHeads
	pack := func(rows, cols int, w []float32, name string) (*quant.Blob, error) {
		if w == nil {
			w = make([]float32, rows*cols)
		}
		if len(w) < rows*cols {
			return nil, fmt.Errorf("gdn: %s short", name)
		}
		return quant.Pack(quant.FormatNone, w[:rows*cols], rows, cols)
	}
	bqkv, err := pack(cd, h, inQKV, "InQKV")
	if err != nil {
		return nil, err
	}
	bz, err := pack(vd, h, inZ, "InZ")
	if err != nil {
		return nil, err
	}
	bb, err := pack(nh, h, inB, "InB")
	if err != nil {
		return nil, err
	}
	ba, err := pack(nh, h, inA, "InA")
	if err != nil {
		return nil, err
	}
	bo, err := pack(h, vd, outW, "Out")
	if err != nil {
		return nil, err
	}
	if convW == nil {
		convW = make([]float32, cd*cfg.ConvKernel)
	}
	if aLog == nil {
		aLog = make([]float32, nh)
		for i := range aLog {
			aLog[i] = -1
		}
	}
	if dtBias == nil {
		dtBias = make([]float32, nh)
	}
	if gamma == nil {
		gamma = make([]float32, cfg.ValueHeadDim)
		for i := range gamma {
			gamma[i] = 1
		}
	}
	l := &Layer{
		Cfg:        cfg,
		InQKV:      bqkv,
		InZ:        bz,
		InB:        bb,
		InA:        ba,
		Out:        bo,
		ConvWeight: append([]float32(nil), convW...),
		ALog:       append([]float32(nil), aLog...),
		DtBias:     append([]float32(nil), dtBias...),
		NormGamma:  append([]float32(nil), gamma...),
		Exec: core.ExecConfig{
			Backend:   core.BackendCPUTiled,
			MultiCore: true,
			TileSize:  32,
		},
	}
	l.syncExec()
	l.Reset()
	return l, nil
}

// New is NewConfigured with zeros.
func New(cfg Config) (*Layer, error) {
	return NewConfigured(cfg, nil, nil, nil, nil, nil, nil, nil, nil, nil)
}

// Pack re-packs projection blobs to FormatNone or BinaryPacked (unpack → Pack).
func (l *Layer) Pack(format quant.Format) error {
	if l == nil {
		return fmt.Errorf("gdn: nil")
	}
	if format != quant.FormatNone && format != quant.FormatBinaryPacked {
		return fmt.Errorf("gdn: Pack only FormatNone or BinaryPacked in v0, got %v", format)
	}
	repack := func(bp **quant.Blob) error {
		b := *bp
		if b == nil {
			return fmt.Errorf("gdn: nil blob")
		}
		if b.Format == format {
			return nil
		}
		f32, err := quant.Unpack(b)
		if err != nil {
			return err
		}
		nb, err := quant.Pack(format, f32, b.Rows, b.Cols)
		if err != nil {
			return err
		}
		*bp = nb
		return nil
	}
	if err := repack(&l.InQKV); err != nil {
		return err
	}
	if err := repack(&l.InZ); err != nil {
		return err
	}
	if err := repack(&l.InB); err != nil {
		return err
	}
	if err := repack(&l.InA); err != nil {
		return err
	}
	return repack(&l.Out)
}

// Place binds GDN onto the grid and copies grid Exec onto the layer.
func Place(g *architecture.Grid, z, y, x, lidx int, layer *Layer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("gdn: Place nil")
	}
	layer.Exec = g.Exec
	layer.syncExec()
	meta := layer.CoreMeta()
	meta.Z, meta.Y, meta.X, meta.L = z, y, x, lidx
	return g.BindOp(z, y, x, lidx, meta, layer)
}

// Forward dispatches on Exec.Backend (host decode loop; SIMD/WebGPU gated).
func Forward[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if l == nil || input == nil {
		return nil, nil, fmt.Errorf("gdn: nil")
	}
	l.syncExec()
	switch l.Exec.Backend {
	case core.BackendSIMD:
		return ForwardSIMD(l, input)
	case core.BackendWebGPU:
		return ForwardWebGPU(l, input)
	default:
		return forwardHost(l, input)
	}
}

// forwardHost runs [batch, seq, hidden] by looping ForwardDecode (resets state once).
func forwardHost[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if len(input.Shape) != 3 || input.Shape[2] != l.Cfg.HiddenSize {
		return nil, nil, fmt.Errorf("gdn: need [B,T,%d], got %v", l.Cfg.HiddenSize, input.Shape)
	}
	b, t, h := input.Shape[0], input.Shape[1], input.Shape[2]
	out := core.NewTensor[T](b, t, h)
	for bi := 0; bi < b; bi++ {
		l.Reset()
		for ti := 0; ti < t; ti++ {
			x := make([]float32, h)
			y := make([]float32, h)
			base := (bi*t + ti) * h
			for i := 0; i < h; i++ {
				x[i] = float32(core.AsFloat64(input.Data[base+i]))
			}
			if err := l.ForwardDecode(x, y); err != nil {
				return nil, nil, err
			}
			for i := 0; i < h; i++ {
				out.Data[base+i] = core.FromFloat64[T](float64(y[i]))
			}
		}
	}
	pre = out.Clone()
	post = out
	return pre, post, nil
}

// PermutationOK — Float32 + FormatNone/BinaryPacked × CPU/SIMD/WebGPU.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	if dt != core.DTypeFloat32 {
		return false
	}
	if format != quant.FormatNone && format != quant.FormatBinaryPacked {
		return false
	}
	return backend == core.BackendCPUTiled || backend == core.BackendSIMD || backend == core.BackendWebGPU
}

// AllPermutations lists GDN's honest coverage matrix (Float32 × None/Binary × 3 backends).
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	for _, f := range []quant.Format{quant.FormatNone, quant.FormatBinaryPacked} {
		for _, be := range []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU} {
			out = append(out, struct {
				DType   core.DType
				Format  quant.Format
				Backend core.Backend
			}{core.DTypeFloat32, f, be})
		}
	}
	return out
}
