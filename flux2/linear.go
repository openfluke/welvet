package flux2

import (
	"fmt"
	"unsafe"

	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/webgpu"
)

// Linear is either a BinaryG128 quantized matvec or a dense [out,in] float32 weight.
type Linear struct {
	Out, In int
	Blob    *quant.Blob // BinaryG128 (or other packed); preferred when non-nil
	Weight  []float32   // dense row-major [Out][In]; used when Blob == nil
	Bias    []float32   // optional length Out
	Name    string
	// UseGPU routes BinaryG128 GEMV through resident WebGPU (set by Model.SyncGPU).
	UseGPU bool
}

// NewDenseLinear builds a dense linear from weight [out*in] row-major.
func NewDenseLinear(out, in int, weight []float32, bias []float32, name string) (*Linear, error) {
	if len(weight) < out*in {
		return nil, fmt.Errorf("NewDenseLinear %s: weight short %d need %d", name, len(weight), out*in)
	}
	if bias != nil && len(bias) < out {
		return nil, fmt.Errorf("NewDenseLinear %s: bias short", name)
	}
	return &Linear{Out: out, In: in, Weight: weight[:out*in], Bias: bias, Name: name}, nil
}

// NewBlobLinear wraps a quantized blob (rows=out, cols=in).
func NewBlobLinear(b *quant.Blob, bias []float32, name string) (*Linear, error) {
	if b == nil {
		return nil, fmt.Errorf("NewBlobLinear %s: nil blob", name)
	}
	if bias != nil && len(bias) < b.Rows {
		return nil, fmt.Errorf("NewBlobLinear %s: bias short", name)
	}
	return &Linear{Out: b.Rows, In: b.Cols, Blob: b, Bias: bias, Name: name}, nil
}

// MatVec computes y = W @ x (+ bias). y is overwritten; len(y) >= Out, len(x) >= In.
func (l *Linear) MatVec(x, y []float32) error {
	if l == nil {
		return fmt.Errorf("Linear.MatVec: nil")
	}
	if len(x) < l.In || len(y) < l.Out {
		return fmt.Errorf("Linear.MatVec %s: shape x=%d need %d, y=%d need %d", l.Name, len(x), l.In, len(y), l.Out)
	}
	if l.Blob != nil {
		if err := l.matVecBlob(x, y, 1); err != nil {
			return fmt.Errorf("Linear.MatVec %s: %w", l.Name, err)
		}
	} else {
		w := l.Weight
		in := l.In
		for i := 0; i < l.Out; i++ {
			var acc float32
			row := w[i*in : i*in+in]
			for j := 0; j < in; j++ {
				acc += row[j] * x[j]
			}
			y[i] = acc
		}
	}
	if l.Bias != nil {
		for i := 0; i < l.Out; i++ {
			y[i] += l.Bias[i]
		}
	}
	return nil
}

// MatMulSeq applies W to each of seq tokens in x [seq*in] → y [seq*out].
// On GPU, this is one batched BinaryG128 / Affine4 GEMV (batch=seq).
func (l *Linear) MatMulSeq(x, y []float32, seq int) error {
	if seq <= 0 {
		return nil
	}
	if seq*l.In > len(x) || seq*l.Out > len(y) {
		return fmt.Errorf("Linear.MatMulSeq %s: buffer short seq=%d", l.Name, seq)
	}
	if l.Blob != nil && l.UseGPU {
		if err := l.matVecBlob(x, y, seq); err != nil {
			return fmt.Errorf("Linear.MatMulSeq %s: %w", l.Name, err)
		}
		if l.Bias != nil {
			for s := 0; s < seq; s++ {
				base := s * l.Out
				for i := 0; i < l.Out; i++ {
					y[base+i] += l.Bias[i]
				}
			}
		}
		return nil
	}
	for s := 0; s < seq; s++ {
		if err := l.MatVec(x[s*l.In:(s+1)*l.In], y[s*l.Out:(s+1)*l.Out]); err != nil {
			return err
		}
	}
	return nil
}

func (l *Linear) matVecBlob(x, y []float32, batch int) error {
	b := l.Blob
	if l.UseGPU && webgpu.Available() {
		key := webgpu.BlobKey(unsafe.Pointer(b))
		if quant.IsBinaryG128(b) {
			if webgpu.HasBinaryWeight(key) {
				return webgpu.DenseGEMVBinaryResident(key, nil, nil, x, y, batch, l.In, l.Out, true)
			}
			if len(b.Raw) >= minBinaryGPUBytes {
				scales, words, g128, ok := dense.BinaryBlobStaging(b)
				if ok {
					if err := webgpu.DenseGEMVBinaryResident(key, scales, words, x, y, batch, l.In, l.Out, g128); err == nil {
						return nil
					}
				}
			}
		}
		if quant.IsAffinePacked(b) {
			if webgpu.HasAffineWeight(key) {
				group := b.BlockWeights
				if group <= 0 {
					group = quant.AffineG64Group
				}
				return webgpu.DenseGEMVAffineResident(key, nil, nil, nil, x, y, batch, l.In, l.Out, group)
			}
			scales, biases, words, group, ok := dense.AffineBlobStaging(b)
			if ok && len(b.Raw) >= 64<<10 {
				if err := webgpu.DenseGEMVAffineResident(key, scales, biases, words, x, y, batch, l.In, l.Out, group); err == nil {
					return nil
				}
			}
		}
	}
	if batch != 1 {
		for s := 0; s < batch; s++ {
			if err := quant.MatVec(b, x[s*l.In:(s+1)*l.In], y[s*l.Out:(s+1)*l.Out]); err != nil {
				return err
			}
		}
		return nil
	}
	return quant.MatVec(b, x, y)
}
