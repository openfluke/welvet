package dense

import (
	"runtime"
	"sync"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/tiling"
	"github.com/openfluke/welvet/weights"
)

// ForwardCPUTiled — activation dtype T is chosen by the caller (not hardcoded).
func ForwardCPUTiled[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	_ = in
	pre = core.NewTensor[T](batch, out)
	post = core.NewTensor[T](batch, out)

	tile := tiling.CPUTile(l.Exec.TileSize)
	if l.Exec.TileSize <= 0 && l.Core.TileSize > 0 {
		tile = tiling.CPUTile(l.Core.TileSize)
	}
	multi := (l.Exec.MultiCore || l.Core.MultiCore) && tiling.PreferMultiCore(batch, out, tile)
	if multi {
		err = forwardTiledMC(l, input.Data, pre.Data, batch, out, tile)
	} else {
		err = forwardSerial(l, input.Data, pre.Data, batch, out)
	}
	if err != nil {
		return nil, nil, err
	}
	applyBiasAct(pre.Data, post.Data, l.Weights.Bias, l.Core.Activation, batch, out)
	return pre, post, nil
}

func forwardSerial[T core.Numeric](l *Layer, x, y []T, batch, out int) error {
	in := l.Core.InputHeight
	row := make([]T, out)
	for b := 0; b < batch; b++ {
		clear(row)
		if err := weights.MatVec(l.Weights, x[b*in:(b+1)*in], row); err != nil {
			return err
		}
		copy(y[b*out:(b+1)*out], row)
	}
	return nil
}

func forwardTiledMC[T core.Numeric](l *Layer, x, y []T, batch, out, tile int) error {
	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error
	sem := make(chan struct{}, workers)
	in := l.Core.InputHeight
	for b := 0; b < batch; b++ {
		sem <- struct{}{}
		wg.Add(1)
		go func(b int) {
			defer func() { <-sem; wg.Done() }()
			row := make([]T, out)
			if err := weights.MatVec(l.Weights, x[b*in:(b+1)*in], row); err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
				return
			}
			copy(y[b*out:(b+1)*out], row)
		}(b)
	}
	wg.Wait()
	_ = tile
	return firstErr
}

// BackwardCPUTiled — grads and activations share dtype T.
func BackwardCPUTiled[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	batch, in, out, err := dims(l, input)
	if err != nil {
		return nil, nil, err
	}
	dPre := make([]T, batch*out)
	act := l.Core.Activation
	for i := 0; i < batch*out; i++ {
		g := core.AsFloat64(gradOut.Data[i]) * core.AsFloat64(core.ActivateDeriv(pre.Data[i], act))
		dPre[i] = core.FromFloat64[T](g)
	}
	gradIn = core.NewTensor[T](batch, in)
	gradW = core.NewTensor[T](out, in)

	for b := 0; b < batch; b++ {
		gy := dPre[b*out : (b+1)*out]
		gx := gradIn.Data[b*in : (b+1)*in]
		if err := weights.MatVecT(l.Weights, gy, gx); err != nil {
			return nil, nil, err
		}
		xRow := input.Data[b*in : (b+1)*in]
		for o := 0; o < out; o++ {
			g := core.AsFloat64(gy[o])
			dw := gradW.Data[o*in : (o+1)*in]
			for i := 0; i < in; i++ {
				dw[i] = core.FromFloat64[T](core.AsFloat64(dw[i]) + g*core.AsFloat64(xRow[i]))
			}
		}
	}
	return gradIn, gradW, nil
}

// PermutationOK reports whether (dtype, format, backend) is a valid dense cell.
func PermutationOK(dt core.DType, format quant.Format, backend core.Backend) bool {
	if !quant.Supported(format) {
		return false
	}
	switch backend {
	case core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU:
		return true
	default:
		return false
	}
}

// AllPermutations lists the dense coverage matrix.
func AllPermutations() (out []struct {
	DType   core.DType
	Format  quant.Format
	Backend core.Backend
}) {
	backends := []core.Backend{core.BackendCPUTiled, core.BackendSIMD, core.BackendWebGPU}
	for _, dt := range core.AllDTypes {
		for _, f := range quant.AllFormats {
			for _, b := range backends {
				if PermutationOK(dt, f, b) {
					out = append(out, struct {
						DType   core.DType
						Format  quant.Format
						Backend core.Backend
					}{dt, f, b})
				}
			}
		}
	}
	return out
}
