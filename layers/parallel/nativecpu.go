package parallel

import (
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
)

// ForwardNativeCPU uses welvet_parallel_forward_f32 for Dense NewConfigured packs;
// otherwise Go CPU tiled (children still inherit Exec.Backend == NativeCPU).
func ForwardNativeCPU[T core.Numeric](l *Layer, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	pack, meta, err := packDenseParallel(l)
	if err != nil {
		return ForwardCPUTiled(l, input)
	}
	batch, seq, err := parallelLayout(l, input)
	if err != nil {
		return ForwardCPUTiled(l, input)
	}
	outDim := l.Cfg.OutDim()
	if outDim <= 0 {
		return ForwardCPUTiled(l, input)
	}
	nx := batch * seq * l.Cfg.Dim
	ny := batch * seq * outDim
	xF := core.SliceAsFloat32(input.Data)
	preF := make([]float32, nx)
	postF := make([]float32, ny)
	if err := nativecpp.ParallelForwardF32(
		int32(batch), int32(seq), int32(l.Cfg.Dim), int32(l.Cfg.OutFeat), int32(l.Cfg.Branches),
		meta.combine, int32(l.Cfg.SparseK), meta.dtype, nativecpp.CABBackendCPU, l.Cfg.disagreeBeta(),
		pack, xF, preF, postF,
	); err != nil {
		return ForwardCPUTiled(l, input)
	}
	pre = reshapeParallel[T](preF, batch, seq, l.Cfg.Dim, l.Cfg.SeqLen > 0)
	post = reshapeParallel[T](postF, batch, seq, outDim, l.Cfg.SeqLen > 0)
	return pre, post, nil
}

// BackwardNativeCPU uses welvet_parallel_normalbp_f32 (use_target=0, lr=0).
func BackwardNativeCPU[T core.Numeric](l *Layer, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = pre
	pack, meta, err := packDenseParallel(l)
	if err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	batch, seq, err := parallelLayout(l, input)
	if err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	outDim := l.Cfg.OutDim()
	if outDim <= 0 {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	nx := batch * seq * l.Cfg.Dim
	ny := batch * seq * outDim
	nw := len(pack)
	xF := core.SliceAsFloat32(input.Data)
	gF := core.SliceAsFloat32(gradOut.Data)
	preF := make([]float32, nx)
	postF := make([]float32, ny)
	dX := make([]float32, nx)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	if _, err := nativecpp.ParallelNormalBPF32(
		int32(batch), int32(seq), int32(l.Cfg.Dim), int32(l.Cfg.OutFeat), int32(l.Cfg.Branches),
		meta.combine, int32(l.Cfg.SparseK), meta.dtype, nativecpp.CABBackendCPU, l.Cfg.disagreeBeta(),
		pack, xF, gF, false, 0,
		preF, postF, dX, dW, wAfter,
	); err != nil {
		return BackwardCPUTiled(l, gradOut, input, pre)
	}
	gradIn = reshapeParallel[T](dX, batch, seq, l.Cfg.Dim, l.Cfg.SeqLen > 0)
	gradW = core.NewTensor[T](nw)
	core.SliceFromFloat32(dW, gradW.Data)
	return gradIn, gradW, nil
}

func reshapeParallel[T core.Numeric](flat []float32, batch, seq, feat int, hasSeq bool) *core.Tensor[T] {
	var t *core.Tensor[T]
	if hasSeq {
		t = core.NewTensor[T](batch, seq, feat)
	} else {
		t = core.NewTensor[T](batch, feat)
	}
	core.SliceFromFloat32(flat, t.Data)
	return t
}

// ApplyExecConfig resolves a nativecpp.ExecConfig onto this layer's Exec and syncs branches.
func (l *Layer) ApplyExecConfig(cfg nativecpp.ExecConfig, volume, sector, layerKey string) {
	if l == nil {
		return
	}
	nativecpp.ApplyExecConfig(cfg, volume, sector, layerKey, &l.Exec)
	l.SyncBranchExec()
}

// SetBackend sets Exec.Backend and syncs branches (convenience for tests / planners).
func (l *Layer) SetBackend(b core.Backend) {
	if l == nil {
		return
	}
	l.Exec.Backend = b
	l.SyncBranchExec()
}

// SetBackend sets Exec.Backend and syncs stack children.
func (s *Stack) SetBackend(b core.Backend) {
	if s == nil {
		return
	}
	s.Exec.Backend = b
	s.SyncChildExec()
}

