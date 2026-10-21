package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/weights"
)

// TrainNativeMSE runs C++ Parallel NormalBP (Dense NewConfigured pack) with lr>0.
// Convenience wrapper around ParallelTrainMSE(ModeNormalBP). Branch (+ gate) W_after
// stays pending until SyncWeightsFromNative.
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	return trainParallelNativeMSE(l, input, target, ModeNormalBP, lr)
}

// TrainStackNativeMSE runs C++ Stack TrainMSE ModeNormalBP for Dense NestedStack packs.
// Child W_after stays pending until SyncWeightsFromNative.
func TrainStackNativeMSE[T core.Numeric](s *Stack, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	return trainStackNativeMSE(s, input, target, ModeNormalBP, lr)
}

func useNativeTrainBackend(b core.Backend) bool {
	return nativecpp.NativeCPUAvailable() && b == core.BackendNativeCPU
}

// tryTrainStackNativeMSE routes TrainStackMSE to C ABI when BackendNativeCPU and the
// stack packs as Dense NestedStack (Dim→Dim). ok=false ⇒ caller uses Go path.
func tryTrainStackNativeMSE[T core.Numeric](s *Stack, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, ok bool, err error) {
	if s == nil || !useNativeTrainBackend(s.Exec.Backend) {
		return 0, false, nil
	}
	loss, err = trainStackNativeMSE(s, input, target, mode, lr)
	if err != nil {
		// Heterogeneous / non-Dim stacks cannot pack — fall back to Go.
		if isNativePackFallback(err) {
			return 0, false, nil
		}
		return 0, true, err
	}
	return loss, true, nil
}

// tryTrainParallelNativeMSE routes TrainMSE to C ABI when BackendNativeCPU and the
// layer packs as Dense NewConfigured. ok=false ⇒ caller uses Go path.
func tryTrainParallelNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, ok bool, err error) {
	if l == nil || !useNativeTrainBackend(l.Exec.Backend) {
		return 0, false, nil
	}
	loss, err = trainParallelNativeMSE(l, input, target, mode, lr)
	if err != nil {
		if isNativePackFallback(err) {
			return 0, false, nil
		}
		return 0, true, err
	}
	return loss, true, nil
}

type nativePackFallback struct{ msg string }

func (e nativePackFallback) Error() string { return e.msg }

func isNativePackFallback(err error) bool {
	_, ok := err.(nativePackFallback)
	return ok
}

func trainParallelNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, err error) {
	if l == nil || input == nil || target == nil {
		return 0, fmt.Errorf("parallel: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("parallel: TrainNativeMSE lr must be > 0")
	}
	pack, meta, err := packDenseParallel(l)
	if err != nil {
		return 0, err
	}
	batch, seq, err := parallelLayout(l, input)
	if err != nil {
		return 0, err
	}
	outDim := l.Cfg.OutDim()
	if outDim <= 0 {
		return 0, nativePackFallback{msg: "parallel: TrainNativeMSE need OutFeat>0"}
	}
	nx := batch * seq * l.Cfg.Dim
	ny := batch * seq * outDim
	nw := len(pack)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	if len(xF) < nx || len(tF) < ny {
		return 0, fmt.Errorf("parallel: TrainNativeMSE buffer short")
	}
	wAfter := make([]float32, nw)
	loss, err = nativecpp.ParallelTrainMSE(
		int32(mode), int32(batch), int32(seq), int32(l.Cfg.Dim), int32(l.Cfg.OutFeat), int32(l.Cfg.Branches),
		meta.combine, int32(l.Cfg.SparseK), meta.dtype, nativecpp.CABBackendCPU, l.Cfg.disagreeBeta(),
		pack, xF, tF, lr, wAfter,
	)
	if err != nil {
		return 0, err
	}
	off := 0
	for _, ch := range meta.dense {
		nW := ch.Weights.Rows * ch.Weights.Cols
		ch.EnsureNativeMirror().Capture(wAfter[off:off+nW], ch.Weights)
		off += nW
	}
	if l.Gate != nil {
		nW := l.Gate.Weights.Rows * l.Gate.Weights.Cols
		l.Gate.EnsureNativeMirror().Capture(wAfter[off:off+nW], l.Gate.Weights)
		off += nW
	}
	l.ensureMirror().Capture(wAfter, meta.stores...)
	return loss, nil
}

func trainStackNativeMSE[T core.Numeric](s *Stack, input, target *core.Tensor[T], mode TrainMode, lr float64) (loss float64, err error) {
	if s == nil || input == nil || target == nil {
		return 0, fmt.Errorf("parallel: TrainStackNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("parallel: TrainStackNativeMSE lr must be > 0")
	}
	pack, meta, err := packDenseNestedStack(s)
	if err != nil {
		return 0, err
	}
	batch, seq, err := stackLayout(s, input)
	if err != nil {
		return 0, err
	}
	n := batch * seq * meta.dim
	nw := len(pack)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	if len(xF) < n || len(tF) < n {
		return 0, fmt.Errorf("parallel: TrainStackNativeMSE buffer short")
	}
	wAfter := make([]float32, nw)
	loss, err = nativecpp.StackTrainMSE(
		int32(mode), int32(batch), int32(seq), int32(meta.dim), int32(len(meta.dense)),
		meta.childAct, meta.dtype, nativecpp.CABBackendCPU,
		pack, xF, tF, lr, wAfter,
	)
	if err != nil {
		return 0, err
	}
	off := 0
	for _, ch := range meta.dense {
		nW := ch.Weights.Rows * ch.Weights.Cols
		ch.EnsureNativeMirror().Capture(wAfter[off:off+nW], ch.Weights)
		off += nW
	}
	s.ensureMirror().Capture(wAfter, meta.stores...)
	return loss, nil
}

type parallelPackMeta struct {
	combine int32
	dtype   int32
	dense   []*dense.Layer
	stores  []*weights.Store
}

type stackPackMeta struct {
	dim      int
	childAct int32
	dtype    int32
	dense    []*dense.Layer
	stores   []*weights.Store
}

func packDenseParallel(l *Layer) ([]float32, parallelPackMeta, error) {
	var meta parallelPackMeta
	if l == nil {
		return nil, meta, nativePackFallback{msg: "parallel: nil"}
	}
	if l.Cfg.OutFeat <= 0 {
		return nil, meta, nativePackFallback{msg: "parallel: packed NormalBP needs OutFeat > 0 (NewConfigured Dense)"}
	}
	combine, ok := combineABI(l.Cfg.Combine)
	if !ok {
		return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: bad combine %q", l.Cfg.Combine)}
	}
	meta.combine = combine
	if !nativecpp.NativeCPUAvailable() {
		return nil, meta, fmt.Errorf("parallel: TrainNativeMSE needs libwelvet")
	}
	var parts [][]float32
	total := 0
	for i, b := range l.Branches {
		dl, ok := b.(*dense.Layer)
		if !ok || dl == nil || dl.Weights == nil {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: branch %d not Dense FormatNone", i)}
		}
		if dl.Weights.Format != quant.FormatNone {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: branch %d need FormatNone", i)}
		}
		w, err := dl.NativeMatrixF32()
		if err != nil {
			return nil, meta, err
		}
		want := l.Cfg.OutFeat * l.Cfg.Dim
		if len(w) != want {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: branch %d weight len %d want %d", i, len(w), want)}
		}
		parts = append(parts, w)
		meta.dense = append(meta.dense, dl)
		meta.stores = append(meta.stores, dl.Weights)
		total += len(w)
		if i == 0 {
			meta.dtype = int32(dl.Weights.DType)
		}
	}
	if l.Cfg.Combine == CombineFilter {
		if l.Gate == nil || l.Gate.Weights == nil || l.Gate.Weights.Format != quant.FormatNone {
			return nil, meta, nativePackFallback{msg: "parallel: filter needs FormatNone Gate"}
		}
		w, err := l.Gate.NativeMatrixF32()
		if err != nil {
			return nil, meta, err
		}
		want := l.Cfg.Branches * l.Cfg.Dim
		if len(w) != want {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: gate weight len %d want %d", len(w), want)}
		}
		parts = append(parts, w)
		meta.stores = append(meta.stores, l.Gate.Weights)
		total += len(w)
	}
	out := make([]float32, total)
	off := 0
	for _, p := range parts {
		copy(out[off:], p)
		off += len(p)
	}
	return out, meta, nil
}

// packDenseNestedStack packs Stack Dense children as welvet_stack_* NestedStack
// (each child Dim→Dim, shared activation). Matches C ABI make_nested_stack_dense.
func packDenseNestedStack(s *Stack) ([]float32, stackPackMeta, error) {
	var meta stackPackMeta
	if s == nil || len(s.Children) == 0 {
		return nil, meta, nativePackFallback{msg: "parallel: empty NestedStack"}
	}
	if !nativecpp.NativeCPUAvailable() {
		return nil, meta, fmt.Errorf("parallel: TrainStackNativeMSE needs libwelvet")
	}
	var parts [][]float32
	total := 0
	for i, ch := range s.Children {
		dl, ok := ch.(*dense.Layer)
		if !ok || dl == nil || dl.Weights == nil {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: stack child %d not Dense", i)}
		}
		if dl.Weights.Format != quant.FormatNone {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: stack child %d need FormatNone", i)}
		}
		inH, outH := dl.Core.InputHeight, dl.Core.OutputHeight
		if inH <= 0 || inH != outH {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: stack child %d need square Dim→Dim got %d→%d", i, inH, outH)}
		}
		if i == 0 {
			meta.dim = inH
			meta.childAct = int32(dl.Core.Activation)
			meta.dtype = int32(dl.Weights.DType)
		} else if inH != meta.dim {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: stack child %d dim %d want %d", i, inH, meta.dim)}
		} else if int32(dl.Core.Activation) != meta.childAct {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: stack child %d act mismatch", i)}
		}
		w, err := dl.NativeMatrixF32()
		if err != nil {
			return nil, meta, err
		}
		want := meta.dim * meta.dim
		if len(w) != want {
			return nil, meta, nativePackFallback{msg: fmt.Sprintf("parallel: stack child %d weight len %d want %d", i, len(w), want)}
		}
		parts = append(parts, w)
		meta.dense = append(meta.dense, dl)
		meta.stores = append(meta.stores, dl.Weights)
		total += len(w)
	}
	out := make([]float32, total)
	off := 0
	for _, p := range parts {
		copy(out[off:], p)
		off += len(p)
	}
	return out, meta, nil
}

func combineABI(m CombineMode) (int32, bool) {
	switch m {
	case CombineConcat:
		return 0, true
	case CombineAdd:
		return 1, true
	case CombineAvg:
		return 2, true
	case CombineFilter:
		return 3, true
	case CombineMax:
		return 4, true
	case CombineSparseK:
		return 5, true
	case CombineDisagree:
		return 6, true
	default:
		return 0, false
	}
}

func parallelLayout[T core.Numeric](l *Layer, input *core.Tensor[T]) (batch, seq int, err error) {
	if input == nil || len(input.Shape) < 2 {
		return 0, 0, fmt.Errorf("parallel: bad input shape")
	}
	batch = input.Shape[0]
	if l.Cfg.SeqLen > 0 {
		if len(input.Shape) < 3 || input.Shape[1] != l.Cfg.SeqLen || input.Shape[2] != l.Cfg.Dim {
			return 0, 0, fmt.Errorf("parallel: want [B,%d,%d]", l.Cfg.SeqLen, l.Cfg.Dim)
		}
		return batch, l.Cfg.SeqLen, nil
	}
	if input.Shape[1] != l.Cfg.Dim {
		return 0, 0, fmt.Errorf("parallel: want [B,%d]", l.Cfg.Dim)
	}
	return batch, 1, nil
}

func stackLayout[T core.Numeric](s *Stack, input *core.Tensor[T]) (batch, seq int, err error) {
	if s == nil || input == nil || len(input.Shape) < 2 {
		return 0, 0, fmt.Errorf("parallel: bad stack input shape")
	}
	dim := s.Core.InputHeight
	batch = input.Shape[0]
	if len(input.Shape) >= 3 {
		if input.Shape[2] != dim {
			return 0, 0, fmt.Errorf("parallel: stack want [B,S,%d]", dim)
		}
		return batch, input.Shape[1], nil
	}
	if input.Shape[1] != dim {
		return 0, 0, fmt.Errorf("parallel: stack want [B,%d]", dim)
	}
	return batch, 1, nil
}

// SyncWeightsFromNative pulls pending native weights from Dense branches / Gate
// and any packed layer mirror (CommitStores).
func (l *Layer) SyncWeightsFromNative() error {
	if l == nil {
		return nil
	}
	if l.nativeMir != nil {
		if err := l.nativeMir.SyncToGo(); err != nil {
			return err
		}
	}
	for i, b := range l.Branches {
		if err := syncBranch(b); err != nil {
			return fmt.Errorf("parallel branch %d: %w", i, err)
		}
	}
	if l.Gate != nil {
		if err := l.Gate.SyncWeightsFromNative(); err != nil {
			return err
		}
	}
	return nil
}

// SyncWeightsFromNative pulls pending native weights from Dense NestedStack children.
func (s *Stack) SyncWeightsFromNative() error {
	if s == nil {
		return nil
	}
	if s.nativeMir != nil {
		if err := s.nativeMir.SyncToGo(); err != nil {
			return err
		}
	}
	for i, ch := range s.Children {
		if err := syncBranch(ch); err != nil {
			return fmt.Errorf("stack child %d: %w", i, err)
		}
	}
	return nil
}

func syncBranch(b any) error {
	switch v := b.(type) {
	case *dense.Layer:
		return v.SyncWeightsFromNative()
	case interface{ SyncWeightsFromNative() error }:
		return v.SyncWeightsFromNative()
	default:
		return nil
	}
}

// NativeWeightsPending reports whether SyncWeightsFromNative would write.
func (l *Layer) NativeWeightsPending() bool {
	if l == nil {
		return false
	}
	if l.nativeMir != nil && l.nativeMir.Dirty() {
		return true
	}
	for _, b := range l.Branches {
		if pendingBranch(b) {
			return true
		}
	}
	return l.Gate != nil && l.Gate.NativeWeightsPending()
}

// NativeWeightsPending reports whether SyncWeightsFromNative would write on the Stack.
func (s *Stack) NativeWeightsPending() bool {
	if s == nil {
		return false
	}
	if s.nativeMir != nil && s.nativeMir.Dirty() {
		return true
	}
	for _, ch := range s.Children {
		if pendingBranch(ch) {
			return true
		}
	}
	return false
}

func pendingBranch(b any) bool {
	switch v := b.(type) {
	case *dense.Layer:
		return v.NativeWeightsPending()
	case interface{ NativeWeightsPending() bool }:
		return v.NativeWeightsPending()
	default:
		return false
	}
}

func (l *Layer) ensureMirror() *nativecpp.Mirror {
	if l.nativeMir == nil {
		l.nativeMir = &nativecpp.Mirror{}
	}
	return l.nativeMir
}

func (s *Stack) ensureMirror() *nativecpp.Mirror {
	if s.nativeMir == nil {
		s.nativeMir = &nativecpp.Mirror{}
	}
	return s.nativeMir
}
