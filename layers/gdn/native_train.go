package gdn

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

// TrainNativeMSE runs C++ GDN NormalBP with lr>0. Full packed W_after stays pending
// until SyncWeightsFromNative (blobs + Conv/ALog/DtBias/γ).
func TrainNativeMSE[T core.Numeric](l *Layer, input, target *core.Tensor[T], lr float64) (loss float64, err error) {
	if l == nil || input == nil || target == nil {
		return 0, fmt.Errorf("gdn: TrainNativeMSE nil")
	}
	if lr == 0 {
		return 0, fmt.Errorf("gdn: TrainNativeMSE lr must be > 0")
	}
	if l.InQKV == nil || l.InQKV.Format != quant.FormatNone || !nativecpp.NativeCPUAvailable() {
		return 0, fmt.Errorf("gdn: TrainNativeMSE needs FormatNone + libwelvet")
	}
	if len(input.Shape) != 3 || input.Shape[2] != l.Cfg.HiddenSize {
		return 0, fmt.Errorf("gdn: TrainNativeMSE need [batch,seq,hidden]")
	}
	batch, seq, h := input.Shape[0], input.Shape[1], l.Cfg.HiddenSize
	packed, err := packGDNNative(l)
	if err != nil {
		return 0, err
	}
	n := batch * seq * h
	nw := len(packed)
	xF := core.SliceAsFloat32(input.Data)
	tF := core.SliceAsFloat32(target.Data)
	preF := make([]float32, n)
	postF := make([]float32, n)
	dX := make([]float32, n)
	dW := make([]float32, nw)
	wAfter := make([]float32, nw)
	c := l.Cfg
	loss, err = nativecpp.GDNNormalBPF32(
		int32(batch), int32(seq), int32(h),
		int32(c.NumKeyHeads), int32(c.NumValueHeads),
		int32(c.KeyHeadDim), int32(c.ValueHeadDim), int32(c.ConvKernel),
		int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		packed, xF, tF, true, lr,
		preF, postF, dX, dW, wAfter,
	)
	if err != nil {
		return 0, err
	}
	l.pendingPacked = append([]float32(nil), wAfter...)
	l.packPending = true
	return loss, nil
}

func packGDNNative(l *Layer) ([]float32, error) {
	if l.packPending && len(l.pendingPacked) == l.GradWSize() {
		return append([]float32(nil), l.pendingPacked...), nil
	}
	c := l.Cfg
	cd, vd, nh, h := c.convDim(), c.valueDim(), c.NumValueHeads, c.HiddenSize
	k := c.ConvKernel
	if k < 1 {
		k = 1
	}
	out := make([]float32, l.GradWSize())
	off := 0
	putBlob := func(b *quant.Blob, n int) error {
		f32, err := quant.Unpack(b)
		if err != nil {
			return err
		}
		if len(f32) < n {
			return fmt.Errorf("gdn: blob short")
		}
		copy(out[off:off+n], f32[:n])
		off += n
		return nil
	}
	if err := putBlob(l.InQKV, cd*h); err != nil {
		return nil, err
	}
	if err := putBlob(l.InZ, vd*h); err != nil {
		return nil, err
	}
	if err := putBlob(l.InB, nh*h); err != nil {
		return nil, err
	}
	if err := putBlob(l.InA, nh*h); err != nil {
		return nil, err
	}
	if err := putBlob(l.Out, h*vd); err != nil {
		return nil, err
	}
	copy(out[off:off+cd*k], l.ConvWeight)
	off += cd * k
	copy(out[off:off+nh], l.ALog)
	off += nh
	copy(out[off:off+nh], l.DtBias)
	off += nh
	copy(out[off:off+c.ValueHeadDim], l.NormGamma)
	return out, nil
}

func applyGDNPacked(l *Layer, packed []float32) error {
	c := l.Cfg
	cd, vd, nh, h := c.convDim(), c.valueDim(), c.NumValueHeads, c.HiddenSize
	k := c.ConvKernel
	if k < 1 {
		k = 1
	}
	if len(packed) < l.GradWSize() {
		return fmt.Errorf("gdn: packed short")
	}
	off := 0
	take := func(n int) []float32 {
		s := packed[off : off+n]
		off += n
		return s
	}
	setBlob := func(bp **quant.Blob, n, rows, cols int) error {
		nb, err := quant.Pack((*bp).Format, take(n), rows, cols)
		if err != nil {
			return err
		}
		*bp = nb
		return nil
	}
	if err := setBlob(&l.InQKV, cd*h, cd, h); err != nil {
		return err
	}
	if err := setBlob(&l.InZ, vd*h, vd, h); err != nil {
		return err
	}
	if err := setBlob(&l.InB, nh*h, nh, h); err != nil {
		return err
	}
	if err := setBlob(&l.InA, nh*h, nh, h); err != nil {
		return err
	}
	if err := setBlob(&l.Out, h*vd, h, vd); err != nil {
		return err
	}
	copy(l.ConvWeight, take(cd*k))
	copy(l.ALog, take(nh))
	copy(l.DtBias, take(nh))
	copy(l.NormGamma, take(c.ValueHeadDim))
	return nil
}

func (l *Layer) SyncWeightsFromNative() error {
	if l == nil || !l.packPending {
		return nil
	}
	if err := applyGDNPacked(l, l.pendingPacked); err != nil {
		return err
	}
	l.packPending = false
	l.pendingPacked = nil
	return nil
}

func (l *Layer) NativeWeightsPending() bool {
	return l != nil && l.packPending
}
