package nativecpp_test

import (
	"math"
	"runtime"
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/softmax"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

func requireNative(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("linux/amd64 cgo only")
	}
	if !nativecpp.NativeCPUAvailable() {
		t.Fatal("expected libwelvet loaded")
	}
}

func almostExact(t *testing.T, name string, a, b []float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("%s len %d != %d", name, len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] && math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			// Allow tiny ulp noise only if both finite; prefer exact_bit.
			if math.Abs(float64(a[i]-b[i])) > 0 {
				t.Fatalf("%s[%d]: native=%g go=%g bits %x vs %x",
					name, i, a[i], b[i], math.Float32bits(a[i]), math.Float32bits(b[i]))
			}
		}
	}
}

func TestDenseForwardExactBitVsGoCPU(t *testing.T) {
	requireNative(t)
	const batch, in, out = 2, 4, 3
	w := make([]float32, out*in)
	x := make([]float32, batch*in)
	for i := range w {
		w[i] = float32(i+1) * 0.01
	}
	for i := range x {
		x[i] = float32(i+1) * 0.1
	}
	bias := []float64{0.01, -0.02, 0.03}

	preN := make([]float32, batch*out)
	postN := make([]float32, batch*out)
	if err := nativecpp.DenseForwardF32(
		batch, in, out, int32(core.ActivationReLU), int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		w, bias, x, preN, postN,
	); err != nil {
		t.Fatal(err)
	}

	l, err := dense.NewConfigured(in, out, core.ActivationReLU, core.DTypeFloat32, quant.FormatNone, w)
	if err != nil {
		t.Fatal(err)
	}
	l.Weights.Bias = append([]float64(nil), bias...)
	l.Exec.Backend = core.BackendCPUTiled
	xin := core.NewTensor[float32](batch, in)
	copy(xin.Data, x)
	preG, postG, err := dense.Forward(l, xin)
	if err != nil {
		t.Fatal(err)
	}
	almostExact(t, "dense.pre", preN, preG.Data)
	almostExact(t, "dense.post", postN, postG.Data)
}

func TestDenseNormalBPExactBitVsGoCPU(t *testing.T) {
	requireNative(t)
	const batch, in, out = 2, 3, 2
	w := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
	x := []float32{1, 2, 3, 0.5, 1.5, 2.5}
	target := []float32{0.2, 0.3, 0.4, 0.5}
	preN := make([]float32, batch*out)
	postN := make([]float32, batch*out)
	dX := make([]float32, batch*in)
	dW := make([]float32, out*in)
	wAfter := make([]float32, out*in)
	loss, err := nativecpp.DenseNormalBPF32(
		batch, in, out, int32(core.ActivationLinear), int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		w, nil, x, target, true, 0,
		preN, postN, dX, dW, wAfter,
	)
	if err != nil {
		t.Fatal(err)
	}
	if loss == 0 {
		t.Fatal("expected nonzero MSE loss")
	}

	l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
	if err != nil {
		t.Fatal(err)
	}
	l.Exec.Backend = core.BackendCPUTiled
	xin := core.NewTensor[float32](batch, in)
	copy(xin.Data, x)
	pre, post, err := dense.Forward(l, xin)
	if err != nil {
		t.Fatal(err)
	}
	almostExact(t, "nbp.pre", preN, pre.Data)
	almostExact(t, "nbp.post", postN, post.Data)

	// Grad from MSE: 2/n * (y - t)
	grad := core.NewTensor[float32](batch, out)
	n := float32(batch * out)
	for i := range grad.Data {
		grad.Data[i] = 2 / n * (post.Data[i] - target[i])
	}
	gi, gw, err := dense.Backward(l, grad, xin, pre)
	if err != nil {
		t.Fatal(err)
	}
	almostExact(t, "nbp.dX", dX, gi.Data)
	almostExact(t, "nbp.dW", dW, gw.Data)
}

func TestRMSNormForwardExactBit(t *testing.T) {
	requireNative(t)
	const batch, seq, dim = 1, 2, 4
	gamma := []float32{1, 1, 1, 1}
	x := []float32{1, 2, 3, 4, 0.5, 1.5, 2.5, 3.5}
	preN := make([]float32, batch*seq*dim)
	postN := make([]float32, batch*seq*dim)
	if err := nativecpp.RMSNormForwardF32(
		batch, seq, dim, 1e-6, int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		gamma, x, preN, postN,
	); err != nil {
		t.Fatal(err)
	}
	l, err := rmsnorm.NewConfigured(rmsnorm.Config{Dim: dim, Eps: 1e-6}, core.DTypeFloat32, quant.FormatNone, gamma)
	if err != nil {
		t.Fatal(err)
	}
	l.Exec.Backend = core.BackendCPUTiled
	xin := core.NewTensor[float32](batch, seq, dim)
	copy(xin.Data, x)
	_, postG, err := rmsnorm.Forward(l, xin)
	if err != nil {
		t.Fatal(err)
	}
	almostExact(t, "rmsnorm.post", postN, postG.Data)
}

func TestSoftmaxForwardExactBit(t *testing.T) {
	requireNative(t)
	x := []float32{1, 2, 3, 4}
	preN := make([]float32, len(x))
	postN := make([]float32, len(x))
	shape := []int32{1, 4}
	if err := nativecpp.SoftmaxForwardF32(
		4, 4, 1, 0, 0, 0, 1, 0, 0,
		int32(core.DTypeFloat32), nativecpp.CABBackendCPU,
		2, shape, nil, nil, x, preN, postN,
	); err != nil {
		t.Fatal(err)
	}
	l, err := softmax.New(softmax.Config{Dim: 4, SeqLen: 1})
	if err != nil {
		t.Fatal(err)
	}
	l.Exec.Backend = core.BackendCPUTiled
	xin := core.NewTensor[float32](1, 4)
	copy(xin.Data, x)
	_, postG, err := softmax.Forward(l, xin)
	if err != nil {
		t.Fatal(err)
	}
	almostExact(t, "softmax.post", postN, postG.Data)
}

func TestDenseLayerNativeCPUDispatch(t *testing.T) {
	requireNative(t)
	w := []float32{0.2, -0.1, 0.05, 0.3}
	l, err := dense.NewConfigured(2, 2, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, w)
	if err != nil {
		t.Fatal(err)
	}
	l.Exec.Backend = core.BackendNativeCPU
	xin := core.NewTensor[float32](1, 2)
	xin.Data[0], xin.Data[1] = 1.5, -0.5
	pre, post, err := dense.Forward(l, xin)
	if err != nil {
		t.Fatal(err)
	}
	l.Exec.Backend = core.BackendCPUTiled
	pre2, post2, err := dense.Forward(l, xin)
	if err != nil {
		t.Fatal(err)
	}
	almostExact(t, "dispatch.pre", pre.Data, pre2.Data)
	almostExact(t, "dispatch.post", post.Data, post2.Data)
}

func TestApplyExecConfig(t *testing.T) {
	cfg := nativecpp.ExecConfig{Default: nativecpp.BackendNativeCPU}
	l, err := dense.New(3, 2, core.ActivationLinear, core.DTypeFloat32)
	if err != nil {
		t.Fatal(err)
	}
	l.ApplyExecConfig(cfg, "main", "0,0,0,0", "0")
	if l.Exec.Backend != core.BackendNativeCPU {
		t.Fatalf("got %v", l.Exec.Backend)
	}
}

func TestPreferNativeCPU(t *testing.T) {
	requireNative(t)
	if core.PreferNativeCPU() != core.BackendNativeCPU {
		t.Fatal("expected PreferNativeCPU → NativeCPU when lib loaded")
	}
	// Default stays Go CPU; PreferNativeCPU is the opt-in helper.
	if core.DefaultBackend() != core.BackendCPUTiled {
		t.Fatal("DefaultBackend must remain CPUTiled")
	}
}
