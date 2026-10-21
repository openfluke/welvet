package nativecpp_test

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime"
	"strings"
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/nativecpp"
	"github.com/openfluke/welvet/quant"
)

func TestExecConfigJSONRoundTrip(t *testing.T) {
	raw := []byte(`{
  "default": "native_cpu",
  "volumes": {
    "main": {
      "sectors": {
        "1,0,0,0": {
          "backend": "cuda",
          "device": 0,
          "layers": {
            "3": {"backend": "native_cpu"}
          }
        }
      }
    }
  }
}`)
	cfg, err := nativecpp.ParseExecConfigJSON(raw)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := cfg.Resolve("main", "1,0,0,0", "3")
	if b != nativecpp.BackendNativeCPU {
		t.Fatalf("layer override: got %q", b)
	}
	b2, _ := cfg.Resolve("main", "1,0,0,0", "9")
	if b2 != nativecpp.BackendCUDA {
		t.Fatalf("sector backend: got %q", b2)
	}
	out, err := cfg.ToJSON()
	if err != nil {
		t.Fatal(err)
	}
	var again map[string]any
	if err := json.Unmarshal(out, &again); err != nil {
		t.Fatal(err)
	}
}

func TestEmptyDefaultIsGoCPU(t *testing.T) {
	cfg := nativecpp.ExecConfig{}
	b, _ := cfg.Resolve("main", "0,0,0,0", "0")
	if b != nativecpp.BackendGoCPU {
		t.Fatalf("empty default want go_cpu got %q", b)
	}
}

func TestDefaultBackendIsCPUTiled(t *testing.T) {
	if core.DefaultBackend() != core.BackendCPUTiled {
		t.Fatalf("DefaultBackend = %v want CPUTiled", core.DefaultBackend())
	}
}

func TestShippedNativeVersion(t *testing.T) {
	switch {
	case runtime.GOOS == "linux" && runtime.GOARCH == "amd64":
	case runtime.GOOS == "windows" && runtime.GOARCH == "arm64":
	default:
		t.Skip("no shipped libwelvet for this OS/arch")
	}
	if !nativecpp.NativeCPUAvailable() {
		t.Fatal("expected native lib loaded")
	}
	v := nativecpp.Version()
	if v == "" {
		t.Fatal("empty version")
	}
	t.Logf("welvet.cpp via C ABI: %s", v)
}

func TestDenseNativeCPUMatchesGoCPU(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	const batch, in, out = 2, 4, 3
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32(i+1) * 0.01
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i+1) * 0.1
	}

	goL, err := dense.NewConfigured(in, out, core.ActivationReLU, core.DTypeFloat32, 0, init)
	if err != nil {
		t.Fatal(err)
	}
	goL.Exec.Backend = core.BackendCPUTiled

	natL, err := dense.NewConfigured(in, out, core.ActivationReLU, core.DTypeFloat32, 0, init)
	if err != nil {
		t.Fatal(err)
	}
	natL.Exec.Backend = core.BackendNativeCPU

	preG, postG, err := dense.Forward(goL, x)
	if err != nil {
		t.Fatal(err)
	}
	preN, postN, err := dense.Forward(natL, x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range postG.Data {
		if postG.Data[i] != postN.Data[i] || preG.Data[i] != preN.Data[i] {
			t.Fatalf("fwd mismatch i=%d go=%v/%v nat=%v/%v", i, preG.Data[i], postG.Data[i], preN.Data[i], postN.Data[i])
		}
	}

	tgt := core.NewTensor[float32](batch, out)
	for i := range tgt.Data {
		tgt.Data[i] = postG.Data[i] + 0.05
	}
	// grad = post - tgt for MSE-ish
	grad := core.NewTensor[float32](batch, out)
	for i := range grad.Data {
		grad.Data[i] = postG.Data[i] - tgt.Data[i]
	}
	dXG, dWG, err := dense.Backward(goL, grad, x, preG)
	if err != nil {
		t.Fatal(err)
	}
	dXN, dWN, err := dense.Backward(natL, grad, x, preN)
	if err != nil {
		t.Fatal(err)
	}
	for i := range dXG.Data {
		if dXG.Data[i] != dXN.Data[i] {
			t.Fatalf("dX mismatch i=%d go=%v nat=%v", i, dXG.Data[i], dXN.Data[i])
		}
	}
	for i := range dWG.Data {
		if math.Float32bits(dWG.Data[i]) != math.Float32bits(dWN.Data[i]) {
			t.Fatalf("dW mismatch i=%d go=%v nat=%v", i, dWG.Data[i], dWN.Data[i])
		}
	}
}

func TestDenseNativeTrainSyncsWeights(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	const batch, in, out = 2, 4, 3
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32(i+1) * 0.05
	}
	l, err := dense.NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, 0, init)
	if err != nil {
		t.Fatal(err)
	}
	l.Exec.Backend = core.BackendNativeCPU
	before, err := l.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i+1) * 0.1
	}
	tgt := core.NewTensor[float32](batch, out)
	for i := range tgt.Data {
		tgt.Data[i] = 0.5
	}
	loss, err := dense.TrainNativeMSE(l, x, tgt, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !l.NativeWeightsPending() {
		t.Fatal("expected pending native weights before Sync")
	}
	mid, err := l.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	for i := range before {
		if before[i] != mid[i] {
			t.Fatal("Go Store should be unchanged until SyncWeightsFromNative")
		}
	}
	if err := l.SyncWeightsFromNative(); err != nil {
		t.Fatal(err)
	}
	after, err := l.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	changed := false
	for i := range before {
		if before[i] != after[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("expected W to change after SyncWeightsFromNative loss=%v", loss)
	}
	if l.NativeWeightsPending() {
		t.Fatal("pending should clear after Sync")
	}
	t.Logf("TrainNativeMSE loss=%g; SyncWeightsFromNative pulled into Go for .entity", loss)
}

func TestEmbeddingNativeTrainSyncsWeights(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	cfg := embedding.Config{VocabSize: 8, EmbeddingDim: 4, SeqLen: 3}
	init := make([]float32, cfg.WeightCount())
	for i := range init {
		init[i] = float32(i+1) * 0.02
	}
	l, err := embedding.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, init)
	if err != nil {
		t.Fatal(err)
	}
	before, err := l.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	x := core.NewTensor[float32](2, 3)
	for i := range x.Data {
		x.Data[i] = float32(i % 8)
	}
	tgt := core.NewTensor[float32](2, 3, 4)
	for i := range tgt.Data {
		tgt.Data[i] = 0.25
	}
	loss, err := embedding.TrainNativeMSE(l, x, tgt, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !l.NativeWeightsPending() {
		t.Fatal("expected pending native weights before Sync")
	}
	mid, err := l.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	for i := range before {
		if before[i] != mid[i] {
			t.Fatal("Go Store should be unchanged until SyncWeightsFromNative")
		}
	}
	if err := l.SyncWeightsFromNative(); err != nil {
		t.Fatal(err)
	}
	after, err := l.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	changed := false
	for i := range before {
		if before[i] != after[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("expected W to change after SyncWeightsFromNative loss=%v", loss)
	}
	if l.NativeWeightsPending() {
		t.Fatal("pending should clear after Sync")
	}
	t.Logf("Embedding TrainNativeMSE loss=%g; SyncWeightsFromNative pulled into Go", loss)
}

func TestDenseTrainNativeMSEDtypes(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	const batch, in, out = 2, 4, 3
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32(i+1) * 0.05
	}
	dtypes := []core.DType{
		core.DTypeFloat32, core.DTypeFloat16, core.DTypeBFloat16,
		core.DTypeInt8, core.DTypeUint8, core.DTypeInt16, core.DTypeFloat64,
	}
	for _, dt := range dtypes {
		dt := dt
		t.Run(dt.String(), func(t *testing.T) {
			l, err := dense.NewConfigured(in, out, core.ActivationLinear, dt, quant.FormatNone, init)
			if err != nil {
				t.Fatal(err)
			}
			l.Exec.Backend = core.BackendNativeCPU
			before, err := l.Weights.FlattenF32()
			if err != nil {
				t.Fatal(err)
			}
			x := core.NewTensor[float32](batch, in)
			for i := range x.Data {
				x.Data[i] = float32(i+1) * 0.1
			}
			tgt := core.NewTensor[float32](batch, out)
			for i := range tgt.Data {
				tgt.Data[i] = 0.5
			}
			loss, err := dense.TrainNativeMSE(l, x, tgt, 0.1)
			if err != nil {
				t.Fatalf("TrainNativeMSE: %v", err)
			}
			if !l.NativeWeightsPending() {
				t.Fatal("expected pending before Sync")
			}
			if err := l.SyncWeightsFromNative(); err != nil {
				t.Fatal(err)
			}
			after, err := l.Weights.FlattenF32()
			if err != nil {
				t.Fatal(err)
			}
			changed := false
			for i := range before {
				if before[i] != after[i] {
					changed = true
					break
				}
			}
			if !changed {
				t.Fatalf("expected W change after Sync loss=%v", loss)
			}
		})
	}
}

func TestParallelTrainNativeMSE(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	cfg := parallel.Config{Dim: 4, OutFeat: 3, Branches: 2, Combine: parallel.CombineAdd}
	init := make([]float32, cfg.Branches*cfg.OutFeat*cfg.Dim)
	for i := range init {
		init[i] = float32(i+1) * 0.03
	}
	l, err := parallel.NewConfigured(cfg, core.DTypeFloat32, quant.FormatNone, init, nil)
	if err != nil {
		t.Fatal(err)
	}
	l.Exec.Backend = core.BackendNativeCPU
	l.SyncBranchExec()
	x := core.NewTensor[float32](2, 4)
	for i := range x.Data {
		x.Data[i] = float32(i+1) * 0.1
	}
	tgt := core.NewTensor[float32](2, 3)
	for i := range tgt.Data {
		tgt.Data[i] = 0.4
	}
	d0, ok := l.DenseBranch(0)
	if !ok {
		t.Fatal("expected dense branch")
	}
	b0, err := d0.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	loss, err := parallel.TrainNativeMSE(l, x, tgt, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !l.NativeWeightsPending() {
		t.Fatal("expected pending")
	}
	if err := l.SyncWeightsFromNative(); err != nil {
		t.Fatal(err)
	}
	a0, err := d0.Weights.FlattenF32()
	if err != nil {
		t.Fatal(err)
	}
	changed := false
	for i := range b0 {
		if b0[i] != a0[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("expected branch W change loss=%v", loss)
	}
	t.Logf("Parallel TrainNativeMSE loss=%g", loss)
}

func TestStackTrainMSEModeTweenSyncPending(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	const dim, nChildren, batch = 4, 2, 2
	children := make([]any, nChildren)
	befores := make([][]float32, nChildren)
	for i := 0; i < nChildren; i++ {
		init := make([]float32, dim*dim)
		for j := range init {
			init[j] = float32(i*dim*dim+j+1) * 0.04
		}
		d, err := dense.NewConfigured(dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, init)
		if err != nil {
			t.Fatal(err)
		}
		children[i] = d
		b, err := d.Weights.FlattenF32()
		if err != nil {
			t.Fatal(err)
		}
		befores[i] = append([]float32(nil), b...)
	}
	s, err := parallel.NewStack(children...)
	if err != nil {
		t.Fatal(err)
	}
	if s.Exec.Backend != core.BackendCPUTiled {
		t.Fatalf("default backend want CPUTiled got %v", s.Exec.Backend)
	}
	s.SetBackend(core.BackendNativeCPU)
	x := core.NewTensor[float32](batch, dim)
	for i := range x.Data {
		x.Data[i] = float32(i+1) * 0.1
	}
	tgt := core.NewTensor[float32](batch, dim)
	for i := range tgt.Data {
		tgt.Data[i] = 0.35
	}
	loss, err := parallel.TrainStackMSE(s, x, tgt, parallel.ModeTween, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !s.NativeWeightsPending() {
		t.Fatal("expected pending native weights before Sync")
	}
	for i, ch := range s.Children {
		d := ch.(*dense.Layer)
		mid, err := d.Weights.FlattenF32()
		if err != nil {
			t.Fatal(err)
		}
		for j := range befores[i] {
			if befores[i][j] != mid[j] {
				t.Fatal("Go Store should be unchanged until SyncWeightsFromNative")
			}
		}
	}
	if err := s.SyncWeightsFromNative(); err != nil {
		t.Fatal(err)
	}
	changed := false
	for i, ch := range s.Children {
		d := ch.(*dense.Layer)
		after, err := d.Weights.FlattenF32()
		if err != nil {
			t.Fatal(err)
		}
		for j := range befores[i] {
			if befores[i][j] != after[j] {
				changed = true
				break
			}
		}
	}
	if !changed {
		t.Fatalf("expected W to change after SyncWeightsFromNative loss=%v", loss)
	}
	if s.NativeWeightsPending() {
		t.Fatal("pending should clear after Sync")
	}
	t.Logf("StackTrainMSE ModeTween loss=%g; SyncWeightsFromNative pulled into Go", loss)
}

func TestStackTrainMSEFamiliesSyncPending(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	modes := []parallel.TrainMode{
		parallel.ModeNormalBP,
		parallel.ModeTween,
		parallel.ModeTweenChain,
		parallel.ModeTweenSplit,
		parallel.ModeTweenAlt,
		parallel.ModeFreeze,
	}
	const dim, nChildren, batch = 4, 2, 2
	for _, mode := range modes {
		mode := mode
		t.Run(fmt.Sprintf("mode_%d", mode), func(t *testing.T) {
			children := make([]any, nChildren)
			befores := make([][]float32, nChildren)
			for i := 0; i < nChildren; i++ {
				init := make([]float32, dim*dim)
				for j := range init {
					init[j] = float32(i*dim*dim+j+1) * 0.03
				}
				d, err := dense.NewConfigured(dim, dim, core.ActivationLinear, core.DTypeFloat32, quant.FormatNone, init)
				if err != nil {
					t.Fatal(err)
				}
				children[i] = d
				b, err := d.Weights.FlattenF32()
				if err != nil {
					t.Fatal(err)
				}
				befores[i] = append([]float32(nil), b...)
			}
			s, err := parallel.NewStack(children...)
			if err != nil {
				t.Fatal(err)
			}
			s.SetBackend(core.BackendNativeCPU)
			x := core.NewTensor[float32](batch, dim)
			for i := range x.Data {
				x.Data[i] = float32(i+1) * 0.08
			}
			tgt := core.NewTensor[float32](batch, dim)
			for i := range tgt.Data {
				tgt.Data[i] = 0.4
			}
			loss, err := parallel.TrainStackMSE(s, x, tgt, mode, 0.05)
			if err != nil {
				t.Fatal(err)
			}
			if !s.NativeWeightsPending() {
				t.Fatal("expected pending before Sync")
			}
			if err := s.SyncWeightsFromNative(); err != nil {
				t.Fatal(err)
			}
			changed := false
			for i, ch := range s.Children {
				d := ch.(*dense.Layer)
				after, err := d.Weights.FlattenF32()
				if err != nil {
					t.Fatal(err)
				}
				for j := range befores[i] {
					if befores[i][j] != after[j] {
						changed = true
						break
					}
				}
			}
			if mode == parallel.ModeFreeze {
				if changed {
					t.Fatalf("Freeze should not change W loss=%v", loss)
				}
			} else if !changed {
				t.Fatalf("expected W change mode=%v loss=%v", mode, loss)
			}
		})
	}
}

func TestParallelTrainMSEBranchModesSync(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	modes := []parallel.TrainMode{parallel.ModeTween, parallel.ModeAdversarial}
	const dim, outFeat, branches, batch = 4, 4, 2, 2
	for _, mode := range modes {
		mode := mode
		t.Run(fmt.Sprintf("mode_%d", mode), func(t *testing.T) {
			init := make([]float32, branches*outFeat*dim)
			for i := range init {
				init[i] = float32(i+1) * 0.02
			}
			l, err := parallel.NewConfigured(parallel.Config{
				Dim: dim, OutFeat: outFeat, Branches: branches, Combine: parallel.CombineAdd,
			}, core.DTypeFloat32, quant.FormatNone, init, nil)
			if err != nil {
				t.Fatal(err)
			}
			l.Exec.Backend = core.BackendNativeCPU
			l.SyncBranchExec()
			befores := make([][]float32, branches)
			for i := 0; i < branches; i++ {
				d, ok := l.DenseBranch(i)
				if !ok {
					t.Fatal("expected dense branch")
				}
				w, err := d.Weights.FlattenF32()
				if err != nil {
					t.Fatal(err)
				}
				befores[i] = append([]float32(nil), w...)
			}
			x := core.NewTensor[float32](batch, dim)
			for i := range x.Data {
				x.Data[i] = float32(i+1) * 0.07
			}
			tgt := core.NewTensor[float32](batch, outFeat)
			for i := range tgt.Data {
				tgt.Data[i] = 0.25
			}
			loss, err := parallel.TrainMSE(l, x, tgt, mode, 0.08)
			if err != nil {
				t.Fatal(err)
			}
			if !l.NativeWeightsPending() {
				t.Fatal("expected pending before Sync")
			}
			if err := l.SyncWeightsFromNative(); err != nil {
				t.Fatal(err)
			}
			changed := false
			for i := 0; i < branches; i++ {
				d, ok := l.DenseBranch(i)
				if !ok {
					t.Fatal("expected dense branch")
				}
				after, err := d.Weights.FlattenF32()
				if err != nil {
					t.Fatal(err)
				}
				for j := range befores[i] {
					if befores[i][j] != after[j] {
						changed = true
						break
					}
				}
			}
			if !changed {
				t.Fatalf("expected branch W change mode=%v loss=%v", mode, loss)
			}
		})
	}
}

func TestGridTrainMSEMeshBP(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	const dim, batch, ticks = 3, 2, 2
	const depth, rows, cols, lpc = 1, 1, 2, 1
	nCells := depth * rows * cols * lpc
	packed := make([]float32, nCells*dim*dim)
	for i := range packed {
		packed[i] = float32(i+1) * 0.05
	}
	wAfter := make([]float32, len(packed))
	copy(wAfter, packed)
	x := make([]float32, batch*dim)
	tgt := make([]float32, batch*dim)
	for i := range x {
		x[i] = float32(i+1) * 0.1
		tgt[i] = 0.3
	}
	before := append([]float32(nil), packed...)
	loss, err := nativecpp.GridTrainMSE(
		int32(parallel.ModeMeshBP), depth, rows, cols, lpc, batch, ticks, dim,
		1, /* Dense */ 0, int32(core.ActivationLinear), int32(core.DTypeFloat32), 0,
		packed, x, tgt, 0.1, wAfter,
	)
	if err != nil {
		t.Fatal(err)
	}
	changed := false
	for i := range before {
		if before[i] != wAfter[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("expected grid W change loss=%v", loss)
	}
	t.Logf("GridTrainMSE MeshBP loss=%g", loss)
}

func TestStackLineTrainMSEStepTween(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	const nChildren, nTicks, batch, dim = 2, 3, 2, 3
	inH := []int32{dim, dim}
	outH := []int32{dim, dim}
	act := []int32{int32(core.ActivationLinear), int32(core.ActivationLinear)}
	packed := make([]float32, nChildren*dim*dim)
	for i := range packed {
		packed[i] = float32(i+1) * 0.04
	}
	wAfter := make([]float32, len(packed))
	xTicks := make([]float32, nTicks*batch*dim)
	tgtTicks := make([]float32, nTicks*batch*dim)
	for i := range xTicks {
		xTicks[i] = float32(i+1) * 0.05
		tgtTicks[i] = 0.2
	}
	before := append([]float32(nil), packed...)
	loss, err := nativecpp.StackLineTrainMSE(
		int32(parallel.ModeStepTween), nChildren, nTicks, batch, 0.1, int32(core.DTypeFloat32),
		inH, outH, act, packed, xTicks, tgtTicks, wAfter,
	)
	if err != nil {
		t.Fatal(err)
	}
	changed := false
	for i := range before {
		if before[i] != wAfter[i] {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatalf("expected line W change loss=%v", loss)
	}
	t.Logf("StackLineTrainMSE StepTween loss=%g", loss)
}

func TestNativeEngineVersion085(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" || !nativecpp.NativeCPUAvailable() {
		t.Skip("needs linux/amd64 libwelvet")
	}
	v := nativecpp.Version()
	if !strings.HasPrefix(v, "0.1.85") {
		t.Fatalf("want engine 0.1.85* got %q", v)
	}
}
