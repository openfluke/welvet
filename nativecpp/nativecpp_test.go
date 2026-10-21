package nativecpp_test

import (
	"encoding/json"
	"math"
	"runtime"
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

func TestLinuxAmd64NativeVersion(t *testing.T) {
	if runtime.GOOS != "linux" || runtime.GOARCH != "amd64" {
		t.Skip("linux/amd64 only")
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
