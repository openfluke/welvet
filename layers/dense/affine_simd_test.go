package dense

import (
	"math"
	"testing"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

func TestAffinePackedSIMDMatchesCPU(t *testing.T) {
	if !simd.Enabled() {
		t.Skip("Plan 9 SIMD not enabled")
	}
	const in, out, batch = 64, 32, 2
	init := make([]float32, out*in)
	for i := range init {
		init[i] = float32((i%13)-6) * 0.1
	}
	lCPU, err := NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatAffinePacked, init)
	if err != nil {
		t.Fatalf("cpu layer: %v", err)
	}
	lCPU.Exec.Backend = core.BackendCPUTiled
	lSIMD, err := NewConfigured(in, out, core.ActivationLinear, core.DTypeFloat32, quant.FormatAffinePacked, init)
	if err != nil {
		t.Fatalf("simd layer: %v", err)
	}
	lSIMD.Exec.Backend = core.BackendSIMD

	x := core.NewTensor[float32](batch, in)
	for i := range x.Data {
		x.Data[i] = float32(i%5)*0.25 + 0.1
	}
	preCPU, _, err := Forward(lCPU, x)
	if err != nil {
		t.Fatalf("cpu fwd: %v", err)
	}
	preSIMD, _, err := Forward(lSIMD, x)
	if err != nil {
		t.Fatalf("simd fwd: %v", err)
	}
	const tol = 5e-3
	for i := range preCPU.Data {
		if math.Abs(float64(preCPU.Data[i]-preSIMD.Data[i])) > tol {
			t.Fatalf("idx %d: cpu=%v simd=%v", i, preCPU.Data[i], preSIMD.Data[i])
		}
	}

	y := make([]float32, out)
	if err := MatVecPackedBlob(lSIMD.Weights.Packed, x.Data[:in], y); err != nil {
		t.Fatalf("MatVecPackedBlob Affine: %v", err)
	}
}
