package dense

import (
	"runtime"
	"testing"

	"github.com/openfluke/welvet/quant"
	"github.com/openfluke/welvet/simd"
)

func TestGemvQ4ParallelSmoke(t *testing.T) {
	t.Logf("GOMAXPROCS=%d NumCPU=%d simd=%v", runtime.GOMAXPROCS(0), runtime.NumCPU(), simd.Enabled())
	b, err := quant.Pack(quant.FormatQ4_0, randF32(128*64), 128, 64)
	if err != nil {
		t.Fatal(err)
	}
	quant.EnsureQ4SIMDCache(b)
	x := randF32(64)
	y := make([]float32, 128)
	if err := MatVecQ4_0Blob(b, x, y); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkMatVecQ4_LMHead(b *testing.B) {
	blob, err := quant.Pack(quant.FormatQ4_0, randF32(49152*576), 49152, 576)
	if err != nil {
		b.Fatal(err)
	}
	quant.EnsureQ4SIMDCache(blob)
	x := randF32(576)
	y := make([]float32, 49152)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MatVecQ4_0Blob(blob, x, y); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatVecQ4_Proj(b *testing.B) {
	blob, err := quant.Pack(quant.FormatQ4_0, randF32(576*576), 576, 576)
	if err != nil {
		b.Fatal(err)
	}
	quant.EnsureQ4SIMDCache(blob)
	x := randF32(576)
	y := make([]float32, 576)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatVecQ4_0Blob(blob, x, y)
	}
}

func BenchmarkMatVecQ4_FFN(b *testing.B) {
	blob, err := quant.Pack(quant.FormatQ4_0, randF32(1536*576), 1536, 576)
	if err != nil {
		b.Fatal(err)
	}
	quant.EnsureQ4SIMDCache(blob)
	x := randF32(576)
	y := make([]float32, 1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatVecQ4_0Blob(blob, x, y)
	}
}

func randF32(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(i%17)/17 - 0.5
	}
	return out
}
