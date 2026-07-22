package qwentts

import (
	"math"
	"testing"
)

func TestCausalConvMatchesScalar(t *testing.T) {
	cin, cout, T, k, dil := 32, 24, 1000, 7, 3
	cw := convWeight{cin: cin, cout: cout, k: k, w: make([]float32, cout*cin*k), b: make([]float32, cout)}
	in := make([]float32, cin*T)
	for i := range cw.w {
		cw.w[i] = float32((i%17)-8) * 0.01
	}
	for i := range cw.b {
		cw.b[i] = float32(i) * 0.001
	}
	for i := range in {
		in[i] = float32((i%13)-6) * 0.05
	}
	got := causalConv(in, cin, T, cw, dil)
	want := causalConvScalar(in, cin, T, cw, dil)
	if len(got) != len(want) {
		t.Fatalf("len %d vs %d", len(got), len(want))
	}
	var maxErr float64
	for i := range got {
		e := math.Abs(float64(got[i] - want[i]))
		if e > maxErr {
			maxErr = e
		}
	}
	if maxErr > 1e-4 {
		t.Fatalf("max err %g", maxErr)
	}
}

func causalConvScalar(in []float32, cin, T int, cw convWeight, dilation int) []float32 {
	cout, k := cw.cout, cw.k
	padLeft := (k - 1) * dilation
	out := make([]float32, cout*T)
	for oc := 0; oc < cout; oc++ {
		wbase := oc * cin * k
		for t := 0; t < T; t++ {
			var acc float32
			if cw.b != nil {
				acc = cw.b[oc]
			}
			for ic := 0; ic < cin; ic++ {
				irow := in[ic*T : ic*T+T]
				wrow := cw.w[wbase+ic*k : wbase+ic*k+k]
				for j := 0; j < k; j++ {
					tin := t - padLeft + j*dilation
					if tin >= 0 && tin < T {
						acc += wrow[j] * irow[tin]
					}
				}
			}
			out[oc*T+t] = acc
		}
	}
	return out
}

func BenchmarkCausalConv(b *testing.B) {
	cin, cout, T, k := 192, 192, 32000, 7
	cw := convWeight{cin: cin, cout: cout, k: k, w: make([]float32, cout*cin*k), b: make([]float32, cout)}
	in := make([]float32, cin*T)
	for i := range cw.w {
		cw.w[i] = 0.001
	}
	b.SetBytes(int64(cout * T * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = causalConv(in, cin, T, cw, 1)
	}
}
