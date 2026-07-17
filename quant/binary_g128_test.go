package quant

import (
	"math"
	"testing"
)

func TestMatVecBinaryG128Exact(t *testing.T) {
	rows, cols := 17, 256 // 2 groups/row, non-trivial
	wordsPerRow := cols / 32
	groupsPerRow := cols / BinaryG128Group
	weight := make([]uint32, rows*wordsPerRow)
	scales := make([]float32, rows*groupsPerRow)
	biases := make([]float32, rows*groupsPerRow)
	for r := 0; r < rows; r++ {
		for g := 0; g < groupsPerRow; g++ {
			sg := float32(0.01 * float64(r+1) * float64(g+1))
			scales[r*groupsPerRow+g] = 2 * sg // MLX affine scale
			biases[r*groupsPerRow+g] = -sg
		}
		for w := 0; w < wordsPerRow; w++ {
			weight[r*wordsPerRow+w] = uint32(0xA5A5A5A5 ^ uint32(r*31+w*17))
		}
	}
	b, err := BlobFromMLX1Bit(weight, scales, biases, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.1)) * 0.5
	}
	got := make([]float32, rows)
	if err := matVecBinaryG128(b, x, got); err != nil {
		t.Fatal(err)
	}
	// Reference: decode each row and naive dot.
	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		if err := decodeRowBinaryG128(b, r, row); err != nil {
			t.Fatal(err)
		}
		var want float32
		for c := 0; c < cols; c++ {
			want += row[c] * x[c]
		}
		if math.Abs(float64(got[r]-want)) > 1e-4 {
			t.Fatalf("row %d: got %g want %g", r, got[r], want)
		}
	}
}

func BenchmarkMatVecBinaryG128(b *testing.B) {
	rows, cols := 2048, 2048
	wordsPerRow := cols / 32
	groupsPerRow := cols / BinaryG128Group
	weight := make([]uint32, rows*wordsPerRow)
	scales := make([]float32, rows*groupsPerRow)
	biases := make([]float32, rows*groupsPerRow)
	for i := range scales {
		scales[i] = 0.02
		biases[i] = -0.01
	}
	for i := range weight {
		weight[i] = uint32(i * 2654435761)
	}
	blob, err := BlobFromMLX1Bit(weight, scales, biases, rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	x := make([]float32, cols)
	y := make([]float32, rows)
	for i := range x {
		x[i] = 0.001 * float32(i%17)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := matVecBinaryG128(blob, x, y); err != nil {
			b.Fatal(err)
		}
	}
}
