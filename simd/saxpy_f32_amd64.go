//go:build amd64

package simd

//go:noescape
func saxpyF32Avx2(y *float32, alpha float32, x *float32, n int)

func saxpyF32Simd(y *float32, alpha float32, x *float32, n int) {
	if n <= 0 {
		return
	}
	saxpyF32Avx2(y, alpha, x, n)
}
