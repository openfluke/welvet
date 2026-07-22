//go:build !amd64

package simd

import "unsafe"

func saxpyF32Simd(y *float32, alpha float32, x *float32, n int) {
	if n <= 0 {
		return
	}
	saxpyF32Go(unsafe.Slice(y, n), alpha, unsafe.Slice(x, n), n)
}
