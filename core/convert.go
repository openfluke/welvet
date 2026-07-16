package core

import "math"

// AsFloat64 converts a Numeric value to float64 (kernel bridge — not a hardcoded tensor type).
// Complex → real part. Integer/float → direct cast.
func AsFloat64[T Numeric](v T) float64 {
	switch x := any(v).(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case complex64:
		return float64(real(x))
	case complex128:
		return real(x)
	case int:
		return float64(x)
	case int8:
		return float64(x)
	case int16:
		return float64(x)
	case int32:
		return float64(x)
	case int64:
		return float64(x)
	case uint:
		return float64(x)
	case uint8:
		return float64(x)
	case uint16:
		return float64(x)
	case uint32:
		return float64(x)
	case uint64:
		return float64(x)
	case uintptr:
		return float64(x)
	default:
		return 0
	}
}

// FromFloat64 converts float64 into T (complex → real=v, imag=0).
func FromFloat64[T Numeric](v float64) T {
	var zero T
	switch any(zero).(type) {
	case float64:
		return any(v).(T)
	case float32:
		return any(float32(v)).(T)
	case complex64:
		return any(complex64(complex(float32(v), 0))).(T)
	case complex128:
		return any(complex(v, 0)).(T)
	case int:
		return any(int(math.Round(v))).(T)
	case int8:
		return any(int8(clampI64(int64(math.Round(v)), -128, 127))).(T)
	case int16:
		return any(int16(clampI64(int64(math.Round(v)), -32768, 32767))).(T)
	case int32:
		return any(int32(math.Round(v))).(T)
	case int64:
		return any(int64(math.Round(v))).(T)
	case uint:
		if v <= 0 {
			return any(uint(0)).(T)
		}
		return any(uint(math.Round(v))).(T)
	case uint8:
		return any(uint8(clampI64(int64(math.Round(v)), 0, 255))).(T)
	case uint16:
		return any(uint16(clampI64(int64(math.Round(v)), 0, 65535))).(T)
	case uint32:
		if v <= 0 {
			return any(uint32(0)).(T)
		}
		return any(uint32(math.Round(v))).(T)
	case uint64:
		if v <= 0 {
			return any(uint64(0)).(T)
		}
		return any(uint64(math.Round(v))).(T)
	case uintptr:
		if v <= 0 {
			return any(uintptr(0)).(T)
		}
		return any(uintptr(math.Round(v))).(T)
	default:
		return zero
	}
}

// SliceAsFloat64 copies a Numeric slice into float64.
func SliceAsFloat64[T Numeric](in []T) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = AsFloat64(v)
	}
	return out
}

// SliceFromFloat64 writes float64 into out[i] = FromFloat64[T](in[i]).
func SliceFromFloat64[T Numeric](in []float64, out []T) {
	n := len(in)
	if len(out) < n {
		n = len(out)
	}
	for i := 0; i < n; i++ {
		out[i] = FromFloat64[T](in[i])
	}
}

// SliceAsFloat32 copies Numeric → float32 (for Plan 9 f32 / ggml pack bridges only).
func SliceAsFloat32[T Numeric](in []T) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(AsFloat64(v))
	}
	return out
}

// SliceFromFloat32 writes float32 into Numeric out.
func SliceFromFloat32[T Numeric](in []float32, out []T) {
	n := len(in)
	if len(out) < n {
		n = len(out)
	}
	for i := 0; i < n; i++ {
		out[i] = FromFloat64[T](float64(in[i]))
	}
}

func clampI64(v, lo, hi int64) int64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
