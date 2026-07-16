package core

import "math"

// Activate applies a pointwise activation in float64 then casts back to T.
func Activate[T Numeric](v T, act ActivationType) T {
	x := AsFloat64(v)
	switch act {
	case ActivationLinear:
		return v
	case ActivationReLU:
		if x < 0 {
			return FromFloat64[T](0)
		}
		return v
	case ActivationReLU2:
		if x < 0 {
			return FromFloat64[T](0)
		}
		return FromFloat64[T](x * x)
	case ActivationLeakyReLU:
		if x < 0 {
			return FromFloat64[T](0.01 * x)
		}
		return v
	case ActivationSilu:
		return FromFloat64[T](x / (1 + math.Exp(-x)))
	case ActivationGELU:
		return FromFloat64[T](0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x))))
	case ActivationTanh:
		return FromFloat64[T](math.Tanh(x))
	case ActivationSigmoid:
		return FromFloat64[T](1 / (1 + math.Exp(-x)))
	default:
		return v
	}
}

// ActivateDeriv returns d(act)/d(pre) at pre-activation v.
func ActivateDeriv[T Numeric](v T, act ActivationType) T {
	x := AsFloat64(v)
	switch act {
	case ActivationLinear:
		return FromFloat64[T](1)
	case ActivationReLU:
		if x > 0 {
			return FromFloat64[T](1)
		}
		return FromFloat64[T](0)
	case ActivationReLU2:
		if x > 0 {
			return FromFloat64[T](2 * x)
		}
		return FromFloat64[T](0)
	case ActivationLeakyReLU:
		if x > 0 {
			return FromFloat64[T](1)
		}
		return FromFloat64[T](0.01)
	case ActivationSilu:
		sig := 1 / (1 + math.Exp(-x))
		return FromFloat64[T](sig * (1 + x*(1-sig)))
	case ActivationGELU:
		u := math.Sqrt(2/math.Pi) * (x + 0.044715*x*x*x)
		th := math.Tanh(u)
		du := math.Sqrt(2/math.Pi) * (1 + 3*0.044715*x*x)
		return FromFloat64[T](0.5*(1+th) + 0.5*x*(1-th*th)*du)
	case ActivationTanh:
		t := math.Tanh(x)
		return FromFloat64[T](1 - t*t)
	case ActivationSigmoid:
		s := 1 / (1 + math.Exp(-x))
		return FromFloat64[T](s * (1 - s))
	default:
		return FromFloat64[T](1)
	}
}
