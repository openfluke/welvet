package core

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseDType maps a persistence / ENTITY dtype string to DType.
func ParseDType(s string) DType {
	s = strings.TrimSpace(s)
	if s == "" {
		return DTypeFloat32
	}
	lower := strings.ToLower(s)
	for _, dt := range AllDTypes {
		if strings.ToLower(dt.String()) == lower {
			return dt
		}
	}
	// ENTITY / HF uppercase aliases
	switch strings.ToUpper(s) {
	case "FLOAT32", "F32":
		return DTypeFloat32
	case "FLOAT64", "F64", "DOUBLE":
		return DTypeFloat64
	case "FLOAT16", "F16", "FP16":
		return DTypeFloat16
	case "BFLOAT16", "BF16":
		return DTypeBFloat16
	}
	if n, err := strconv.Atoi(s); err == nil && n >= 0 && n <= 33 {
		return DType(n)
	}
	return DTypeFloat32
}

// ParseLayerType maps a persistence type string to LayerType.
func ParseLayerType(s string) (LayerType, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return LayerDense, fmt.Errorf("core: empty layer type")
	}
	for t := LayerDense; t <= LayerGDN; t++ {
		if t.String() == s {
			return t, nil
		}
	}
	if n, err := strconv.Atoi(s); err == nil && n >= 0 && n <= int(LayerGDN) {
		return LayerType(n), nil
	}
	return LayerDense, fmt.Errorf("core: unknown layer type %q", s)
}

// ParseActivation maps a persistence activation string.
func ParseActivation(s string) ActivationType {
	switch strings.TrimSpace(s) {
	case "ReLU", "0":
		return ActivationReLU
	case "SiLU", "1":
		return ActivationSilu
	case "GELU", "2":
		return ActivationGELU
	case "Tanh", "3":
		return ActivationTanh
	case "Sigmoid", "4":
		return ActivationSigmoid
	case "LeakyReLU", "5":
		return ActivationLeakyReLU
	case "ReLU2", "6":
		return ActivationReLU2
	case "Linear", "-1", "":
		return ActivationLinear
	default:
		return ActivationLinear
	}
}
