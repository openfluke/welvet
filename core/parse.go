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
	for t := LayerDense; t <= LayerStack; t++ {
		if t.String() == s {
			return t, nil
		}
	}
	if n, err := strconv.Atoi(s); err == nil && n >= 0 && n <= int(LayerStack) {
		return LayerType(n), nil
	}
	return LayerDense, fmt.Errorf("core: unknown layer type %q", s)
}

// ParseBackend maps a persistence / ExecConfig backend string.
func ParseBackend(s string) Backend {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "cpu-tiled", "cpu", "go_cpu", "gotiled", "0":
		return BackendCPUTiled
	case "simd", "1":
		return BackendSIMD
	case "webgpu", "gpu", "2":
		return BackendWebGPU
	case "native_cpu", "native-cpu", "native", "cpp", "3":
		return BackendNativeCPU
	case "metal", "apple", "5":
		return BackendMetal
	case "cuda", "6":
		return BackendCuda
	case "qnn", "htp", "npu", "hexagon", "4":
		return BackendQNN
	default:
		if n, err := strconv.Atoi(strings.TrimSpace(s)); err == nil {
			switch Backend(n) {
			case BackendCPUTiled, BackendSIMD, BackendWebGPU, BackendNativeCPU, BackendQNN, BackendMetal, BackendCuda:
				return Backend(n)
			}
		}
		return BackendCPUTiled
	}
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
