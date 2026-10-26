package nativecpp

import "github.com/openfluke/welvet/core"

func init() {
	core.SetNativeCPUAvailableChecker(NativeCPUAvailable)
}

// ToCoreBackend maps ExecConfig backend name strings to core.Backend.
func ToCoreBackend(name string) core.Backend {
	return core.ParseBackend(name)
}

// FromCoreBackend maps core.Backend to ExecConfig name strings.
func FromCoreBackend(b core.Backend) string {
	switch b {
	case core.BackendNativeCPU:
		return BackendNativeCPU
	case core.BackendMetal:
		return BackendMetal
	case core.BackendQNN:
		return "qnn"
	case core.BackendSIMD:
		return "simd"
	case core.BackendWebGPU:
		return "webgpu"
	default:
		return BackendGoCPU
	}
}

// ApplyExecConfig resolves cfg for (volume, sector, layer) and writes into exec.
func ApplyExecConfig(cfg ExecConfig, volume, sector, layer string, exec *core.ExecConfig) {
	if exec == nil {
		return
	}
	name, _ := cfg.Resolve(volume, sector, layer)
	exec.Backend = ToCoreBackend(name)
	exec.BackendPinned = true
}

// PreferBackend returns core.PreferNativeCPU() — NativeCPU when lib loaded, else Go CPU.
func PreferBackend() core.Backend {
	return core.PreferNativeCPU()
}
