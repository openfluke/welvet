package core

// nativeCPUAvailable reports whether welvet.cpp is loaded for NativeCPU dispatch.
// nativecpp registers the real checker from init(); default is false (pure Go).
var nativeCPUAvailable = func() bool { return false }

// qnnAvailable reports whether welvet.qnn is loaded for BackendQNN dispatch.
// nativeqnn registers the real checker from init(); default is false.
var qnnAvailable = func() bool { return false }

// mlxAvailable reports whether welvet.mlx is loaded for BackendMLX dispatch.
// nativemlx registers the real checker from init(); default is false.
var mlxAvailable = func() bool { return false }

// SetNativeCPUAvailableChecker lets nativecpp register libwelvet availability.
// Pass nil to reset to "unavailable".
func SetNativeCPUAvailableChecker(fn func() bool) {
	if fn == nil {
		nativeCPUAvailable = func() bool { return false }
		return
	}
	nativeCPUAvailable = fn
}

// SetQNNAvailableChecker lets nativeqnn register libwelvet_qnn availability.
func SetQNNAvailableChecker(fn func() bool) {
	if fn == nil {
		qnnAvailable = func() bool { return false }
		return
	}
	qnnAvailable = fn
}

// SetMLXAvailableChecker lets nativemlx register libwelvet_mlx availability.
func SetMLXAvailableChecker(fn func() bool) {
	if fn == nil {
		mlxAvailable = func() bool { return false }
		return
	}
	mlxAvailable = fn
}

// PreferNativeCPU returns BackendNativeCPU when libwelvet is loaded, else BackendCPUTiled.
// Opt-in helper for callers that want C ABI when available; New() defaults stay
// BackendCPUTiled via DefaultBackend (set BackendNativeCPU explicitly to train natively).
func PreferNativeCPU() Backend {
	if nativeCPUAvailable() {
		return BackendNativeCPU
	}
	return BackendCPUTiled
}

// PreferQNN returns BackendQNN when libwelvet_qnn is loaded, else BackendCPUTiled.
func PreferQNN() Backend {
	if qnnAvailable() {
		return BackendQNN
	}
	return BackendCPUTiled
}

// PreferMLX returns BackendMLX when libwelvet_mlx + Metal GPU are ready, else BackendCPUTiled.
func PreferMLX() Backend {
	if mlxAvailable() {
		return BackendMLX
	}
	return BackendCPUTiled
}

// QNNAvailable reports whether the Hexagon NPU C ABI is linked.
func QNNAvailable() bool { return qnnAvailable() }

// MLXAvailable reports whether the Apple MLX GPU C ABI is ready.
func MLXAvailable() bool { return mlxAvailable() }

// DefaultBackend is always BackendCPUTiled. NativeCPU / QNN / MLX are opt-in
// (PreferNativeCPU / PreferQNN / PreferMLX / Exec.Backend = …).
func DefaultBackend() Backend { return BackendCPUTiled }
