package core

// nativeCPUAvailable reports whether welvet.cpp is loaded for NativeCPU dispatch.
// nativecpp registers the real checker from init(); default is false (pure Go).
var nativeCPUAvailable = func() bool { return false }

// SetNativeCPUAvailableChecker lets nativecpp register libwelvet availability.
// Pass nil to reset to "unavailable".
func SetNativeCPUAvailableChecker(fn func() bool) {
	if fn == nil {
		nativeCPUAvailable = func() bool { return false }
		return
	}
	nativeCPUAvailable = fn
}

// PreferNativeCPU returns BackendNativeCPU when libwelvet is loaded, else BackendCPUTiled.
// Layers that call this for New() defaults get C ABI acceleration without manual config;
// when the platform lib is missing, execution stays pure Go.
func PreferNativeCPU() Backend {
	if nativeCPUAvailable() {
		return BackendNativeCPU
	}
	return BackendCPUTiled
}

// DefaultBackend is an alias of PreferNativeCPU for ExecConfig defaults.
func DefaultBackend() Backend { return PreferNativeCPU() }
