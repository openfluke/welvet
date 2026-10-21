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
// Opt-in helper for callers that want C ABI when available; New() defaults stay
// BackendCPUTiled via DefaultBackend (set BackendNativeCPU explicitly to train natively).
func PreferNativeCPU() Backend {
	if nativeCPUAvailable() {
		return BackendNativeCPU
	}
	return BackendCPUTiled
}

// DefaultBackend is always BackendCPUTiled. NativeCPU is opt-in (PreferNativeCPU /
// Exec.Backend = BackendNativeCPU).
func DefaultBackend() Backend { return BackendCPUTiled }
