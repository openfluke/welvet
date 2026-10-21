package core

// nativeCPUAvailable reports whether welvet.cpp is loaded for NativeCPU dispatch.
var nativeCPUAvailable func() bool

// SetNativeCPUAvailableChecker lets nativecpp register libwelvet availability.
func SetNativeCPUAvailableChecker(fn func() bool) {
	nativeCPUAvailable = fn
}

// NativeLibLoaded reports whether the welvet.cpp shared library is available.
func NativeLibLoaded() bool {
	return nativeCPUAvailable != nil && nativeCPUAvailable()
}

// PreferNativeCPU returns BackendNativeCPU when libwelvet is loaded, else BackendCPUTiled.
// Opt-in helper — not used as the layer New() default.
func PreferNativeCPU() Backend {
	if NativeLibLoaded() {
		return BackendNativeCPU
	}
	return BackendCPUTiled
}

// DefaultBackend is always Go CPU tiled. Set Exec.Backend = BackendNativeCPU explicitly
// (or PreferNativeCPU()) to route through welvet.cpp.
func DefaultBackend() Backend { return BackendCPUTiled }
