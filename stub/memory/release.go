package memory

import (
	"runtime"
	"runtime/debug"
	"sync"
)

var scavengerOnce sync.Once

// InitScavenger warms the scavenger path (call once at process bootstrap).
func InitScavenger() {
	scavengerOnce.Do(func() {})
}

// InitMemoryScavenger is the loom alias.
func InitMemoryScavenger() { InitScavenger() }

// ReleaseTransient forces GC and returns unused heap pages to the OS.
func ReleaseTransient() {
	runtime.GC()
	debug.FreeOSMemory()
}

// ReleaseInferenceTransientMemory is the loom alias.
func ReleaseInferenceTransientMemory() { ReleaseTransient() }

// ReleaseAggressive runs an extra GC/scavenge pass.
func ReleaseAggressive() {
	runtime.GC()
	debug.FreeOSMemory()
	runtime.GC()
	debug.FreeOSMemory()
}

// AggressiveReleaseMemoryToOS is the loom alias.
func AggressiveReleaseMemoryToOS() { ReleaseAggressive() }
