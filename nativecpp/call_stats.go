package nativecpp

import (
	"sync"
	"sync/atomic"
)

var completedCalls atomic.Uint64
var namedCalls struct {
	sync.Mutex
	counts map[string]uint64
}

// CompletedCalls counts successful C ABI calls checked by the Go binding.
// Counters are process-wide and monotonic; take snapshots around a workload.
func CompletedCalls() uint64 { return completedCalls.Load() }

// CompletedCallsFor counts successful calls to an exact ABI operation name.
// This distinguishes a full-layer call from Go orchestration calling Dense.
func CompletedCallsFor(name string) uint64 {
	namedCalls.Lock()
	defer namedCalls.Unlock()
	return namedCalls.counts[name]
}
func recordCompleted(name string) {
	completedCalls.Add(1)
	namedCalls.Lock()
	defer namedCalls.Unlock()
	if namedCalls.counts == nil {
		namedCalls.counts = make(map[string]uint64)
	}
	namedCalls.counts[name]++
}
