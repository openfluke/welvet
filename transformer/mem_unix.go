//go:build unix && !linux

package transformer

import "syscall"

func processRSSBytes() uint64 {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	// Darwin / BSD: Maxrss is bytes (peak). Best available without Mach APIs.
	return uint64(ru.Maxrss)
}
