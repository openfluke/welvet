//go:build !darwin && !linux && !freebsd && !openbsd

package memory

func processRSSBytes() uint64 {
	return 0
}
