// Package hardware probes host OS/CPU/RAM/GPU/disk/network (loom hardware_*).
//
// Audit() is pure Go /proc+sysfs on Linux; other GOOS use fallbacks.
// Tests live in github.com/openfluke/w2a — not here.
package hardware
