package quant

import "fmt"

func errShape(op string, rows, cols, n int) error {
	return fmt.Errorf("%s: bad shape rows=%d cols=%d len=%d", op, rows, cols, n)
}

func errFormat(op string, b *Blob) error {
	if b == nil {
		return fmt.Errorf("%s: nil blob", op)
	}
	return fmt.Errorf("%s: expected format mismatch got %s", op, b.Format)
}

// ErrUnsupported is returned when a Format has no native kernel yet.
func ErrUnsupported(f Format, op string) error {
	return fmt.Errorf("quant: %s not implemented for %s (native path TBD — no QAT fallback)", op, f)
}
