package transformer

import (
	"github.com/openfluke/welvet/core"
)

type fwdScratch struct {
	h *core.Tensor[float32]
}

func (m *Model) ensureScratch(shape []int, minLen int) *fwdScratch {
	if m.scratch == nil {
		m.scratch = &fwdScratch{}
	}
	if m.scratch.h == nil || len(m.scratch.h.Data) < minLen || !sameShape(m.scratch.h.Shape, shape) {
		m.scratch.h = core.NewTensor[float32](shape...)
	}
	return m.scratch
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func residualAddInPlace(dst *core.Tensor[float32], add *core.Tensor[float32]) {
	n := len(dst.Data)
	if len(add.Data) < n {
		n = len(add.Data)
	}
	for i := 0; i < n; i++ {
		dst.Data[i] += add.Data[i]
	}
}
