package tween

import (
	"fmt"

	"github.com/openfluke/welvet/simd"
	"github.com/openfluke/welvet/weights"
)

// applyStoreHebbian updates one weight store from input×gap (Dense out×in layout).
// Outer-product rows use Plan-9 SaxpyF32AccF64 (same primitive as dense.BackwardSIMD dW).
func applyStoreHebbian(s *weights.Store, input, gap []float32, layerRate, mom float32, vel *[]float32) error {
	if s == nil || len(gap) == 0 {
		return nil
	}
	w, err := s.FlattenF32()
	if err != nil {
		return err
	}
	outSize, inSize := s.Rows, s.Cols
	need := outSize*inSize + outSize
	if vel == nil {
		tmp := make([]float32, need)
		vel = &tmp
	}
	if len(*vel) != need {
		*vel = make([]float32, need)
	}
	v := *vel

	if len(input) >= inSize && len(gap) >= outSize && inSize > 0 && outSize > 0 {
		xin := input
		if len(input) != inSize {
			xin = make([]float32, inSize)
			for in := 0; in < inSize; in++ {
				xin[in] = input[in%len(input)]
			}
		}
		dW := make([]float64, outSize*inSize)
		for out := 0; out < outSize; out++ {
			alpha := float64(layerRate * gap[out])
			if alpha == 0 {
				continue
			}
			simd.SaxpyF32AccF64(dW[out*inSize:(out+1)*inSize], alpha, xin, inSize)
		}
		for i := 0; i < outSize*inSize && i < len(w); i++ {
			delta := float32(dW[i])
			v[i] = mom*v[i] + (1-mom)*delta
			w[i] += v[i]
		}
		for out := 0; out < outSize; out++ {
			if len(s.Bias) <= out {
				break
			}
			bIdx := outSize*inSize + out
			delta := layerRate * gap[out]
			v[bIdx] = mom*v[bIdx] + (1-mom)*delta
			s.Bias[out] += float64(v[bIdx])
		}
		return s.SetFromF32(w)
	}

	// Fallback: scale each weight by mean gap (norms / odd shapes).
	var mean float32
	for _, g := range gap {
		mean += g
	}
	mean /= float32(len(gap))
	for i := range w {
		delta := layerRate * mean * 0.01
		if i < len(v) {
			v[i] = mom*v[i] + (1-mom)*delta
			w[i] += v[i]
		} else {
			w[i] += delta
		}
	}
	return s.SetFromF32(w)
}

// hebbianImportances computes importance = Wᵀ @ target and L1 column norms via Saxpy/DotTile.
// W is row-major [out×in]. Used by layerwise target propagation.
func hebbianImportances(w []float32, target []float32, outSize, inSize int) (importance, totalWeight []float32, err error) {
	if outSize <= 0 || inSize <= 0 {
		return nil, nil, fmt.Errorf("tween: bad hebbian shape %dx%d", outSize, inSize)
	}
	if len(w) < outSize*inSize {
		return nil, nil, fmt.Errorf("tween: weight len %d < %d", len(w), outSize*inSize)
	}
	imp64 := make([]float64, inSize)
	l1 := make([]float64, inSize)
	nOut := outSize
	if len(target) < nOut {
		nOut = len(target)
	}
	for out := 0; out < nOut; out++ {
		row := w[out*inSize : (out+1)*inSize]
		simd.SaxpyF32AccF64(imp64, float64(target[out]), row, inSize)
		// Column L1 — DotTile against a sign-free abs pass is awkward; tight scalar is fine.
		for in := 0; in < inSize; in++ {
			ww := row[in]
			if ww < 0 {
				l1[in] -= float64(ww)
			} else {
				l1[in] += float64(ww)
			}
		}
	}
	importance = make([]float32, inSize)
	totalWeight = make([]float32, inSize)
	for i := 0; i < inSize; i++ {
		importance[i] = float32(imp64[i])
		totalWeight[i] = float32(l1[i])
	}
	return importance, totalWeight, nil
}
