package weights

import (
	"fmt"
	"math"
)

// BlendStores pulls every store toward the clique mean (bidirectional sync).
//
//	w_i ← (1-α)·w_i + α·mean
//
// alpha is clamped to (0,1]: 0.01 = gentle 1% pull, 1.0 = hard average.
// All stores must share the same Rows×Cols. Matching Bias lengths are blended too.
func BlendStores(stores []*Store, alpha float64) error {
	alphas := make([]float64, len(stores))
	for i := range alphas {
		alphas[i] = alpha
	}
	return BlendStoresWeighted(stores, alphas)
}

// BlendStoresWeighted is asymmetric CamSync: each store i is written with its own α_i.
// α_i ≤ 0 skips writing that store (one-way: it still contributes to the mean).
// α_i > 1 is clamped to 1.
func BlendStoresWeighted(stores []*Store, alphas []float64) error {
	if len(stores) < 2 {
		return nil
	}
	if len(alphas) != len(stores) {
		return fmt.Errorf("weights: BlendStoresWeighted len alphas %d vs stores %d", len(alphas), len(stores))
	}
	var ref *Store
	for _, s := range stores {
		if s == nil {
			return fmt.Errorf("weights: BlendStores nil store")
		}
		if ref == nil {
			ref = s
			continue
		}
		if s.Rows != ref.Rows || s.Cols != ref.Cols {
			return fmt.Errorf("weights: BlendStores shape mismatch %dx%d vs %dx%d",
				s.Rows, s.Cols, ref.Rows, ref.Cols)
		}
	}
	n := ref.Rows * ref.Cols
	if n <= 0 {
		return nil
	}

	flats := make([][]float32, len(stores))
	for i, s := range stores {
		v, err := s.FlattenF32()
		if err != nil {
			return fmt.Errorf("weights: BlendStores flatten %d: %w", i, err)
		}
		if len(v) < n {
			return fmt.Errorf("weights: BlendStores short flatten %d", i)
		}
		flats[i] = v[:n]
	}

	mean := make([]float32, n)
	inv := 1.0 / float64(len(flats))
	for _, v := range flats {
		for j, x := range v {
			mean[j] += float32(float64(x) * inv)
		}
	}

	anyWrite := false
	for i, s := range stores {
		a := alphas[i]
		if a <= 0 {
			continue
		}
		if a > 1 {
			a = 1
		}
		anyWrite = true
		af := float32(a)
		oma := float32(1 - a)
		out := make([]float32, n)
		for j := 0; j < n; j++ {
			out[j] = oma*flats[i][j] + af*mean[j]
		}
		if err := s.SetFromF32(out); err != nil {
			return fmt.Errorf("weights: BlendStores set %d: %w", i, err)
		}
	}
	if !anyWrite {
		return nil
	}

	biasLen := -1
	for _, s := range stores {
		if s.Bias == nil {
			biasLen = -2
			break
		}
		if biasLen < 0 {
			biasLen = len(s.Bias)
		} else if len(s.Bias) != biasLen {
			biasLen = -2
			break
		}
	}
	if biasLen > 0 {
		bMean := make([]float64, biasLen)
		invB := 1.0 / float64(len(stores))
		for _, s := range stores {
			for j, x := range s.Bias {
				bMean[j] += x * invB
			}
		}
		for i, s := range stores {
			a := alphas[i]
			if a <= 0 {
				continue
			}
			if a > 1 {
				a = 1
			}
			for j := range s.Bias {
				s.Bias[j] = (1-a)*s.Bias[j] + a*bMean[j]
			}
		}
	}
	return nil
}

// StoreCosine is cosine similarity of two flattened weight matrices (0..1-ish).
func StoreCosine(a, b *Store) (float64, error) {
	if a == nil || b == nil {
		return 0, fmt.Errorf("weights: StoreCosine nil")
	}
	fa, err := a.FlattenF32()
	if err != nil {
		return 0, err
	}
	fb, err := b.FlattenF32()
	if err != nil {
		return 0, err
	}
	n := len(fa)
	if n == 0 || n != len(fb) {
		return 0, fmt.Errorf("weights: StoreCosine len %d vs %d", len(fa), len(fb))
	}
	var dot, na, nb float64
	for i := 0; i < n; i++ {
		x, y := float64(fa[i]), float64(fb[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na < 1e-30 || nb < 1e-30 {
		return 0, nil
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb)), nil
}

// StoreFrobenius is √Σ w² of a flattened store (plasticity / ΔW meters).
func StoreFrobenius(s *Store) (float64, error) {
	if s == nil {
		return 0, fmt.Errorf("weights: StoreFrobenius nil")
	}
	v, err := s.FlattenF32()
	if err != nil {
		return 0, err
	}
	var sum float64
	for _, x := range v {
		f := float64(x)
		sum += f * f
	}
	return math.Sqrt(sum), nil
}
