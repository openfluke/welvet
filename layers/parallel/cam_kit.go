package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/weights"
)

// CamKit bundles the "weird but useful" cameral controls that sit beside BranchModes.
// Attach with SetCamKit; nil kit ⇒ defaults (off).
type CamKit struct {
	// ShadowCoef is the KD / consistency weight when any cam uses ModeShadow.
	// Student grads get +coef·(student−teacher)/n. 0 ⇒ 1.0 when Shadow cams exist.
	ShadowCoef float64

	// DNAReg soft weight regularizer after each sample sync window:
	//   >0 diversify (push away from clique mean), <0 attract (weak sync), 0 off.
	// Magnitude is |α| for BlendStoresWeighted.
	DNAReg float64

	// SurpriseThresh gates ModeMemory: cam trains only when LastLoss ≥ thresh.
	// 0 ⇒ always train Memory cams (thresh disabled).
	SurpriseThresh float64

	// LastLoss is updated by TrainMSE / hosts that call NoteLoss.
	LastLoss float64

	// Dream is an optional replay buffer for DreamPulse consolidation ticks.
	Dream *DreamBuffer

	// Plasticity[i] is ‖ΔW‖ of branch i after the last train (Frobenius of flat Δ).
	Plasticity []float64

	// LastMetrics is filled by RefreshMetrics (cosines + plasticity snapshot).
	LastMetrics CamMetrics
}

// CamMetrics is a River-friendly per-cam diagnostic snapshot.
type CamMetrics struct {
	BranchCosines []float64 `json:"branch_cosines,omitempty"` // pairwise vs cam0 (len=nb)
	Plasticity    []float64 `json:"plasticity,omitempty"`
	DNAReg        float64   `json:"dna_reg,omitempty"`
	LastLoss      float64   `json:"last_loss,omitempty"`
	ActiveModes   []string  `json:"active_modes,omitempty"`
}

// DreamBuffer stores recent (x,y) flats for offline DreamPulse replay.
type DreamBuffer struct {
	Cap    int
	xShape []int
	yShape []int
	xs     [][]float32
	ys     [][]float32
}

// RotateSchedule cycles BranchModes every Period samples (sleep / plasticity rotate).
type RotateSchedule struct {
	Slots  [][]TrainMode // each slot is a full BranchModes stamp
	Period int           // samples per slot (≤0 ⇒ 1)
	tick   int
	slot   int
}

// SetCamKit attaches kit policy to a Parallel layer.
func (l *Layer) SetCamKit(k CamKit) {
	if l == nil {
		return
	}
	cp := k
	l.CamKit = &cp
}

// SetBranchLRs sets per-cam LR multipliers (empty ⇒ 1.0 each). 0 ⇒ skip update (soft freeze).
func (l *Layer) SetBranchLRs(scales ...float64) {
	if l == nil {
		return
	}
	l.BranchLRs = append([]float64(nil), scales...)
}

// EffectiveBranchLR returns parentLR × BranchLRs[i] (missing ⇒ 1).
func (l *Layer) EffectiveBranchLR(i int, parentLR float64) float64 {
	if l == nil || i < 0 {
		return parentLR
	}
	if i >= len(l.BranchLRs) {
		return parentLR
	}
	return parentLR * l.BranchLRs[i]
}

// SetRotateSchedule installs Mode rotation; ApplyRotateSlot stamps current slot.
func (l *Layer) SetRotateSchedule(slots [][]TrainMode, period int) {
	if l == nil || len(slots) == 0 {
		return
	}
	l.Rotate = &RotateSchedule{Slots: slots, Period: period}
	l.ApplyRotateSlot()
}

// ApplyRotateSlot stamps BranchModes from the current rotate slot.
func (l *Layer) ApplyRotateSlot() {
	if l == nil || l.Rotate == nil || len(l.Rotate.Slots) == 0 {
		return
	}
	s := l.Rotate.Slots[l.Rotate.slot%len(l.Rotate.Slots)]
	l.SetBranchModes(s...)
}

// AdvanceRotate bumps the sample counter; flips slot every Period samples.
func (l *Layer) AdvanceRotate() {
	if l == nil || l.Rotate == nil || len(l.Rotate.Slots) == 0 {
		return
	}
	r := l.Rotate
	per := r.Period
	if per <= 0 {
		per = 1
	}
	r.tick++
	if r.tick%per == 0 {
		r.slot = (r.slot + 1) % len(r.Slots)
		l.ApplyRotateSlot()
	}
}

// NoteLoss records loss for ModeMemory / metrics.
func (l *Layer) NoteLoss(loss float64) {
	if l == nil {
		return
	}
	if l.CamKit == nil {
		l.CamKit = &CamKit{}
	}
	l.CamKit.LastLoss = loss
}

// Remember pushes a sample into the dream buffer (no-op if Dream unset).
func (l *Layer) Remember(x, y *core.Tensor[float32]) {
	if l == nil || l.CamKit == nil || l.CamKit.Dream == nil || x == nil || y == nil {
		return
	}
	l.CamKit.Dream.push(x, y)
}

// DreamPulse replays up to n buffered samples with TrainMSE (consolidation).
// Also fires SyncAfterPulse.
func (l *Layer) DreamPulse(n int, mode TrainMode, lr float64) (avgLoss float64, err error) {
	if l == nil {
		return 0, fmt.Errorf("parallel: DreamPulse nil")
	}
	if l.CamKit == nil || l.CamKit.Dream == nil || l.CamKit.Dream.len() == 0 {
		_ = l.Pulse()
		return 0, nil
	}
	d := l.CamKit.Dream
	if n <= 0 || n > d.len() {
		n = d.len()
	}
	var sum float64
	for i := 0; i < n; i++ {
		x, y := d.at(i)
		loss, errT := TrainMSE(l, x, y, mode, lr)
		if errT != nil {
			return sum / float64(i+1), errT
		}
		sum += loss
	}
	_ = l.Pulse()
	return sum / float64(n), nil
}

// RefreshMetrics fills LastMetrics (pairwise cos vs cam0 + plasticity + modes).
func (l *Layer) RefreshMetrics() CamMetrics {
	m := CamMetrics{}
	if l == nil {
		return m
	}
	nb := len(l.Branches)
	m.ActiveModes = make([]string, nb)
	for i := 0; i < nb; i++ {
		m.ActiveModes[i] = l.EffectiveBranchMode(i, ModeNormalBP).String()
	}
	m.BranchCosines = make([]float64, nb)
	var ref *weights.Store
	sts0 := opWeightStores(l.Branches[0])
	if len(sts0) > 0 {
		ref = sts0[0]
	}
	for i := 0; i < nb; i++ {
		if i == 0 {
			m.BranchCosines[i] = 1
			continue
		}
		sts := opWeightStores(l.Branches[i])
		if ref == nil || len(sts) == 0 || sts[0] == nil {
			continue
		}
		c, err := weights.StoreCosine(ref, sts[0])
		if err == nil {
			m.BranchCosines[i] = c
		}
	}
	if l.CamKit != nil {
		m.Plasticity = append([]float64(nil), l.CamKit.Plasticity...)
		m.DNAReg = l.CamKit.DNAReg
		m.LastLoss = l.CamKit.LastLoss
		l.CamKit.LastMetrics = m
	}
	return m
}

func (d *DreamBuffer) push(x, y *core.Tensor[float32]) {
	if d.Cap <= 0 {
		d.Cap = 64
	}
	xf := append([]float32(nil), x.Data...)
	yf := append([]float32(nil), y.Data...)
	if len(d.xs) == 0 {
		d.xShape = append([]int(nil), x.Shape...)
		d.yShape = append([]int(nil), y.Shape...)
	}
	if len(d.xs) >= d.Cap {
		d.xs = d.xs[1:]
		d.ys = d.ys[1:]
	}
	d.xs = append(d.xs, xf)
	d.ys = append(d.ys, yf)
}

func (d *DreamBuffer) len() int {
	if d == nil {
		return 0
	}
	return len(d.xs)
}

func (d *DreamBuffer) at(i int) (x, y *core.Tensor[float32]) {
	x = core.NewTensor[float32](d.xShape...)
	y = core.NewTensor[float32](d.yShape...)
	copy(x.Data, d.xs[i])
	copy(y.Data, d.ys[i])
	return x, y
}

func (k *CamKit) shadowCoef() float64 {
	if k == nil || k.ShadowCoef == 0 {
		return 1
	}
	return k.ShadowCoef
}

func (l *Layer) memoryAllowsTrain() bool {
	if l == nil || l.CamKit == nil {
		return true
	}
	th := l.CamKit.SurpriseThresh
	if th <= 0 {
		return true
	}
	return l.CamKit.LastLoss >= th
}

// applyDNAReg soft-pushes cams using BlendStoresWeighted (|DNAReg| as α).
// Positive DNAReg diversifies (invert pull: write with negative sense via 1-way away).
func (l *Layer) applyDNAReg() error {
	if l == nil || l.CamKit == nil || l.CamKit.DNAReg == 0 {
		return nil
	}
	reg := l.CamKit.DNAReg
	alpha := reg
	if alpha < 0 {
		alpha = -alpha
	}
	if alpha > 1 {
		alpha = 1
	}
	nb := len(l.Branches)
	// Collect primary stores.
	stores := make([]*weights.Store, 0, nb)
	idxs := make([]int, 0, nb)
	for i := 0; i < nb; i++ {
		if l.EffectiveBranchMode(i, ModeNormalBP).IsFrozen() {
			continue
		}
		sts := opWeightStores(l.Branches[i])
		if len(sts) == 0 || sts[0] == nil {
			continue
		}
		stores = append(stores, sts[0])
		idxs = append(idxs, i)
	}
	if len(stores) < 2 {
		return nil
	}
	if reg < 0 {
		// Attract toward mean (weak sync).
		alphas := make([]float64, len(stores))
		for i := range alphas {
			alphas[i] = alpha
		}
		return weights.BlendStoresWeighted(stores, alphas)
	}
	// Diversify: w ← w + α·(w − mean) = (1+α)w − α·mean, clamp via two-step:
	// first blend toward mean with tiny a, then reflect — simpler: w ← (1+α)w − α·mean.
	flats := make([][]float32, len(stores))
	n := stores[0].Rows * stores[0].Cols
	for i, s := range stores {
		v, err := s.FlattenF32()
		if err != nil {
			return err
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
	a := float32(alpha)
	for i, s := range stores {
		out := make([]float32, n)
		for j := 0; j < n; j++ {
			out[j] = flats[i][j] + a*(flats[i][j]-mean[j])
		}
		if err := s.SetFromF32(out); err != nil {
			return err
		}
	}
	_ = idxs
	return nil
}

func snapshotBranchNorms(l *Layer) []float64 {
	if l == nil {
		return nil
	}
	out := make([]float64, len(l.Branches))
	for i, ch := range l.Branches {
		sts := opWeightStores(ch)
		if len(sts) == 0 || sts[0] == nil {
			continue
		}
		n, err := weights.StoreFrobenius(sts[0])
		if err == nil {
			out[i] = n
		}
	}
	return out
}

func recordPlasticity(l *Layer, before []float64) {
	if l == nil {
		return
	}
	after := snapshotBranchNorms(l)
	delta := make([]float64, len(after))
	for i := range after {
		d := after[i]
		if i < len(before) {
			d = after[i] - before[i]
			if d < 0 {
				d = -d
			}
		}
		delta[i] = d
	}
	if l.CamKit == nil {
		l.CamKit = &CamKit{}
	}
	l.CamKit.Plasticity = delta
}
