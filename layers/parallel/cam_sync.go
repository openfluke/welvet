package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/layers/cnn1"
	"github.com/openfluke/welvet/layers/cnn2"
	"github.com/openfluke/welvet/layers/cnn3"
	"github.com/openfluke/welvet/layers/convt1"
	"github.com/openfluke/welvet/layers/convt2"
	"github.com/openfluke/welvet/layers/convt3"
	"github.com/openfluke/welvet/layers/dense"
	"github.com/openfluke/welvet/layers/embedding"
	"github.com/openfluke/welvet/layers/kmeans"
	"github.com/openfluke/welvet/layers/layernorm"
	"github.com/openfluke/welvet/layers/lstm"
	"github.com/openfluke/welvet/layers/mamba"
	"github.com/openfluke/welvet/layers/metacognition"
	"github.com/openfluke/welvet/layers/mha"
	"github.com/openfluke/welvet/layers/residual"
	"github.com/openfluke/welvet/layers/rmsnorm"
	"github.com/openfluke/welvet/layers/rnn"
	"github.com/openfluke/welvet/layers/sequential"
	"github.com/openfluke/welvet/layers/swiglu"
	"github.com/openfluke/welvet/weights"
)

// SyncWhen selects when CamSync runs relative to training.
type SyncWhen int

const (
	// SyncManual — host must call SyncNow / SyncCams explicitly.
	SyncManual SyncWhen = iota
	// SyncAfterSample — after each TrainStackMSE / TrainStackCE / TrainMSE sample.
	SyncAfterSample
	// SyncAfterStep — after each TrainStack / Train update (incl. Step* ticks).
	SyncAfterStep
	// SyncAfterPulse — after Pulse() (host marks a pulse / micro-batch boundary).
	SyncAfterPulse
)

// SyncEndpoint names one weight store on a Stack child.
// For a Parallel child, Branch selects the hemisphere; Store indexes
// CollectStores order within that branch (0 = primary Dense weights).
// StackIdx is the child index in Stack.Children.
type SyncEndpoint struct {
	StackIdx int
	Branch   int
	Store    int
}

// SyncPair is a bidirectional link: A and B pull toward their joint mean.
type SyncPair struct {
	A, B SyncEndpoint
}

// CamSyncConfig is inter-cameral (and optional cross-layer) weight averaging.
//
// Alpha is the bidirectional blend strength in (0,1]:
//
//	w_i ← (1-α)·w_i + α·mean(group)
//
// 0.01 = 1% pull, 1.0 = hard sync (all cams share the mean).
//
// BranchAlpha[i] overrides Alpha when writing cam i. α_i≤0 ⇒ one-way (contribute
// to mean, do not write) — asymmetric / teacher sync.
type CamSyncConfig struct {
	Enabled bool
	Alpha   float64  // default 1.0 when Enabled and Alpha≤0
	When    SyncWhen // default SyncManual

	// BranchAlpha optional per-cam write strength (len 0 ⇒ use Alpha for all).
	BranchAlpha []float64

	// Groups lists within-Parallel cam index cliques (into Layer.Branches).
	// Empty ⇒ one group of every branch when syncing a Parallel layer.
	// Example: {{0,1},{2,3}} syncs pairs; {{0,1,2}} syncs three cams.
	Groups [][]int

	// Cross pairs same-shaped stores across Stack children / cams
	// (e.g. cam0 of Parallel at child 1 ↔ cam1 of Parallel at child 1,
	//  or Dense stem ↔ matching cam Dense when Rows×Cols match).
	Cross []SyncPair
}

// DefaultCamSync returns full hard-average of all cams after each sample.
func DefaultCamSync() CamSyncConfig {
	return CamSyncConfig{
		Enabled: true,
		Alpha:   1.0,
		When:    SyncAfterSample,
	}
}

func (c *CamSyncConfig) alpha() float64 {
	if c == nil {
		return 0
	}
	a := c.Alpha
	if a <= 0 {
		a = 1
	}
	if a > 1 {
		a = 1
	}
	return a
}

func (c *CamSyncConfig) wants(when SyncWhen) bool {
	if c == nil || !c.Enabled {
		return false
	}
	return c.When == when
}

// SetCamSync attaches sync policy to a Parallel layer.
func (l *Layer) SetCamSync(cfg CamSyncConfig) {
	if l == nil {
		return
	}
	l.CamSync = &cfg
}

// SetCamSync attaches sync policy to a Stack (within nested Parallels + Cross).
func (s *Stack) SetCamSync(cfg CamSyncConfig) {
	if s == nil {
		return
	}
	s.CamSync = &cfg
	// Push When/Alpha/Groups onto nested Parallel children that have no cfg yet,
	// so within-layer cam groups fire from Stack.MaybeSync too.
	for _, ch := range s.Children {
		if p, ok := ch.(*Layer); ok && p != nil && p.CamSync == nil {
			cp := cfg
			cp.Cross = nil // Cross only resolved at Stack level
			p.CamSync = &cp
		}
	}
}

// SyncNow runs cam sync if Enabled, regardless of When (manual / diagnostics).
func (l *Layer) SyncNow() error {
	if l == nil {
		return nil
	}
	return syncParallelLayer(l, true)
}

// SyncNow runs Stack-level cam + cross sync if Enabled.
func (s *Stack) SyncNow() error {
	if s == nil {
		return nil
	}
	return syncStack(s, true)
}

// Pulse marks a pulse boundary; runs sync when When == SyncAfterPulse.
func (l *Layer) Pulse() error {
	return l.MaybeSync(SyncAfterPulse)
}

// Pulse marks a pulse boundary on the Stack.
func (s *Stack) Pulse() error {
	return s.MaybeSync(SyncAfterPulse)
}

// MaybeSync runs sync when cfg.When matches.
func (l *Layer) MaybeSync(when SyncWhen) error {
	if l == nil || l.CamSync == nil || !l.CamSync.wants(when) {
		return nil
	}
	return syncParallelLayer(l, false)
}

// MaybeSync runs Stack (+ nested Parallel) sync when cfg.When matches.
func (s *Stack) MaybeSync(when SyncWhen) error {
	if s == nil {
		return nil
	}
	if s.CamSync != nil && s.CamSync.wants(when) {
		return syncStack(s, false)
	}
	// Nested Parallel may have its own When.
	for _, ch := range s.Children {
		if p, ok := ch.(*Layer); ok && p != nil {
			if err := p.MaybeSync(when); err != nil {
				return err
			}
		}
		if nested, ok := ch.(*Stack); ok && nested != nil {
			if err := nested.MaybeSync(when); err != nil {
				return err
			}
		}
	}
	return nil
}

func syncParallelLayer(l *Layer, force bool) error {
	if l == nil {
		return nil
	}
	cfg := l.CamSync
	if cfg == nil || (!cfg.Enabled && !force) {
		return nil
	}
	if !cfg.Enabled && force {
		// SyncNow with disabled cfg still no-ops unless Enabled.
		return nil
	}
	alpha := cfg.alpha()
	groups := cfg.Groups
	if len(groups) == 0 {
		all := make([]int, len(l.Branches))
		for i := range all {
			all[i] = i
		}
		groups = [][]int{all}
	}
	for gi, g := range groups {
		live := filterTrainableBranchIdxs(l, g)
		if len(live) < 2 {
			continue
		}
		if err := blendBranchGroup(l, live, alpha); err != nil {
			return fmt.Errorf("parallel: CamSync group %d: %w", gi, err)
		}
	}
	return nil
}

// filterTrainableBranchIdxs drops Freeze BranchModes so CamSync cannot overwrite idle cams.
func filterTrainableBranchIdxs(l *Layer, idxs []int) []int {
	if l == nil {
		return idxs
	}
	out := make([]int, 0, len(idxs))
	for _, bi := range idxs {
		if bi < 0 || bi >= len(l.Branches) {
			continue
		}
		if l.EffectiveBranchMode(bi, ModeNormalBP).IsFrozen() {
			continue
		}
		out = append(out, bi)
	}
	return out
}

func syncStack(s *Stack, force bool) error {
	if s == nil {
		return nil
	}
	cfg := s.CamSync
	if cfg == nil || (!cfg.Enabled && !force) {
		return nil
	}
	if !cfg.Enabled {
		return nil
	}
	// 1) Within each nested Parallel (groups / all-cams).
	for i, ch := range s.Children {
		p, ok := ch.(*Layer)
		if !ok || p == nil {
			continue
		}
		if p.CamSync == nil {
			cp := *cfg
			cp.Cross = nil
			p.CamSync = &cp
		}
		if err := syncParallelLayer(p, force); err != nil {
			return fmt.Errorf("parallel: CamSync stack child %d: %w", i, err)
		}
	}
	// 2) Explicit cross-layer / cross-cam pairs (same shape only).
	alpha := cfg.alpha()
	for pi, pair := range cfg.Cross {
		sa, err := resolveEndpoint(s, pair.A)
		if err != nil {
			return fmt.Errorf("parallel: CamSync cross %d A: %w", pi, err)
		}
		sb, err := resolveEndpoint(s, pair.B)
		if err != nil {
			return fmt.Errorf("parallel: CamSync cross %d B: %w", pi, err)
		}
		if sa.Rows != sb.Rows || sa.Cols != sb.Cols {
			return fmt.Errorf("parallel: CamSync cross %d shape %dx%d vs %dx%d",
				pi, sa.Rows, sa.Cols, sb.Rows, sb.Cols)
		}
		if err := weights.BlendStores([]*weights.Store{sa, sb}, alpha); err != nil {
			return fmt.Errorf("parallel: CamSync cross %d: %w", pi, err)
		}
	}
	return nil
}

func blendBranchGroup(l *Layer, idxs []int, alpha float64) error {
	// Collect primary stores (slot 0) per branch; require matching shape.
	type slot struct {
		stores []*weights.Store
	}
	// Group by store index: sync slot 0 across cams, slot 1 across cams, …
	maxSlots := 0
	perBranch := make([][]*weights.Store, len(idxs))
	writeA := make([]float64, len(idxs))
	baseA := alpha
	for i, bi := range idxs {
		if bi < 0 || bi >= len(l.Branches) {
			return fmt.Errorf("branch index %d out of range", bi)
		}
		a := baseA
		if l.CamSync != nil && bi < len(l.CamSync.BranchAlpha) {
			a = l.CamSync.BranchAlpha[bi]
		}
		writeA[i] = a
		sts := opWeightStores(l.Branches[bi])
		perBranch[i] = sts
		if len(sts) > maxSlots {
			maxSlots = len(sts)
		}
	}
	for slot := 0; slot < maxSlots; slot++ {
		var clique []*weights.Store
		var alphas []float64
		var shapeR, shapeC int
		for i, sts := range perBranch {
			if slot >= len(sts) || sts[slot] == nil {
				continue
			}
			s := sts[slot]
			if len(clique) == 0 {
				shapeR, shapeC = s.Rows, s.Cols
			} else if s.Rows != shapeR || s.Cols != shapeC {
				clique = nil
				alphas = nil
				break
			}
			clique = append(clique, s)
			alphas = append(alphas, writeA[i])
		}
		if len(clique) >= 2 {
			if err := weights.BlendStoresWeighted(clique, alphas); err != nil {
				return err
			}
		}
	}
	return nil
}

func resolveEndpoint(s *Stack, e SyncEndpoint) (*weights.Store, error) {
	if s == nil {
		return nil, fmt.Errorf("nil stack")
	}
	if e.StackIdx < 0 || e.StackIdx >= len(s.Children) {
		return nil, fmt.Errorf("stack idx %d", e.StackIdx)
	}
	ch := s.Children[e.StackIdx]
	switch v := ch.(type) {
	case *Layer:
		if e.Branch < 0 || e.Branch >= len(v.Branches) {
			return nil, fmt.Errorf("branch %d", e.Branch)
		}
		sts := opWeightStores(v.Branches[e.Branch])
		if e.Store < 0 || e.Store >= len(sts) || sts[e.Store] == nil {
			return nil, fmt.Errorf("store slot %d", e.Store)
		}
		return sts[e.Store], nil
	case *dense.Layer:
		sts := opWeightStores(v)
		if e.Store < 0 || e.Store >= len(sts) || sts[e.Store] == nil {
			return nil, fmt.Errorf("store slot %d", e.Store)
		}
		return sts[e.Store], nil
	case *Stack:
		return resolveEndpoint(v, SyncEndpoint{StackIdx: e.Branch, Branch: e.Store, Store: 0})
	default:
		sts := opWeightStores(ch)
		if e.Store < 0 || e.Store >= len(sts) || sts[e.Store] == nil {
			return nil, fmt.Errorf("unsupported child %T store %d", ch, e.Store)
		}
		return sts[e.Store], nil
	}
}

// opWeightStores lists weight matrices for a branch Op (no dna import — avoid cycles).
// Must stay in sync with dna.CollectStores for every Parallel-legal Op that owns Stores.
// Softmax/GDN/View/Flatten: no *weights.Store (GDN uses quant.Blob — not blendable here).
func opWeightStores(op any) []*weights.Store {
	if op == nil {
		return nil
	}
	switch v := op.(type) {
	case *dense.Layer:
		return storesFromDense(v)
	case *mha.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.Q)...)
		out = append(out, storesFromDense(v.K)...)
		out = append(out, storesFromDense(v.V)...)
		out = append(out, storesFromDense(v.O)...)
		return out
	case *swiglu.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.Gate)...)
		out = append(out, storesFromDense(v.Up)...)
		out = append(out, storesFromDense(v.Down)...)
		return out
	case *rmsnorm.Layer:
		if v.Gamma != nil {
			return []*weights.Store{v.Gamma}
		}
	case *layernorm.Layer:
		var out []*weights.Store
		if v.Gamma != nil {
			out = append(out, v.Gamma)
		}
		if v.Beta != nil {
			out = append(out, v.Beta)
		}
		return out
	case *cnn1.Layer:
		return storesFromDense(v.Proj)
	case *cnn2.Layer:
		return storesFromDense(v.Proj)
	case *cnn3.Layer:
		return storesFromDense(v.Proj)
	case *convt1.Layer:
		return storesFromDense(v.Proj)
	case *convt2.Layer:
		return storesFromDense(v.Proj)
	case *convt3.Layer:
		return storesFromDense(v.Proj)
	case *rnn.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.IH)...)
		out = append(out, storesFromDense(v.HH)...)
		return out
	case *lstm.Layer:
		var out []*weights.Store
		for _, g := range []*lstm.Gate{v.I, v.F, v.G, v.O} {
			if g == nil {
				continue
			}
			out = append(out, storesFromDense(g.IH)...)
			out = append(out, storesFromDense(g.HH)...)
		}
		return out
	case *embedding.Layer:
		if v.Weights != nil {
			return []*weights.Store{v.Weights}
		}
	case *sequential.Layer:
		var out []*weights.Store
		for _, ch := range v.ChildOps() {
			out = append(out, opWeightStores(ch)...)
		}
		return out
	case *residual.Layer:
		var out []*weights.Store
		for _, ch := range v.ChildOps() {
			out = append(out, opWeightStores(ch)...)
		}
		return out
	case *ResidualSkip:
		return opWeightStores(v.F)
	case *kmeans.Layer:
		return storesFromDense(v.Centers)
	case *mamba.Layer:
		var out []*weights.Store
		out = append(out, storesFromDense(v.InProj)...)
		out = append(out, storesFromDense(v.OutProj)...)
		return out
	case *metacognition.Layer:
		return storesFromDense(v.Observed)
	case *Layer:
		var out []*weights.Store
		for _, ch := range v.Branches {
			out = append(out, opWeightStores(ch)...)
		}
		out = append(out, storesFromDense(v.Gate)...)
		return out
	case *Stack:
		var out []*weights.Store
		for _, ch := range v.Children {
			out = append(out, opWeightStores(ch)...)
		}
		return out
	}
	return nil
}

func storesFromDense(d *dense.Layer) []*weights.Store {
	if d == nil || d.Weights == nil {
		return nil
	}
	return []*weights.Store{d.Weights}
}
