package lucy

import (
	"math"
	"sort"
)

// Consciousness / synthetic-organism metric: Acc, Throughput, Availability.
// Lucy Score = T × Avail × Acc / 10,000. SoftAcc is serve-confidence, not Score.
// Memory density condenses the three pillars vs Acc-champ RAM. Tiny dtypes at
// chance Acc are traps (LPD = 0). SGD that blocks serve dies on Availability.
const (
	LPDKeepFloor = 0.70
	LPDGoldKeep  = 0.80
	LPDLeanKeep  = 0.95 // Acc keep floor for "lean" champs (sacrifice peak Acc for RAM/Thru/Avail)
	LPDGoldRAM   = 0.20
	LPDNearRAM   = 0.50
	LPDShrinkCap = 32.0
)

// Sample is one finished permutation for density ranking. Hosts map their own
// IDs onto these fields; this package does not parse cell strings.
type Sample struct {
	Tide   string  `json:"tide,omitempty"`
	ID     string  `json:"id"`
	Mode   string  `json:"mode"`
	DType  string  `json:"dtype"`
	Format string  `json:"format"`
	Arch   string  `json:"arch"`
	Score  float64 `json:"score"`
	Soft   float64 `json:"soft_acc"`
	Acc    float64 `json:"avg_accuracy"`
	Thru   float64 `json:"throughput"`
	Avail  float64 `json:"availability"`
	RAMKiB float64 `json:"ram_kib"`
}

// LPDChamp is a named reference cell (Score champ, Acc champ, live-fit champ).
type LPDChamp struct {
	ID     string  `json:"id"`
	Tide   string  `json:"tide,omitempty"`
	Mode   string  `json:"mode"`
	DType  string  `json:"dtype"`
	Arch   string  `json:"arch"`
	Score  float64 `json:"score"`
	Soft   float64 `json:"soft_acc"`
	Acc    float64 `json:"avg_accuracy"`
	Thru   float64 `json:"throughput"`
	Avail  float64 `json:"availability"`
	RAMKiB float64 `json:"ram_kib"`
}

// LPDRow is one cell against the three pillars and Acc-champ RAM.
type LPDRow struct {
	Tide      string  `json:"tide,omitempty"`
	ID        string  `json:"id"`
	Mode      string  `json:"mode"`
	DType     string  `json:"dtype"`
	Format    string  `json:"format"`
	Arch      string  `json:"arch"`
	Score     float64 `json:"score"`
	Soft      float64 `json:"soft_acc"`
	Acc       float64 `json:"avg_accuracy"`
	Thru      float64 `json:"throughput"`
	Avail     float64 `json:"availability"`
	RAMKiB    float64 `json:"ram_kib"`
	RelScore  float64 `json:"rel_score"`
	RelSoft   float64 `json:"rel_soft"`
	RelAcc    float64 `json:"rel_acc"`
	RelThru   float64 `json:"rel_thru"`
	RelAvail  float64 `json:"rel_avail"`
	Q         float64 `json:"q"`        // geomean of Acc/Thru/Avail keep vs learner peaks
	RAMFrac   float64 `json:"ram_frac"` // this / Acc-champ RAM
	Shrink    float64 `json:"shrink"`   // Acc-champ RAM / this (× smaller)
	LPD       float64 `json:"lpd"`      // Q × shrink; 0 unless Acc keep ≥ 70%
	RelFast   float64 `json:"rel_fast"` // Thru / board fastest (trap can win)
	RelDuty   float64 `json:"rel_duty"` // Avail / board best (trap can win)
	MSpeed    float64 `json:"mspeed"`   // RelThru if Acc keep, else 0
	MAvail    float64 `json:"mavail"`   // RelAvail if Acc keep, else 0
	Mix       float64 `json:"mix"`      // Q if Acc keep, else 0 (consciousness)
	DensAcc   float64 `json:"dens_acc"` // RelAcc × shrink (0 if trap)
	DensThru  float64 `json:"dens_thru"`
	DensAvail float64 `json:"dens_avail"`
	Pillars   int     `json:"pillars"` // how many of Acc/Thru/Avail ≥ 80%
	Band      string  `json:"band"`    // gold | near | keep | acc | trap | —
	Gold      bool    `json:"gold"`
}

// Consciousness is Acc/Thru/Avail keep vs learner peaks — the live radar, [0,1].
func (r LPDRow) Consciousness() [3]float64 {
	return [3]float64{r.RelAcc, r.RelThru, r.RelAvail}
}

// MemoryDensity is the same pillars × shrink vs Acc-champ RAM (traps sit at 0).
func (r LPDRow) MemoryDensity() [3]float64 {
	return [3]float64{r.DensAcc, r.DensThru, r.DensAvail}
}

// Sample returns the row's identity + raw pillars (for host heatmaps).
func (r LPDRow) Sample() Sample {
	return Sample{
		Tide: r.Tide, ID: r.ID, Mode: r.Mode, DType: r.DType, Format: r.Format, Arch: r.Arch,
		Score: r.Score, Soft: r.Soft, Acc: r.Acc, Thru: r.Thru, Avail: r.Avail, RAMKiB: r.RAMKiB,
	}
}

// LPDMode is one train mode among 2+ pillar (gold-std) cells.
type LPDMode struct {
	Mode     string  `json:"mode"`
	N        int     `json:"n"`
	BestAcc  float64 `json:"best_acc"`
	MinRAM   float64 `json:"min_ram_kib"`
	MaxThru  float64 `json:"max_thru"`
	BestQ    float64 `json:"best_q"`
	Smallest string  `json:"smallest"`
	Fastest  string  `json:"fastest"`
}

// LPD is the goldilocks snapshot for a board of samples.
type LPD struct {
	Formula       string    `json:"formula"`
	Champ         LPDChamp  `json:"champ"`      // Lucy Score champ
	AccChamp      LPDChamp  `json:"acc_champ"`  // highest hard Acc — RAM reference
	SoftChamp     LPDChamp  `json:"soft_champ"` // SoftAcc champ (display)
	LiveChamp     LPDChamp  `json:"live_champ"` // best consciousness Q among learners
	PeakScore     float64   `json:"peak_score"`
	PeakSoft      float64   `json:"peak_soft"`
	PeakAcc       float64   `json:"peak_acc"`
	PeakThru      float64   `json:"peak_thru"`  // fastest Thru among Acc-keepers
	PeakAvail     float64   `json:"peak_avail"` // best Avail among Acc-keepers
	FastThru      float64   `json:"fast_thru"`  // board fastest (trap can own this)
	FastID        string    `json:"fast_id,omitempty"`
	BestAvail     float64   `json:"best_avail"`
	AvailID       string    `json:"avail_id,omitempty"`
	PeakDensAcc   float64   `json:"peak_dens_acc"`
	PeakDensThru  float64   `json:"peak_dens_thru"`
	PeakDensAvail float64   `json:"peak_dens_avail"`
	N             int       `json:"n"`
	Gold          []LPDRow  `json:"gold,omitempty"`
	Near          []LPDRow  `json:"near,omitempty"`
	Top           []LPDRow  `json:"top,omitempty"`
	Pool          []LPDRow  `json:"-"`                  // full ranked rows for radars/scatters (not JSON)
	Lean          []LPDRow  `json:"lean,omitempty"`      // Acc keep ≥95%, then smallest RAM / fastest / avail
	LeanChamp     LPDRow    `json:"lean_champ,omitempty"`
	Trap          []LPDRow  `json:"trap,omitempty"`
	TopSpeed      []LPDRow  `json:"top_speed,omitempty"`
	TopAvail      []LPDRow  `json:"top_avail,omitempty"`
	TopMix        []LPDRow  `json:"top_mix,omitempty"`
	GoldStd       LPDRow    `json:"gold_std,omitempty"`
	GoldModes     []LPDMode `json:"gold_modes,omitempty"`
	LeanByArch    []LPDMode `json:"lean_by_arch,omitempty"` // cam / arch lean winners
}

// DensityFormula is the board legend (JSON + PDFs).
func DensityFormula() string {
	return "Synthetic organism / Lucy density: run+train in a small box. Hard Acc = argmax accuracy. Acc keep % = this Acc / Acc-champ Acc (1.0 = matches the best Acc). SoftAcc = serve-confidence (not Score). Score = Thru x Avail x HardAcc / 10,000. Q = geomean Acc-keep/Thru-keep/Avail-keep vs learner peaks. LPD = Q x shrink vs Acc-champ RAM; 0 unless Acc keep >=70%. Gold = all 3 pillars >=80% at <=20% Acc-champ RAM. Gold-std = Acc keep >=80% plus Thru or Avail, then smallest then fastest. Lean = Acc keep >=95% of Acc champ, then smallest RAM / fastest Thru / best Avail — sacrifice peak Acc only within that band."
}

// BuildLPD ranks samples for consciousness (Acc/Thru/Avail) then memory density.
func BuildLPD(pts []Sample) LPD {
	out := LPD{Formula: DensityFormula()}
	if len(pts) == 0 {
		return out
	}
	var champ, accChamp, softChamp Sample
	var fastThru, bestAvail float64
	var fastID, availID string
	for i, p := range pts {
		better := p.Score > champ.Score
		if p.Score == champ.Score && champ.ID != "" {
			if p.Acc > champ.Acc || (p.Acc == champ.Acc && p.Soft > champ.Soft) {
				better = true
			}
		}
		if i == 0 || better {
			champ = p
		}
		if i == 0 || p.Acc > accChamp.Acc || (p.Acc == accChamp.Acc && (p.Soft > accChamp.Soft || (p.Soft == accChamp.Soft && p.Score > accChamp.Score))) {
			accChamp = p
		}
		if i == 0 || p.Soft > softChamp.Soft || (p.Soft == softChamp.Soft && (p.Acc > softChamp.Acc || (p.Acc == softChamp.Acc && p.Score > softChamp.Score))) {
			softChamp = p
		}
		if i == 0 || p.Thru > fastThru {
			fastThru, fastID = p.Thru, p.ID
		}
		if i == 0 || p.Avail > bestAvail {
			bestAvail, availID = p.Avail, p.ID
		}
	}
	out.PeakScore, out.PeakSoft, out.PeakAcc = champ.Score, softChamp.Soft, accChamp.Acc
	out.FastThru, out.FastID = fastThru, fastID
	out.BestAvail, out.AvailID = bestAvail, availID
	liveThru, liveAvail := learnerPeaks(pts, accChamp.Acc)
	out.PeakThru, out.PeakAvail = liveThru, liveAvail
	if accChamp.RAMKiB <= 0 {
		accChamp.RAMKiB = 1e-6
	}
	out.Champ = lpdChampOf(champ)
	out.AccChamp = lpdChampOf(accChamp)
	out.SoftChamp = lpdChampOf(softChamp)
	out.N = len(pts)
	rows := make([]LPDRow, 0, len(pts))
	var live LPDRow
	for _, p := range pts {
		r := lpdRow(p, out)
		rows = append(rows, r)
		if r.RelAcc >= LPDKeepFloor && (live.ID == "" || r.Q > live.Q) {
			live = r
		}
		switch r.Band {
		case "gold":
			out.Gold = append(out.Gold, r)
		case "near":
			out.Near = append(out.Near, r)
		case "trap":
			out.Trap = append(out.Trap, r)
		}
		if r.DensAcc > out.PeakDensAcc {
			out.PeakDensAcc = r.DensAcc
		}
		if r.DensThru > out.PeakDensThru {
			out.PeakDensThru = r.DensThru
		}
		if r.DensAvail > out.PeakDensAvail {
			out.PeakDensAvail = r.DensAvail
		}
	}
	if live.ID != "" {
		out.LiveChamp = LPDChamp{
			ID: live.ID, Tide: live.Tide, Mode: live.Mode, DType: live.DType, Arch: live.Arch,
			Score: live.Score, Soft: live.Soft, Acc: live.Acc, Thru: live.Thru, Avail: live.Avail, RAMKiB: live.RAMKiB,
		}
	}
	sort.SliceStable(rows, func(i, j int) bool {
		if rows[i].LPD != rows[j].LPD {
			return rows[i].LPD > rows[j].LPD
		}
		if rows[i].Q != rows[j].Q {
			return rows[i].Q > rows[j].Q
		}
		return rows[i].RAMKiB < rows[j].RAMKiB
	})
	sort.SliceStable(out.Gold, func(i, j int) bool { return out.Gold[i].LPD > out.Gold[j].LPD })
	sort.SliceStable(out.Near, func(i, j int) bool { return out.Near[i].LPD > out.Near[j].LPD })
	sort.SliceStable(out.Trap, func(i, j int) bool { return out.Trap[i].RAMFrac < out.Trap[j].RAMFrac })
	if len(out.Gold) > 24 {
		out.Gold = out.Gold[:24]
	}
	if len(out.Near) > 16 {
		out.Near = out.Near[:16]
	}
	if len(out.Trap) > 12 {
		out.Trap = out.Trap[:12]
	}
	out.Pool = rows // full set for radars/scatters (json:"-" — not shipped on /api/live)
	out.Top = rows
	if len(out.Top) > 40 {
		out.Top = out.Top[:40]
	}
	out.TopSpeed = rankLPD(rows, func(r LPDRow) float64 { return r.MSpeed }, 12)
	out.TopAvail = rankLPD(rows, func(r LPDRow) float64 { return r.MAvail }, 12)
	out.TopMix = rankLPD(rows, func(r LPDRow) float64 { return r.Mix }, 12)
	out.GoldStd, out.GoldModes = goldStandard(rows)
	out.LeanChamp, out.Lean, out.LeanByArch = leanStandard(rows)
	return out
}

func learnerPeaks(pts []Sample, peakAcc float64) (thru, avail float64) {
	floor := peakAcc * LPDKeepFloor
	first := true
	for _, p := range pts {
		if peakAcc > 0 && p.Acc < floor {
			continue
		}
		if first || p.Thru > thru {
			thru = p.Thru
		}
		if first || p.Avail > avail {
			avail = p.Avail
		}
		first = false
	}
	return thru, avail
}

func lpdChampOf(p Sample) LPDChamp {
	return LPDChamp{
		ID: p.ID, Tide: p.Tide, Mode: p.Mode, DType: p.DType, Arch: p.Arch,
		Score: p.Score, Soft: p.Soft, Acc: p.Acc, Thru: p.Thru, Avail: p.Avail, RAMKiB: p.RAMKiB,
	}
}

func goldStandard(rows []LPDRow) (LPDRow, []LPDMode) {
	var keep []LPDRow
	for _, r := range rows {
		if r.RelAcc >= LPDGoldKeep && r.Pillars >= 2 {
			keep = append(keep, r)
		}
	}
	if len(keep) == 0 {
		return LPDRow{}, nil
	}
	sort.SliceStable(keep, func(i, j int) bool {
		if keep[i].RAMKiB != keep[j].RAMKiB {
			return keep[i].RAMKiB < keep[j].RAMKiB
		}
		if keep[i].Thru != keep[j].Thru {
			return keep[i].Thru > keep[j].Thru
		}
		return keep[i].Acc > keep[j].Acc
	})
	std := keep[0]
	type agg struct {
		n                               int
		bestAcc, maxThru, bestQ, minRAM float64
		smallest, fastest               string
	}
	by := map[string]*agg{}
	order := []string{}
	for _, r := range keep {
		a := by[r.Mode]
		if a == nil {
			a = &agg{minRAM: r.RAMKiB, smallest: r.ID, fastest: r.ID}
			by[r.Mode] = a
			order = append(order, r.Mode)
		}
		a.n++
		if r.Acc > a.bestAcc {
			a.bestAcc = r.Acc
		}
		if r.Q > a.bestQ {
			a.bestQ = r.Q
		}
		if r.RAMKiB < a.minRAM {
			a.minRAM, a.smallest = r.RAMKiB, r.ID
		}
		if r.Thru > a.maxThru {
			a.maxThru, a.fastest = r.Thru, r.ID
		}
	}
	modes := make([]LPDMode, 0, len(order))
	for _, m := range order {
		a := by[m]
		modes = append(modes, LPDMode{
			Mode: m, N: a.n, BestAcc: a.bestAcc, MinRAM: a.minRAM, MaxThru: a.maxThru,
			BestQ: a.bestQ, Smallest: a.smallest, Fastest: a.fastest,
		})
	}
	sort.SliceStable(modes, func(i, j int) bool {
		if modes[i].MinRAM != modes[j].MinRAM {
			return modes[i].MinRAM < modes[j].MinRAM
		}
		return modes[i].MaxThru > modes[j].MaxThru
	})
	if len(modes) > 12 {
		modes = modes[:12]
	}
	return std, modes
}

// leanStandard: Acc keep ≥95% of Acc champ, then smallest RAM, fastest Thru, best Avail.
// This is the "sacrifice a little Acc for footprint/speed" board — below 95% keep is out.
func leanStandard(rows []LPDRow) (LPDRow, []LPDRow, []LPDMode) {
	var keep []LPDRow
	for _, r := range rows {
		if r.RelAcc >= LPDLeanKeep {
			keep = append(keep, r)
		}
	}
	if len(keep) == 0 {
		return LPDRow{}, nil, nil
	}
	sort.SliceStable(keep, func(i, j int) bool {
		a, b := keep[i], keep[j]
		if a.RAMKiB != b.RAMKiB {
			return a.RAMKiB < b.RAMKiB
		}
		if a.Thru != b.Thru {
			return a.Thru > b.Thru
		}
		if a.Avail != b.Avail {
			return a.Avail > b.Avail
		}
		return a.Acc > b.Acc
	})
	champ := keep[0]
	list := keep
	if len(list) > 24 {
		list = list[:24]
	}
	// Per-arch lean winner (cam differences).
	byArch := map[string]LPDRow{}
	order := []string{}
	for _, r := range keep {
		prev, ok := byArch[r.Arch]
		if !ok {
			byArch[r.Arch] = r
			order = append(order, r.Arch)
			continue
		}
		better := r.RAMKiB < prev.RAMKiB ||
			(r.RAMKiB == prev.RAMKiB && r.Thru > prev.Thru) ||
			(r.RAMKiB == prev.RAMKiB && r.Thru == prev.Thru && r.Avail > prev.Avail)
		if better {
			byArch[r.Arch] = r
		}
	}
	sort.Strings(order)
	archModes := make([]LPDMode, 0, len(order))
	for _, arch := range order {
		r := byArch[arch]
		archModes = append(archModes, LPDMode{
			Mode: arch, N: 1, BestAcc: r.Acc, MinRAM: r.RAMKiB, MaxThru: r.Thru,
			BestQ: r.Q, Smallest: r.ID, Fastest: r.ID,
		})
	}
	return champ, list, archModes
}

func rankLPD(rows []LPDRow, val func(LPDRow) float64, max int) []LPDRow {
	cp := append([]LPDRow(nil), rows...)
	sort.SliceStable(cp, func(i, j int) bool {
		if val(cp[i]) != val(cp[j]) {
			return val(cp[i]) > val(cp[j])
		}
		return cp[i].Q > cp[j].Q
	})
	if len(cp) > max {
		cp = cp[:max]
	}
	return cp
}

func lpdRow(p Sample, board LPD) LPDRow {
	rel := func(v, peak float64) float64 {
		if peak <= 0 {
			return 1
		}
		x := v / peak
		if x < 0 {
			return 0
		}
		if x > 1 {
			return 1
		}
		return x
	}
	rs, rso := rel(p.Score, board.PeakScore), rel(p.Soft, board.PeakSoft)
	ra := rel(p.Acc, board.PeakAcc)
	rt := rel(p.Thru, board.PeakThru)
	rv := rel(p.Avail, board.PeakAvail)
	q := geomean3(ra, rt, rv)
	relFast := rel(p.Thru, board.FastThru)
	relDuty := rel(p.Avail, board.BestAvail)
	ram := p.RAMKiB
	if ram <= 0 {
		ram = 1e-6
	}
	ref := board.AccChamp.RAMKiB
	if ref <= 0 {
		ref = 1e-6
	}
	frac := ram / ref
	shrink := ref / ram
	if shrink > LPDShrinkCap {
		shrink = LPDShrinkCap
	}
	lpd, mspeed, mavail, mix := 0.0, 0.0, 0.0, 0.0
	da, dt, dv := 0.0, 0.0, 0.0
	accKeep := ra >= LPDKeepFloor
	if accKeep {
		lpd = q * shrink
		mspeed, mavail, mix = rt, rv, q
		da, dt, dv = ra*shrink, rt*shrink, rv*shrink
	}
	pillars := 0
	if ra >= LPDGoldKeep {
		pillars++
	}
	if rt >= LPDGoldKeep {
		pillars++
	}
	if rv >= LPDGoldKeep {
		pillars++
	}
	pair := ra >= LPDGoldKeep && pillars >= 2
	trifecta := pillars >= 3
	gold := trifecta && frac <= LPDGoldRAM
	band := "—"
	switch {
	case gold:
		band = "gold"
	case pair && frac <= LPDNearRAM:
		band = "near"
	case pair:
		band = "keep"
	case ra >= LPDGoldKeep:
		band = "acc"
	case frac <= LPDGoldRAM && !accKeep:
		band = "trap"
	}
	return LPDRow{
		Tide: p.Tide, ID: p.ID, Mode: p.Mode, DType: p.DType, Format: p.Format, Arch: p.Arch,
		Score: p.Score, Soft: p.Soft, Acc: p.Acc, Thru: p.Thru, Avail: p.Avail, RAMKiB: p.RAMKiB,
		RelScore: rs, RelSoft: rso, RelAcc: ra, RelThru: rt, RelAvail: rv,
		Q: q, RAMFrac: frac, Shrink: shrink, LPD: lpd, Band: band, Gold: gold, Pillars: pillars,
		RelFast: relFast, RelDuty: relDuty, MSpeed: mspeed, MAvail: mavail, Mix: mix,
		DensAcc: da, DensThru: dt, DensAvail: dv,
	}
}

func geomean3(a, b, c float64) float64 {
	const eps = 1e-6
	if a < eps {
		a = eps
	}
	if b < eps {
		b = eps
	}
	if c < eps {
		c = eps
	}
	return math.Pow(a*b*c, 1.0/3.0)
}
