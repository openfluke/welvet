// Package lucy is the Lucy mid-stream adaptation measuring harness.
//
// SoftAcc / Availability / AdaptPct / Score — shared by test41-w benches,
// tide, and live_mnist. Pure measuring math; no datasets, no train loops.
//
//	SoftAcc      = 100 × (1 − |pred−target| / scale)   clamped [0,100]
//	Availability = InferMs / (InferMs + TrainMs) × 100
//	Score        = Throughput × Availability × SoftAcc / 10_000
//	AdaptPct     = mean SoftAcc in AdaptWindows after each switch marker
//	ZeroDowntime = SoftAcc × Availability / 100
//
// Sine SoftAcc uses SoftAccScaleSine (0.10). Classification SoftAcc on p(true)
// uses SoftAccScaleClass (1.0) so SoftAcc ≈ 100×p(true).
package lucy

import (
	"math"
	"time"
)

// SoftAcc scales.
const (
	SoftAccScaleSine  = 0.10 // test41 sine / Lucy legacy
	SoftAccScaleClass = 1.0  // MNIST SoftAccProb vs 1.0
)

// SoftAccScale is the sine default (alias for SoftAccScaleSine).
const SoftAccScale = SoftAccScaleSine

// ConsThreshold — window SoftAcc ≥ this counts toward Consistency (%).
const ConsThreshold = 10.0

// AdaptWindowsDefault — pulse windows after a switch folded into AdaptPct
// (tide 1s pulses / short perm runs). Long 50ms sine runs often use 10.
const AdaptWindowsDefault = 4

// Acc thresholds for time-to-accuracy tracking (hard Acc).
const (
	AccThreshold25 = 25.0
	AccThreshold50 = 50.0
)

// MaxRetainedWindows caps in-memory sparkline history.
const MaxRetainedWindows = 120

// Window is one pulse / time-slice sample.
type Window struct {
	At            time.Time     `json:"at,omitempty"`
	Outputs       int64         `json:"outputs"`
	Correct       int64         `json:"correct,omitempty"`
	TrainSteps    int64         `json:"train_steps,omitempty"`
	InferMs       float64       `json:"infer_ms"`
	TrainMs       float64       `json:"train_ms"`
	BlockedTrain  time.Duration `json:"blocked_train,omitempty"`
	Phase         string        `json:"phase,omitempty"`
	PhaseSwitches int           `json:"phase_switches"`
	Accuracy      float64       `json:"accuracy"` // hard Acc 0–100 (optional)
	SoftAcc       float64       `json:"soft_acc"` // SoftAcc 0–100
	Throughput    float64       `json:"throughput,omitempty"`
}

// Snapshot is the Lucy-style aggregate for one permutation / mode run.
type Snapshot struct {
	TotalOutputs int64         `json:"total_outputs"`
	TotalCorrect int64         `json:"total_correct"`
	TotalTrain   int64         `json:"total_train"`
	InferMs      float64       `json:"infer_ms"`
	TrainMs      float64       `json:"train_ms"`
	BlockedTrain time.Duration `json:"blocked_train"`
	Duration     time.Duration `json:"duration"`

	AvgAccuracy float64 `json:"avg_accuracy"` // hard Acc 0–100
	SoftAcc     float64 `json:"soft_acc"`     // SoftAcc 0–100 — Acc term in Score
	AdaptPct    float64 `json:"adapt_pct"`
	Stability   float64 `json:"stability"`
	Consistency float64 `json:"consistency"`

	Throughput   float64 `json:"throughput"`
	Availability float64 `json:"availability"`
	Score        float64 `json:"score"`
	ZeroDowntime float64 `json:"zero_downtime"`

	WeightBytes int64   `json:"weight_bytes"`
	WeightMiB   float64 `json:"weight_mib"`
	HeapBytes   int64   `json:"heap_bytes"`
	HeapMiB     float64 `json:"heap_mib"`

	MobileScore        float64 `json:"mobile_score"`
	MobileThroughput   float64 `json:"mobile_throughput"`
	MobileAvailability float64 `json:"mobile_availability"`
	MobileAccuracy     float64 `json:"mobile_accuracy"`

	Windows []Window `json:"windows,omitempty"`

	SoftAccBlocks []float64 `json:"soft_acc_blocks,omitempty"`
	PhaseBlocks   []string  `json:"phase_blocks,omitempty"`
	SwitchBlocks  []bool    `json:"switch_blocks,omitempty"`

	// AccuracyPulses >0 means SoftAcc was folded live; Finalize won't overwrite SoftAcc mean.
	AccuracyPulses int64 `json:"-"`

	TimeToAcc25Sec  float64 `json:"time_to_acc25_sec"`
	TimeToAcc50Sec  float64 `json:"time_to_acc50_sec"`
	AccPerSec       float64 `json:"acc_per_sec"`
	MobileAccPerSec float64 `json:"mobile_acc_per_sec"`
}

// Options tunes Finalize (AdaptWindows / ConsThreshold).
type Options struct {
	AdaptWindows  int
	ConsThreshold float64
}

func (o Options) withDefaults() Options {
	if o.AdaptWindows <= 0 {
		o.AdaptWindows = AdaptWindowsDefault
	}
	if o.ConsThreshold <= 0 {
		o.ConsThreshold = ConsThreshold
	}
	return o
}

// SoftAcc is SoftAcc for one pred/target pair at the given scale.
func SoftAcc(pred, target, scale float64) float64 {
	if scale <= 0 || math.IsNaN(pred) || math.IsInf(pred, 0) {
		return 0
	}
	a := 100 * (1 - math.Abs(pred-target)/scale)
	if a < 0 {
		return 0
	}
	if a > 100 {
		return 100
	}
	return a
}

// SoftAccOne is SoftAcc at SoftAccScaleSine (0.10) — test41 sine formula.
func SoftAccOne(pred, target float32) float64 {
	return SoftAcc(float64(pred), float64(target), SoftAccScaleSine)
}

// SoftAccProb is SoftAcc at SoftAccScaleClass (1.0) — MNIST p(true) vs 1.0.
func SoftAccProb(pred, target float32) float64 {
	return SoftAcc(float64(pred), float64(target), SoftAccScaleClass)
}

// SoftAccBatch means SoftAccOne across all elements.
func SoftAccBatch(pred, target []float32) float64 {
	n := len(pred)
	if n == 0 || len(target) < n {
		return 0
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += SoftAccOne(pred[i], target[i])
	}
	return sum / float64(n)
}

// Availability is InferMs / (InferMs+TrainMs) × 100.
func Availability(inferMs, trainMs float64) float64 {
	busy := inferMs + trainMs
	if busy <= 0 {
		return 0
	}
	a := 100 * inferMs / busy
	if a < 0 {
		return 0
	}
	if a > 100 {
		return 100
	}
	return a
}

// Score is Throughput × Availability × SoftAcc / 10_000.
func Score(throughput, availability, softAcc float64) float64 {
	s := throughput * availability * softAcc / 10000
	if math.IsNaN(s) || math.IsInf(s, 0) {
		return 0
	}
	return s
}

// ZeroDowntime is SoftAcc × Availability / 100.
func ZeroDowntime(softAcc, availability float64) float64 {
	return softAcc * availability / 100
}

// AppendWindow adds w and drops the oldest when over MaxRetainedWindows.
func AppendWindow(dst []Window, w Window) []Window {
	dst = append(dst, w)
	if len(dst) > MaxRetainedWindows {
		dst = append([]Window(nil), dst[len(dst)-MaxRetainedWindows:]...)
	}
	return dst
}

// WindowAccuracy returns 0–100 hard accuracy for a window.
func WindowAccuracy(correct, outputs int64) float64 {
	if outputs <= 0 {
		return 0
	}
	return 100 * float64(correct) / float64(outputs)
}

// Finalize computes Lucy aggregates on s (mutates in place).
func Finalize(s *Snapshot, opts ...Options) {
	if s == nil {
		return
	}
	o := Options{}
	if len(opts) > 0 {
		o = opts[0]
	}
	o = o.withDefaults()

	if s.Duration > 0 {
		s.Throughput = float64(s.TotalOutputs) / s.Duration.Seconds()
	}
	busy := s.InferMs + s.TrainMs
	if busy > 0 {
		s.Availability = 100 * s.InferMs / busy
	} else if s.Duration > 0 {
		avail := s.Duration - s.BlockedTrain
		if avail < 0 {
			avail = 0
		}
		s.Availability = float64(avail) / float64(s.Duration) * 100
	}

	if s.AccuracyPulses == 0 {
		if len(s.Windows) > 0 {
			var hard, soft float64
			for _, w := range s.Windows {
				hard += w.Accuracy
				soft += w.SoftAcc
			}
			n := float64(len(s.Windows))
			s.AvgAccuracy = hard / n
			s.SoftAcc = soft / n
		} else if s.TotalOutputs > 0 {
			s.AvgAccuracy = 100 * float64(s.TotalCorrect) / float64(s.TotalOutputs)
		}
	}

	nWin := len(s.Windows)
	blocks := s.SoftAccBlocks
	switches := s.SwitchBlocks
	if len(blocks) == 0 && nWin > 0 {
		blocks = make([]float64, nWin)
		switches = make([]bool, nWin)
		for i, w := range s.Windows {
			blocks[i] = w.SoftAcc
			switches[i] = w.PhaseSwitches > 0
		}
	}
	nBlk := len(blocks)
	if nBlk > 0 {
		mean := s.SoftAcc
		vari := 0.0
		valid := 0
		above := 0
		for _, a := range blocks {
			if math.IsNaN(a) {
				continue
			}
			d := a - mean
			vari += d * d
			valid++
			if a >= o.ConsThreshold {
				above++
			}
		}
		if valid > 0 {
			vari /= float64(valid)
			s.Stability = math.Max(0, 100-math.Sqrt(vari))
		}
		s.Consistency = float64(above) / float64(nBlk) * 100

		adaptSum, adaptN := 0.0, 0
		for i := range blocks {
			sw := false
			if i < len(switches) {
				sw = switches[i]
			} else if i < nWin {
				sw = s.Windows[i].PhaseSwitches > 0
			}
			if !sw {
				continue
			}
			for k := 0; k < o.AdaptWindows && i+k < nBlk; k++ {
				adaptSum += blocks[i+k]
				adaptN++
			}
		}
		if adaptN > 0 {
			s.AdaptPct = adaptSum / float64(adaptN)
		}
	}

	if len(s.SoftAccBlocks) == 0 && nWin > 0 {
		s.SoftAccBlocks = make([]float64, nWin)
		s.PhaseBlocks = make([]string, nWin)
		s.SwitchBlocks = make([]bool, nWin)
		for i, w := range s.Windows {
			s.SoftAccBlocks[i] = w.SoftAcc
			s.PhaseBlocks[i] = w.Phase
			s.SwitchBlocks[i] = w.PhaseSwitches > 0
		}
	}

	s.Score = Score(s.Throughput, s.Availability, s.SoftAcc)
	s.ZeroDowntime = ZeroDowntime(s.SoftAcc, s.Availability)
	s.WeightMiB = float64(s.WeightBytes) / (1024 * 1024)
	s.HeapMiB = float64(s.HeapBytes) / (1024 * 1024)
	mb := s.WeightMiB
	if mb < 1e-9 {
		mb = 1e-9
	}
	s.MobileScore = s.Score / mb
	s.MobileThroughput = s.Throughput / mb
	s.MobileAvailability = s.Availability / mb
	s.MobileAccuracy = s.SoftAcc / mb
	if s.Duration > 0 {
		s.AccPerSec = s.SoftAcc / s.Duration.Seconds()
		s.MobileAccPerSec = s.AccPerSec / mb
	}
}
