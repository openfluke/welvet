package evaluation

import (
	"fmt"
	"math"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/runtime/forward"
)

// TrainingMetrics captures performance metrics for a training run.
type TrainingMetrics struct {
	Steps        int                   `json:"steps"`
	Accuracy     float64               `json:"accuracy"`
	Loss         float64               `json:"loss"`
	TimeTotal    time.Duration         `json:"time_total"`
	TimeToTarget time.Duration         `json:"time_to_target"`
	MemoryPeakMB float64               `json:"memory_peak_mb"`
	Milestones   map[int]time.Duration `json:"milestones"`
}

// NewTrainingMetrics creates an initialized TrainingMetrics.
func NewTrainingMetrics() TrainingMetrics {
	milestones := make(map[int]time.Duration)
	for i := 10; i <= 100; i += 10 {
		milestones[i] = 0
	}
	return TrainingMetrics{Milestones: milestones}
}

// ComparisonResult holds results from comparing multiple training methods.
type ComparisonResult struct {
	Name      string                     `json:"name"`
	NumLayers int                        `json:"num_layers"`
	Methods   map[string]TrainingMetrics `json:"methods"`
}

// NewComparisonResult initializes a ComparisonResult.
func NewComparisonResult(name string, numLayers int) *ComparisonResult {
	return &ComparisonResult{Name: name, NumLayers: numLayers, Methods: make(map[string]TrainingMetrics)}
}

// DetermineBest returns the name of the best performing training method.
func (cr *ComparisonResult) DetermineBest() string {
	bestName := ""
	bestAcc := -1.0
	bestLoss := math.MaxFloat64
	for name, m := range cr.Methods {
		if m.Accuracy > bestAcc+1 {
			bestAcc = m.Accuracy
			bestLoss = m.Loss
			bestName = name
		} else if math.Abs(m.Accuracy-bestAcc) <= 1 && m.Loss < bestLoss {
			bestLoss = m.Loss
			bestName = name
		}
	}
	return bestName + " ✓"
}

// DeviationBucket represents a deviation percentage range.
type DeviationBucket struct {
	RangeMin float64 `json:"range_min"`
	RangeMax float64 `json:"range_max"`
	Count    int     `json:"count"`
	Samples  []int   `json:"samples"`
}

// PredictionResult represents model performance on a single prediction.
type PredictionResult struct {
	SampleIndex    int     `json:"sample_index"`
	ExpectedOutput float64 `json:"expected"`
	ActualOutput   float64 `json:"actual"`
	Deviation      float64 `json:"deviation"`
	Bucket         string  `json:"bucket"`
}

// DeviationMetrics stores the model performance breakdown.
type DeviationMetrics struct {
	Buckets          map[string]*DeviationBucket `json:"buckets"`
	Score            float64                     `json:"score"`
	TotalSamples     int                         `json:"total_samples"`
	Failures         int                         `json:"failures"`
	Results          []PredictionResult          `json:"results"`
	AverageDeviation float64                     `json:"avg_deviation"`
	CorrectCount     int                         `json:"correct_count"`
	Accuracy         float64                     `json:"accuracy"`
}

// NewDeviationMetrics initializes empty metrics.
func NewDeviationMetrics() *DeviationMetrics {
	return &DeviationMetrics{
		Buckets: map[string]*DeviationBucket{
			"0-10%": {0, 10, 0, []int{}}, "10-20%": {10, 20, 0, []int{}},
			"20-30%": {20, 30, 0, []int{}}, "30-40%": {30, 40, 0, []int{}},
			"40-50%": {40, 50, 0, []int{}}, "50-100%": {50, 100, 0, []int{}},
			"100%+": {100, math.Inf(1), 0, []int{}},
		},
		Results: []PredictionResult{},
	}
}

// EvaluatePrediction categorizes expected vs actual results.
func EvaluatePrediction(sampleIndex int, expected, actual float64) PredictionResult {
	var deviation float64
	if math.Abs(expected) < 1e-10 {
		deviation = math.Abs(actual-expected) * 100
	} else {
		deviation = math.Abs((actual - expected) / expected * 100)
	}
	if math.IsNaN(deviation) || math.IsInf(deviation, 0) {
		deviation = 100
	}
	var bucketName string
	switch {
	case deviation <= 10:
		bucketName = "0-10%"
	case deviation <= 20:
		bucketName = "10-20%"
	case deviation <= 30:
		bucketName = "20-30%"
	case deviation <= 40:
		bucketName = "30-40%"
	case deviation <= 50:
		bucketName = "40-50%"
	case deviation <= 100:
		bucketName = "50-100%"
	default:
		bucketName = "100%+"
	}
	return PredictionResult{
		SampleIndex: sampleIndex, ExpectedOutput: expected,
		ActualOutput: actual, Deviation: deviation, Bucket: bucketName,
	}
}

// UpdateMetrics adds one prediction to the metrics.
func (dm *DeviationMetrics) UpdateMetrics(result PredictionResult) {
	bucket := dm.Buckets[result.Bucket]
	bucket.Count++
	bucket.Samples = append(bucket.Samples, result.SampleIndex)
	dm.TotalSamples++
	if result.Bucket == "100%+" {
		dm.Failures++
	}
	dm.Results = append(dm.Results, result)
	dm.Score += math.Max(0, 100-result.Deviation)
	if result.ActualOutput == result.ExpectedOutput {
		dm.CorrectCount++
	}
}

// ComputeFinalMetrics completes the scoring.
func (dm *DeviationMetrics) ComputeFinalMetrics() {
	if dm.TotalSamples == 0 {
		return
	}
	dm.Score = math.Max(0, dm.Score/float64(dm.TotalSamples))
	totalDev := 0.0
	for _, r := range dm.Results {
		totalDev += r.Deviation
	}
	dm.AverageDeviation = totalDev / float64(dm.TotalSamples)
	dm.Accuracy = float64(dm.CorrectCount) / float64(dm.TotalSamples) * 100
}

// EvaluateNetwork evaluates a Grid across multiple float32 inputs.
func EvaluateNetwork(g *architecture.Grid, inputs []*core.Tensor[float32], expected []float64) (*DeviationMetrics, error) {
	if len(inputs) != len(expected) {
		return nil, fmt.Errorf("evaluation: length mismatch inputs=%d expected=%d", len(inputs), len(expected))
	}
	metrics := NewDeviationMetrics()
	for i, input := range inputs {
		res, err := forward.Forward(g, input)
		if err != nil {
			return nil, fmt.Errorf("sample %d: %w", i, err)
		}
		output := res.Output
		var actual float64
		if len(output.Data) == 1 {
			actual = float64(output.Data[0])
		} else {
			maxIdx, maxVal := 0, output.Data[0]
			for j := 1; j < len(output.Data); j++ {
				if output.Data[j] > maxVal {
					maxVal = output.Data[j]
					maxIdx = j
				}
			}
			actual = float64(maxIdx)
		}
		metrics.UpdateMetrics(EvaluatePrediction(i, expected[i], actual))
	}
	metrics.ComputeFinalMetrics()
	return metrics, nil
}

// MultiNetworkEvaluation benchmarks multiple grids on the same data.
func MultiNetworkEvaluation(models map[string]*architecture.Grid, inputs []*core.Tensor[float32], expected []float64) (map[string]*DeviationMetrics, error) {
	results := make(map[string]*DeviationMetrics)
	for name, g := range models {
		m, err := EvaluateNetwork(g, inputs, expected)
		if err != nil {
			return nil, fmt.Errorf("model %s: %w", name, err)
		}
		results[name] = m
	}
	return results, nil
}

// PrintMultiNetworkSummary prints a compact comparison table.
func PrintMultiNetworkSummary(results map[string]*DeviationMetrics) {
	fmt.Println("\n╔══════════════════════════════════════════════════════╗")
	fmt.Println("║           MULTI-MODEL PERFORMANCE COMPARISON            ║")
	fmt.Println("╠══════════════════════════════════════════════════════╣")
	for name, m := range results {
		fmt.Printf("║ %-22s | %7.2f%% | %5.1f | %7.2f%%   ║\n",
			name, m.Accuracy, m.Score, m.AverageDeviation)
	}
	fmt.Println("╚══════════════════════════════════════════════════════╝")
}

// EvaluateCategorical evaluates categorical performance.
func EvaluateCategorical(expected, actual []int) *DeviationMetrics {
	dm := NewDeviationMetrics()
	for i := range expected {
		dm.UpdateMetrics(EvaluatePrediction(i, float64(expected[i]), float64(actual[i])))
	}
	dm.ComputeFinalMetrics()
	return dm
}
