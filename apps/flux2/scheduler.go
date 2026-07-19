package flux2

import (
	"fmt"
	"math"
)

// FlowMatchEulerDiscreteScheduler mirrors diffusers FlowMatchEulerDiscreteScheduler
// enough for 4-step Klein sampling (shift=3.0 + use_dynamic_shifting).
type FlowMatchEulerDiscreteScheduler struct {
	NumTrainTimesteps int
	Shift             float64
	UseDynamicShifting bool
	TimeShiftType     string // "exponential" or "linear"

	SigmaMin float64
	SigmaMax float64

	Timesteps []float64 // length = num_inference_steps (sigma * num_train)
	Sigmas    []float64 // length = num_inference_steps + 1 (terminal 0 appended)

	stepIndex  int
	beginIndex int
	hasBegin   bool
}

// NewFlowMatchEulerDiscreteScheduler builds a scheduler with Klein-friendly defaults.
func NewFlowMatchEulerDiscreteScheduler(shift float64, useDynamicShifting bool) *FlowMatchEulerDiscreteScheduler {
	const n = 1000
	sigmas := make([]float64, n)
	for i := 0; i < n; i++ {
		// linspace(1, n, n)[::-1] / n
		t := float64(n - i)
		sigmas[i] = t / float64(n)
	}
	s := &FlowMatchEulerDiscreteScheduler{
		NumTrainTimesteps:  n,
		Shift:              shift,
		UseDynamicShifting: useDynamicShifting,
		TimeShiftType:      "exponential",
		SigmaMax:           sigmas[0],
		SigmaMin:           sigmas[n-1],
	}
	if !useDynamicShifting {
		for i, sigma := range sigmas {
			sigmas[i] = shift * sigma / (1 + (shift-1)*sigma)
		}
	}
	s.Sigmas = append([]float64(nil), sigmas...)
	s.Timesteps = make([]float64, len(sigmas))
	for i, sigma := range sigmas {
		s.Timesteps[i] = sigma * float64(n)
	}
	return s
}

// SetBeginIndex sets the first step index (img2img); 0 for txt2img.
func (s *FlowMatchEulerDiscreteScheduler) SetBeginIndex(i int) {
	s.beginIndex = i
	s.hasBegin = true
}

// ComputeEmpiricalMu matches pipeline_flux2_klein.compute_empirical_mu.
func ComputeEmpiricalMu(imageSeqLen, numSteps int) float64 {
	const (
		a1 = 8.73809524e-05
		b1 = 1.89833333
		a2 = 0.00016927
		b2 = 0.45666666
	)
	seq := float64(imageSeqLen)
	if imageSeqLen > 4300 {
		return a2*seq + b2
	}
	m200 := a2*seq + b2
	m10 := a1*seq + b1
	a := (m200 - m10) / 190.0
	b := m200 - 200.0*a
	return a*float64(numSteps) + b
}

// SetTimesteps prepares the inference schedule.
// When UseDynamicShifting, mu must be provided (see ComputeEmpiricalMu).
func (s *FlowMatchEulerDiscreteScheduler) SetTimesteps(numInferenceSteps int, mu float64) error {
	if numInferenceSteps <= 0 {
		return fmt.Errorf("SetTimesteps: numInferenceSteps must be > 0")
	}

	// linspace(sigma_max*n, sigma_min*n, steps) / n
	tMax := s.SigmaMax * float64(s.NumTrainTimesteps)
	tMin := s.SigmaMin * float64(s.NumTrainTimesteps)
	sigmas := make([]float64, numInferenceSteps)
	if numInferenceSteps == 1 {
		sigmas[0] = tMax / float64(s.NumTrainTimesteps)
	} else {
		for i := 0; i < numInferenceSteps; i++ {
			t := tMax + (tMin-tMax)*float64(i)/float64(numInferenceSteps-1)
			sigmas[i] = t / float64(s.NumTrainTimesteps)
		}
	}

	if s.UseDynamicShifting {
		for i, sigma := range sigmas {
			sigmas[i] = s.timeShift(mu, 1.0, sigma)
		}
	} else {
		sh := s.Shift
		for i, sigma := range sigmas {
			sigmas[i] = sh * sigma / (1 + (sh-1)*sigma)
		}
	}

	timesteps := make([]float64, numInferenceSteps)
	for i, sigma := range sigmas {
		timesteps[i] = sigma * float64(s.NumTrainTimesteps)
	}
	// append terminal sigma 0
	s.Sigmas = append(sigmas, 0)
	s.Timesteps = timesteps
	s.stepIndex = -1
	s.hasBegin = false
	return nil
}

func (s *FlowMatchEulerDiscreteScheduler) timeShift(mu, sigma, t float64) float64 {
	switch s.TimeShiftType {
	case "linear":
		return mu / (mu + math.Pow(1/t-1, sigma))
	default: // exponential
		return math.Exp(mu) / (math.Exp(mu) + math.Pow(1/t-1, sigma))
	}
}

// Step performs one Euler flow-match update:
//
//	latents = latents + (sigma_next - sigma) * noise_pred
func (s *FlowMatchEulerDiscreteScheduler) Step(noisePred, latents []float32, timestep float64) ([]float32, error) {
	if len(s.Sigmas) < 2 || len(s.Timesteps) == 0 {
		return nil, fmt.Errorf("Step: call SetTimesteps first")
	}
	if len(noisePred) != len(latents) {
		return nil, fmt.Errorf("Step: noisePred/latents length mismatch")
	}
	if s.stepIndex < 0 {
		s.stepIndex = s.indexForTimestep(timestep)
	}
	if s.stepIndex+1 >= len(s.Sigmas) {
		return nil, fmt.Errorf("Step: step index %d out of range", s.stepIndex)
	}
	sigma := s.Sigmas[s.stepIndex]
	sigmaNext := s.Sigmas[s.stepIndex+1]
	dt := sigmaNext - sigma
	out := make([]float32, len(latents))
	for i := range latents {
		out[i] = latents[i] + float32(dt)*noisePred[i]
	}
	s.stepIndex++
	return out, nil
}

func (s *FlowMatchEulerDiscreteScheduler) indexForTimestep(timestep float64) int {
	best := 0
	bestDiff := math.Abs(s.Timesteps[0] - timestep)
	for i := 1; i < len(s.Timesteps); i++ {
		d := math.Abs(s.Timesteps[i] - timestep)
		if d < bestDiff {
			bestDiff = d
			best = i
		}
	}
	return best
}

// ResetStepIndex clears the internal step counter (call before a new Generate).
func (s *FlowMatchEulerDiscreteScheduler) ResetStepIndex() { s.stepIndex = -1 }
