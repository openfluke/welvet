package kmeans

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

// OutputMode selects assignment probabilities vs soft reconstruction.
type OutputMode string

const (
	OutputProbabilities OutputMode = "probabilities"
	OutputFeatures      OutputMode = "features"
)

// Config is soft K-Means geometry.
type Config struct {
	NumClusters int
	FeatureDim  int
	Temperature float64 // σ; 0 → 1
	OutputMode  OutputMode
	Activation  core.ActivationType
}

// Validate fills defaults.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("kmeans: nil config")
	}
	if c.NumClusters <= 0 || c.FeatureDim <= 0 {
		return fmt.Errorf("kmeans: need positive NumClusters/FeatureDim")
	}
	if c.Temperature < 0 {
		return fmt.Errorf("kmeans: Temperature < 0")
	}
	if c.OutputMode == "" {
		c.OutputMode = OutputProbabilities
	}
	switch c.OutputMode {
	case OutputProbabilities, OutputFeatures:
	default:
		return fmt.Errorf("kmeans: unknown OutputMode %q", c.OutputMode)
	}
	return nil
}

func (c Config) temp() float64 {
	if c.Temperature == 0 {
		return 1
	}
	return c.Temperature
}

// OutDim is K (probs) or FeatureDim (features).
func (c Config) OutDim() int {
	if c.OutputMode == OutputFeatures {
		return c.FeatureDim
	}
	return c.NumClusters
}
