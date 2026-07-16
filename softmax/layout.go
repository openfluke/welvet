package softmax

import (
	"fmt"

	"github.com/openfluke/welvet/core"
)

type layout struct {
	shape  []int
	n      int
	rows   int
	cols   int
	temp   float64
}

func parseLayout[T core.Numeric](cfg Config, input *core.Tensor[T]) (layout, error) {
	var lay layout
	if input == nil || len(input.Data) == 0 {
		return lay, fmt.Errorf("softmax: empty input")
	}
	lay.shape = append([]int(nil), input.Shape...)
	lay.n = input.Len()
	lay.temp = cfg.Temperature
	if lay.temp == 0 {
		lay.temp = 1
	}

	switch cfg.Kind {
	case KindGrid:
		lay.cols = cfg.Cols
		if lay.cols <= 0 {
			lay.cols = cfg.Dim
		}
		lay.rows = cfg.Rows
		if lay.rows <= 0 {
			if lay.n%lay.cols != 0 {
				return lay, fmt.Errorf("softmax: grid n=%d not divisible by cols=%d", lay.n, lay.cols)
			}
			lay.rows = lay.n / lay.cols
		}
		if lay.rows*lay.cols > lay.n {
			return lay, fmt.Errorf("softmax: grid %d×%d > n=%d", lay.rows, lay.cols, lay.n)
		}
	default:
		// Standard: Softmax over last axis.
		if len(input.Shape) == 0 {
			return lay, fmt.Errorf("softmax: scalar input")
		}
		lay.cols = input.Shape[len(input.Shape)-1]
		if cfg.Dim > 0 && lay.cols != cfg.Dim {
			return lay, fmt.Errorf("softmax: last dim %d != Dim %d", lay.cols, cfg.Dim)
		}
		lay.rows = lay.n / lay.cols
		if lay.rows*lay.cols != lay.n {
			return lay, fmt.Errorf("softmax: n=%d not divisible by cols=%d", lay.n, lay.cols)
		}
	}
	return lay, nil
}
