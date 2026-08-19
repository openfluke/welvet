package parallel

import (
	"fmt"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/quant"
)

// View is a zero-weight reshape. Same number of elements; used so a cameral
// hemisphere (CNN / LSTM / MHA / …) can sit on a Dense stem's [batch, hidden]
// vector without changing Sandwich depth.
type View struct {
	Shape []int
}

// NewView copies shape. Product of dims must match the tensor length at forward.
func NewView(shape ...int) (*View, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("parallel: View needs a shape")
	}
	n := 1
	out := make([]int, len(shape))
	for i, d := range shape {
		if d <= 0 {
			return nil, fmt.Errorf("parallel: View dim %d <= 0", d)
		}
		out[i] = d
		n *= d
	}
	if n <= 0 {
		return nil, fmt.Errorf("parallel: View empty")
	}
	return &View{Shape: out}, nil
}

func (v *View) elems() int {
	if v == nil {
		return 0
	}
	n := 1
	for _, d := range v.Shape {
		n *= d
	}
	return n
}

// ForwardView copies input into Shape. pre is the original (for backward shape).
func ForwardView[T core.Numeric](v *View, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if v == nil || input == nil {
		return nil, nil, fmt.Errorf("parallel: nil View/input")
	}
	need := v.elems()
	if input.Len() != need {
		return nil, nil, fmt.Errorf("parallel: View shape %v needs %d elems, got %v (%d)",
			v.Shape, need, input.Shape, input.Len())
	}
	pre = core.NewTensor[T](input.Shape...)
	copy(pre.Data, input.Data)
	post = core.NewTensor[T](v.Shape...)
	copy(post.Data, input.Data)
	return pre, post, nil
}

// BackwardView copies gradOut into the input rank. No weights.
func BackwardView[T core.Numeric](v *View, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = v
	_ = pre
	if input == nil {
		return nil, nil, fmt.Errorf("parallel: View bwd nil input")
	}
	gradIn = core.NewTensor[T](input.Shape...)
	if gradOut != nil {
		n := gradIn.Len()
		if gradOut.Len() < n {
			n = gradOut.Len()
		}
		copy(gradIn.Data, gradOut.Data[:n])
	}
	return gradIn, nil, nil
}

func (v *View) Pack(format quant.Format) error {
	_ = format
	if v == nil {
		return fmt.Errorf("parallel: nil View")
	}
	return nil
}

func (v *View) SetDType(dt core.DType) error {
	_ = dt
	if v == nil {
		return fmt.Errorf("parallel: nil View")
	}
	return nil
}

func (v *View) GradWSize() int { return 0 }

// Flatten is a zero-weight reshape to [batch, Feat] so a 4D CNN stem can
// sit in front of Dense on a step line without collapsing the CNN into one tick.
type Flatten struct {
	Feat int
}

// ForwardFlatten copies input into [batch, Feat]. batch is input.Shape[0].
func ForwardFlatten[T core.Numeric](v *Flatten, input *core.Tensor[T]) (pre, post *core.Tensor[T], err error) {
	if v == nil || input == nil {
		return nil, nil, fmt.Errorf("parallel: nil Flatten/input")
	}
	batch := 1
	if len(input.Shape) > 0 {
		batch = input.Shape[0]
	}
	feat := v.Feat
	if feat <= 0 {
		if batch <= 0 || input.Len()%batch != 0 {
			return nil, nil, fmt.Errorf("parallel: Flatten cannot infer feat from shape %v", input.Shape)
		}
		feat = input.Len() / batch
	}
	if input.Len() != batch*feat {
		return nil, nil, fmt.Errorf("parallel: Flatten feat %d batch %d, got %v (%d)", feat, batch, input.Shape, input.Len())
	}
	pre = core.NewTensor[T](input.Shape...)
	copy(pre.Data, input.Data)
	post = core.NewTensor[T](batch, feat)
	copy(post.Data, input.Data)
	return pre, post, nil
}

// BackwardFlatten copies gradOut into the input rank.
func BackwardFlatten[T core.Numeric](v *Flatten, gradOut, input, pre *core.Tensor[T]) (gradIn, gradW *core.Tensor[T], err error) {
	_ = v
	_ = pre
	if input == nil {
		return nil, nil, fmt.Errorf("parallel: Flatten bwd nil input")
	}
	gradIn = core.NewTensor[T](input.Shape...)
	if gradOut != nil {
		n := gradIn.Len()
		if gradOut.Len() < n {
			n = gradOut.Len()
		}
		copy(gradIn.Data, gradOut.Data[:n])
	}
	return gradIn, nil, nil
}

func (v *Flatten) Pack(format quant.Format) error {
	_ = format
	if v == nil {
		return fmt.Errorf("parallel: nil Flatten")
	}
	return nil
}

func (v *Flatten) SetDType(dt core.DType) error {
	_ = dt
	if v == nil {
		return fmt.Errorf("parallel: nil Flatten")
	}
	return nil
}

func (v *Flatten) GradWSize() int { return 0 }
