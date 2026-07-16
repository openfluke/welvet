package core

// Numeric is the Go host type constraint for tensor payloads.
// Low-bit / k-quant weights live in quant.Blob, not necessarily in Tensor[T].
type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~uintptr |
		~float32 | ~float64 |
		~complex64 | ~complex128
}

// Tensor is a row-major host tensor.
type Tensor[T Numeric] struct {
	Shape []int
	Data  []T
}

// NewTensor allocates a zero tensor with the given shape (product of dims).
func NewTensor[T Numeric](shape ...int) *Tensor[T] {
	n := 1
	for _, d := range shape {
		if d <= 0 {
			n = 0
			break
		}
		n *= d
	}
	return &Tensor[T]{Shape: append([]int(nil), shape...), Data: make([]T, n)}
}

// Len returns len(Data).
func (t *Tensor[T]) Len() int {
	if t == nil {
		return 0
	}
	return len(t.Data)
}
